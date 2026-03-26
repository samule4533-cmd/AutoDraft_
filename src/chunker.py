# src/chunker.py
#
# 역할: Markdown 텍스트를 RAG용 청크로 변환한다.
#
# ── 청크 타입 ──────────────────────────────────────────────────────────────────
# section : 짧은 섹션 전체 → ChromaDB 저장 대상
# parent  : 긴 섹션의 대표 맥락(intro) → JSON/in-memory 전용, ChromaDB X
# child   : 긴 섹션의 분할 조각 → ChromaDB 저장 대상
#
# ── 긴 섹션 판단 (OR 조건) ────────────────────────────────────────────────────
# 1. 본문 길이 > 1500자
# 2. 의미 문단 수 > 3개
# 3. 나열 패턴(청구항·제N항·번호 목록) 3회 이상 감지
#
# ── parent intro (adaptive) ──────────────────────────────────────────────────
# 헤더 + 문단 누적. 목표 600자, 하한 350자, 상한 800자.
# 첫 문단부터 시작해 목표 길이 도달 시 중단.
#
# ── child 청크 ID 체계 ────────────────────────────────────────────────────────
# section : {document_id}_c{section_counter}
# parent  : {document_id}_p{sec_idx}
# child   : {document_id}_p{sec_idx}_ch{child_idx}
#
# ── 반환값 ────────────────────────────────────────────────────────────────────
# section + parent + child 청크를 포함한 단일 리스트.
# company_vectordb.py에서 chunk_type으로 라우팅:
#   "section" | "child"  → ChromaDB upsert
#   "parent"             → parent_index.json 집계

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =============================================================================
# Config
# =============================================================================
CHUNKER_CONFIG = {
    # 섹션 분류 기준
    "section_max_len":       1500,  # 본문 길이 상한 (초과 시 긴 섹션)
    "max_paragraphs":           3,  # 의미 문단 수 상한 (초과 시 긴 섹션)
    "enum_min_hits":            3,  # 나열 패턴 감지 최소 횟수
    # child 분할 기준
    "group_max_len":         1300,  # child 1개 최대 길이
    "min_child_len":          200,  # child 최소 길이 (미만이면 인접 child와 병합)
    # 섹션 품질 필터
    "min_chunk_content_len":   80,  # 헤더 제외 실질 본문 최소 길이
    # parent intro 길이 (adaptive)
    "parent_intro_target":    600,  # 목표 길이
    "parent_intro_min":       350,  # 하한 (미달 시 다음 문단 강제 추가)
    "parent_intro_max":       800,  # 상한 (초과 시 강제 중단)
}

# 나열 패턴: 제1항 / 청구항 N / (1) ... / 1. 내용
_ENUM_PATTERN = re.compile(
    r"제\s*\d+\s*항"       # 제1항, 제 2 항
    r"|청구항\s+\d+"        # 청구항 1, 청구항 2
    r"|^\s*\(\d+\)\s"       # (1) 내용
    r"|^\s*\d+\.\s+\S",     # 1. 내용
    re.MULTILINE,
)


# =============================================================================
# 내부 헬퍼: 섹션 분류
# =============================================================================

def _has_enumeration_pattern(text: str) -> bool:
    """나열 패턴이 enum_min_hits 이상이면 True."""
    return len(_ENUM_PATTERN.findall(text)) >= CHUNKER_CONFIG["enum_min_hits"]


def _needs_parent_child(section_text: str, non_empty_paras: List[str]) -> bool:
    """
    긴 섹션 여부 판단 (OR 조건).
    True → parent + child 구조 생성.
    False → section 청크 1개.

    non_empty_paras: 길이 30자 이상인 의미 문단 리스트 (호출자가 전달).
    """
    if len(section_text) > CHUNKER_CONFIG["section_max_len"]:
        return True
    if len(non_empty_paras) > CHUNKER_CONFIG["max_paragraphs"]:
        return True
    if _has_enumeration_pattern(section_text):
        return True
    return False


_SENTENCE_END = re.compile(
    r"[다임음됩습]\.(?=\s|$)"   # 한국어 종결어미: 다. 임. 음. 됩. 습.
    r"|[.!?](?=\s|$)",           # 영어/기호 문장 끝
    re.MULTILINE,
)


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """
    max_chars 이하의 마지막 완결 문장 끝 위치에서 자른다.
    경계를 찾지 못하면 빈 문자열 반환 (호출자가 처리).
    """
    window = text[:max_chars]
    last_end = -1
    for m in _SENTENCE_END.finditer(window):
        last_end = m.end()
    if last_end > 0:
        return window[:last_end].rstrip()
    return ""


def _extract_parent_intro(header: str, body_paras: List[str]) -> str:
    """
    헤더 + 앞쪽 문단 누적으로 parent intro 생성.

    규칙:
    - 섹션 앞쪽 문단부터 순서대로 누적 (앞 문단에 맥락이 집중됨)
    - 문단 전체가 max_len을 초과하면:
        - 이미 min_len 이상 확보 → 깔끔하게 중단
        - min_len 미달 → 문장 끝 경계에서 잘라 추가 (문자 단위 컷 금지)
    - target에 도달하면 중단

    body_paras: 헤더 라인을 제외한 실질 문단 리스트 (순서 보존).
    """
    cfg = CHUNKER_CONFIG
    target  = cfg["parent_intro_target"]
    min_len = cfg["parent_intro_min"]
    max_len = cfg["parent_intro_max"]

    intro = header
    for para in body_paras:
        p = para.strip()
        if not p:
            continue
        candidate = intro + "\n\n" + p
        if len(candidate) > max_len:
            if len(intro) >= min_len:
                break  # 충분히 확보됨 → 깔끔하게 중단
            # min_len 미달 → 문장 경계에서 자르기 (문자 단위 컷 금지)
            space = max_len - len(intro) - 2
            truncated = _truncate_at_sentence(p, space) if space > 0 else ""
            if truncated:
                intro = intro + "\n\n" + truncated
            break
        intro = candidate
        if len(intro) >= target:
            break

    return intro.strip()


# =============================================================================
# 내부 헬퍼: 표·텍스트 분할 (기존 로직 유지)
# =============================================================================

def _is_table_paragraph(text: str) -> bool:
    """문단의 50% 이상 행이 마크다운 표 형식(| 시작)이면 표로 판단."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    return sum(1 for l in lines if l.startswith("|")) / len(lines) >= 0.5


def _split_table_paragraph(text: str, max_len: int) -> List[str]:
    """
    마크다운 표를 헤더 행 보존하며 데이터 행 단위로 분리.
    각 조각에 헤더+구분선을 반복 삽입해 독립적으로 읽을 수 있게 함.
    구분선(| --- |)이 없는 표는 원본 그대로 반환.
    """
    lines = text.strip().splitlines()
    header_lines: List[str] = []
    data_lines:   List[str] = []
    found_sep = False

    for line in lines:
        if not found_sep:
            header_lines.append(line)
            if re.match(r"^\|\s*[-:]+", line):
                found_sep = True
        else:
            data_lines.append(line)

    if not found_sep or not data_lines:
        return [text]

    header_text    = "\n".join(header_lines)
    result:        List[str] = []
    current_rows:  List[str] = []
    current_len    = len(header_text)

    for row in data_lines:
        row_len = len(row) + 1
        if current_len + row_len > max_len and current_rows:
            result.append(header_text + "\n" + "\n".join(current_rows))
            current_rows = [row]
            current_len  = len(header_text) + row_len
        else:
            current_rows.append(row)
            current_len += row_len

    if current_rows:
        result.append(header_text + "\n" + "\n".join(current_rows))

    return result or [text]


def _split_text_paragraph(text: str, max_len: int) -> List[str]:
    """
    긴 텍스트 문단 분리.
    1단계: 한국어 문장 경계(다. 임. 음. 등) 우선.
    2단계: 여전히 긴 경우 줄바꿈 기준 강제 분리.
    """
    if len(text) <= max_len:
        return [text]

    sentence_end = r"(?<=[다임음됨함있없니]\.)\s+|(?<=[.!?])\s+"
    sentences    = re.split(sentence_end, text)

    groups: List[str] = []
    buf = ""
    for sent in sentences:
        candidate = (buf + " " + sent).strip() if buf else sent
        if len(candidate) > max_len and buf:
            groups.append(buf.strip())
            buf = sent
        else:
            buf = candidate
    if buf:
        groups.append(buf.strip())

    result: List[str] = []
    for group in groups:
        if len(group) <= max_len:
            result.append(group)
            continue
        line_buf = ""
        for line in group.splitlines():
            candidate = (line_buf + "\n" + line).strip() if line_buf else line
            if len(candidate) > max_len and line_buf:
                result.append(line_buf.strip())
                line_buf = line
            else:
                line_buf = candidate
        if line_buf:
            result.append(line_buf.strip())

    return [c for c in result if c] or [text]


def _preprocess_paragraph(text: str, max_len: int) -> Tuple[List[str], bool]:
    """문단을 max_len 이하 조각으로 분리. 반환: (조각 리스트, is_table)."""
    if _is_table_paragraph(text):
        return _split_table_paragraph(text, max_len), True
    return _split_text_paragraph(text, max_len), False


# =============================================================================
# 내부 헬퍼: child 청크 생성
# =============================================================================

def _build_child_chunks(
    sec:           Dict[str, Any],
    base_metadata: Dict[str, Any],
    document_id:   str,
    sec_idx:       int,
) -> List[Dict[str, Any]]:
    """
    긴 섹션을 child 청크 리스트로 분할한다.

    처리 순서:
      1. 빈 줄 기준 문단 분리
      2. 표 → 독립 child (헤더 행 보존 분리)
         텍스트 → group_max_len 이하로 그룹핑
      3. min_child_len 미만 텍스트 child → 인접 child와 병합
      4. chunk_id 할당 ({document_id}_p{sec_idx}_ch{i})

    chunk_position / child_index / child_count / parent_chunk_id 는
    호출자(_split_markdown_into_chunks)에서 설정한다.
    """
    cfg        = CHUNKER_CONFIG
    group_max  = cfg["group_max_len"]
    min_child  = cfg["min_child_len"]
    sec_text   = sec["text"].strip()

    # ── 1단계: 문단 분리 후 표/텍스트 조각으로 분해 ─────────────────────────
    raw_pieces: List[Tuple[str, bool]] = []  # (text, is_table)
    for para in re.split(r"\n\s*\n", sec_text):
        p = para.strip()
        if not p:
            continue
        pieces, is_table = _preprocess_paragraph(p, group_max)
        for piece in pieces:
            piece = piece.strip()
            if piece:
                raw_pieces.append((piece, is_table))

    # ── 2단계: 표는 독립, 텍스트는 group_max_len 이하로 묶기 ─────────────────
    grouped: List[Tuple[str, bool]] = []
    buf:     List[str] = []
    buf_len  = 0

    for piece, is_table in raw_pieces:
        if is_table:
            if buf:
                grouped.append(("\n\n".join(buf).strip(), False))
                buf     = []
                buf_len = 0
            grouped.append((piece, True))
        else:
            if buf_len + len(piece) > group_max and buf:
                grouped.append(("\n\n".join(buf).strip(), False))
                buf     = []
                buf_len = 0
            buf.append(piece)
            buf_len += len(piece)

    if buf:
        grouped.append(("\n\n".join(buf).strip(), False))

    # ── 3단계: min_child_len 미만 텍스트 child → 앞 child와 병합 ─────────────
    # 표 child는 병합 제외 (표 구조 훼손 방지)
    merged: List[Tuple[str, bool]] = []
    for text, is_table in grouped:
        if (
            not is_table
            and merged
            and len(text) < min_child
            and not merged[-1][1]
        ):
            prev_text, _ = merged[-1]
            merged[-1]   = (prev_text + "\n\n" + text, False)
        else:
            merged.append((text, is_table))

    # ── 4단계: child 청크 객체 생성 ─────────────────────────────────────────
    children: List[Dict[str, Any]] = []
    for i, (text, is_table) in enumerate(merged):
        has_tbl = is_table or any(
            l.strip().startswith("|") for l in text.splitlines()
        )
        children.append({
            "chunk_id":   f"{document_id}_p{sec_idx}_ch{i}",
            "chunk_type": "child",
            "header":     sec["header"],
            "text":       text,
            "metadata": {
                **base_metadata,
                "chunk_type": "child",
                "has_table":  has_tbl,
                # chunk_position / child_index / child_count / parent_chunk_id
                # → 호출자에서 설정
            },
        })

    return children


# =============================================================================
# 메인 청킹 함수
# =============================================================================

def split_markdown_into_chunks(
    markdown_text: str,
    document_id:   str,
    source_pdf:    Path,
    model_name:    str,
    source_type:   str = "gemini_file_api_markdown",
) -> List[Dict[str, Any]]:
    """
    Markdown 텍스트를 RAG용 청크 리스트로 변환한다.

    반환 리스트에는 section / parent / child 청크가 모두 포함된다.
    company_vectordb.py에서 chunk_type으로 라우팅해 ChromaDB/parent_index에 분리 저장한다.

    chunk_id 체계:
      section  {document_id}_c{N}           (N: section 청크 전용 카운터)
      parent   {document_id}_p{sec_idx}     (sec_idx: 헤더 기준 섹션 순서)
      child    {document_id}_p{sec_idx}_ch{i}
    """
    cfg  = CHUNKER_CONFIG
    text = (markdown_text or "").strip()
    if not text:
        return []

    # ── 1단계: 헤더 기준 섹션 분리 ─────────────────────────────────────────
    lines:           List[str]         = text.splitlines()
    sections:        List[Dict]        = []
    current_header   = "ROOT"
    current_lines:   List[str]         = []

    def flush_section() -> None:
        nonlocal current_lines, current_header
        body = "\n".join(current_lines).strip()
        if not body:
            current_lines = []
            return

        # 헤더 라인·구분자 제거 후 실질 본문 길이 확인
        content_lines = [
            l for l in body.splitlines()
            if not re.match(r"^\s*#{1,6}\s+", l)
        ]
        meaningful = [
            l for l in content_lines
            if l.strip() and not re.match(r"^[-=_*]{2,}\s*$", l.strip())
        ]
        content_text = "\n".join(meaningful).strip()

        if content_text and len(content_text) >= cfg["min_chunk_content_len"]:
            sections.append({"header": current_header, "text": body})

        current_lines = []

    for line in lines:
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            flush_section()
            current_header = line.strip()
            current_lines.append(line)
        else:
            current_lines.append(line)
    flush_section()

    # ── 2단계: 섹션 → 청크 변환 ─────────────────────────────────────────────
    all_chunks:       List[Dict[str, Any]] = []
    section_counter   = 1   # section 청크 전용 카운터

    for sec_idx, sec in enumerate(sections, start=1):
        sec_text = sec["text"].strip()

        base_metadata: Dict[str, Any] = {
            "document_id":   document_id,
            "source_file":   source_pdf.name,
            "section_order": sec_idx,
            "header":        sec["header"],
            "source":        source_type,
            "model":         model_name,
        }

        # 문단 분리: 헤더 제외 실질 문단 (needs_parent_child + intro 추출에 공통 사용)
        raw_paras  = re.split(r"\n\s*\n", sec_text)
        body_paras = [
            p.strip() for p in raw_paras
            if p.strip() and not re.match(r"^\s*#{1,6}\s+", p.strip())
        ]
        non_empty_paras = [p for p in body_paras if len(p) > 30]

        if not _needs_parent_child(sec_text, non_empty_paras):
            # ── 짧은 섹션: section 청크 1개 ─────────────────────────────────
            all_chunks.append({
                "chunk_id":   f"{document_id}_c{section_counter}",
                "chunk_type": "section",
                "header":     sec["header"],
                "text":       sec_text,
                "metadata": {
                    **base_metadata,
                    "chunk_type":     "section",
                    "has_table":      _is_table_paragraph(sec_text),
                    "chunk_position": "only",
                },
            })
            section_counter += 1

        else:
            # ── 긴 섹션: parent + child 구조 ────────────────────────────────
            parent_id = f"{document_id}_p{sec_idx}"

            # child 청크 생성
            children    = _build_child_chunks(sec, base_metadata, document_id, sec_idx)
            child_count = len(children)

            # child 메타데이터 완성
            for i, child in enumerate(children):
                if child_count == 1:
                    pos = "only"
                elif i == 0:
                    pos = "first"
                elif i == child_count - 1:
                    pos = "last"
                else:
                    pos = "middle"
                child["metadata"].update({
                    "parent_chunk_id": parent_id,
                    "child_index":     i,
                    "child_count":     child_count,
                    "chunk_position":  pos,
                })

            # parent intro 추출 (adaptive)
            intro_text = _extract_parent_intro(sec["header"], body_paras)

            # parent 청크 ("text" 필드 없음 → ChromaDB 혼입 방지)
            parent_chunk: Dict[str, Any] = {
                "chunk_id":   parent_id,
                "chunk_type": "parent",
                "header":     sec["header"],
                "intro_text": intro_text,
                "child_ids":  [c["chunk_id"] for c in children],
                "metadata": {
                    **base_metadata,
                    "chunk_type":  "parent",
                    "child_count": child_count,
                    "has_table":   any(c["metadata"]["has_table"] for c in children),
                },
            }

            all_chunks.append(parent_chunk)
            all_chunks.extend(children)

    return all_chunks
