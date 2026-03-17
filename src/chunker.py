import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# 내부 분리 헬퍼
# ---------------------------------------------------------------------------

def _is_table_paragraph(text: str) -> bool:
    """문단의 50% 이상 행이 마크다운 표 형식(| 시작)이면 표로 판단"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    return sum(1 for l in lines if l.startswith("|")) / len(lines) >= 0.5


def _split_table_paragraph(text: str, max_len: int) -> List[str]:
    """
    마크다운 표를 헤더 행 보존하며 데이터 행 단위로 분리.
    각 청크에 헤더+구분선을 반복 삽입해 컨텍스트 없이도 독립적으로 읽을 수 있게 함.
    구분선(| --- |)이 없는 표는 원본 그대로 반환.
    """
    lines = text.strip().splitlines()
    header_lines: List[str] = []
    data_lines: List[str] = []
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

    header_text = "\n".join(header_lines)
    result: List[str] = []
    current_rows: List[str] = []
    current_len = len(header_text)

    for row in data_lines:
        row_len = len(row) + 1  # +1 for \n
        if current_len + row_len > max_len and current_rows:
            result.append(header_text + "\n" + "\n".join(current_rows))
            current_rows = [row]
            current_len = len(header_text) + row_len
        else:
            current_rows.append(row)
            current_len += row_len

    if current_rows:
        result.append(header_text + "\n" + "\n".join(current_rows))

    return result or [text]


def _split_text_paragraph(text: str, max_len: int) -> List[str]:
    """
    긴 텍스트 문단을 분리:
    1단계: 한국어 문장 경계(다. 임. 음. 등) 우선
    2단계: 여전히 긴 경우 줄바꿈(\\n) 기준 강제 분리
    단어/어절 중간 절단을 최대한 방지.
    """
    if len(text) <= max_len:
        return [text]

    # 한국어 문장 끝 패턴 뒤 공백에서 분리
    sentence_end = r"(?<=[다임음됨함있없니]\.)\s+|(?<=[.!?])\s+"
    sentences = re.split(sentence_end, text)

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

    # 2단계: 여전히 긴 조각 → 줄바꿈 기준 분리
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
    """
    문단을 max_len 이하 조각으로 분리.
    반환: (조각 리스트, is_table)
    """
    if _is_table_paragraph(text):
        return _split_table_paragraph(text, max_len), True
    return _split_text_paragraph(text, max_len), False


# ---------------------------------------------------------------------------
# 메인 청킹 함수
# ---------------------------------------------------------------------------

def split_markdown_into_chunks(
    markdown_text: str,
    document_id: str,
    source_pdf: Path,
    model_name: str,
    section_max_len: int = 1500,
    group_max_len: int = 1300,
) -> List[Dict[str, Any]]:
    """
    RAG용 chunking:
    - Markdown 헤더(#, ##, ###) 기준 1차 분리
    - 긴 섹션은 문단 단위 2차 분리
      · 표(chunk_type='table'): 헤더 행 보존하며 행 그룹 분리
      · 텍스트(chunk_type='paragraph_group'): 문장 경계 → 줄바꿈 기준 분리
    - 메타데이터: has_table, chunk_position 포함
      · has_table: 청크 내 표 포함 여부 (retrieval 필터링용)
      · chunk_position: 섹션 내 위치 (only/first/middle/last) - 계층적 청킹 전환 시 활용
    """
    text = (markdown_text or "").strip()
    if not text:
        return []

    # 1단계: 헤더 기준 섹션 분리
    lines = text.splitlines()
    sections: List[Dict[str, Any]] = []
    current_header = "ROOT"
    current_lines: List[str] = []

    def flush_section() -> None:
        nonlocal current_lines, current_header
        body = "\n".join(current_lines).strip()
        if body:
            # 헤더 라인(# 으로 시작)을 제외한 실제 내용이 있는지 확인한다.
            #
            # 문제 상황:
            #   PDF에서 변환된 Markdown에 헤더만 있고 내용이 없는 섹션이 존재한다.
            #   예: "## 출원일자" 다음에 바로 다른 헤더가 오는 경우.
            #
            # 기존 코드에서 발생하는 문제:
            #   current_lines에 헤더 라인 자체도 포함되어 있어서
            #   body = "## 출원일자" 만으로도 if body 조건을 통과한다.
            #   이 청크가 벡터 DB에 들어가면 "출원일자" 관련 질문에 매칭되지만
            #   LLM에 전달되는 내용은 헤더 한 줄뿐이라 환각의 원인이 된다.
            #
            # [주의] 이 수정 후 최초 1회 벡터 DB 재적재가 필요하다.
            #   기존에 이미 적재된 헤더 전용 청크는 이 필터로 걸러지지 않는다.
            content_lines = [
                l for l in body.splitlines()
                if not re.match(r"^\s*#{1,6}\s+", l)
            ]
            if "\n".join(content_lines).strip():
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

    # 2단계: 섹션 → 청크
    all_chunks: List[Dict[str, Any]] = []
    chunk_idx = 1

    for sec_idx, sec in enumerate(sections, start=1):
        sec_text = sec["text"].strip()
        base_metadata = {
            "document_id": document_id,
            "source_file": source_pdf.name,
            "section_order": sec_idx,
            "header": sec["header"],
            "source": "gemini_file_api_markdown",
            "model": model_name,
            "mode": "fast_gemini_file_api",
        }

        # 이 섹션의 청크를 먼저 수집 → chunk_position 계산 후 추가
        sec_chunks: List[Dict[str, Any]] = []

        if len(sec_text) <= section_max_len:
            # 섹션 전체가 짧으면 1개 청크
            has_table = _is_table_paragraph(sec_text)
            sec_chunks.append({
                "chunk_id": f"{document_id}_c{chunk_idx}",
                "header": sec["header"],
                "text": sec_text,
                "metadata": {
                    **base_metadata,
                    "chunk_type": "section",
                    "has_table": has_table,
                },
            })
            chunk_idx += 1
        else:
            # 긴 섹션: 빈 줄 기준 문단 분리 후 처리
            paras = re.split(r"\n\s*\n", sec_text)
            buf: List[str] = []
            buf_len = 0

            for para in paras:
                p = para.strip()
                if not p:
                    continue

                pieces, is_table = _preprocess_paragraph(p, group_max_len)

                if is_table:
                    # 표: 쌓인 텍스트 buf 먼저 flush 후 표 청크 독립 추가
                    if buf:
                        joined = "\n\n".join(buf).strip()
                        has_tbl = any(l.strip().startswith("|") for l in joined.splitlines())
                        sec_chunks.append({
                            "chunk_id": f"{document_id}_c{chunk_idx}",
                            "header": sec["header"],
                            "text": joined,
                            "metadata": {
                                **base_metadata,
                                "chunk_type": "paragraph_group",
                                "has_table": has_tbl,
                            },
                        })
                        chunk_idx += 1
                        buf = []
                        buf_len = 0

                    for piece in pieces:
                        piece = piece.strip()
                        if not piece:
                            continue
                        sec_chunks.append({
                            "chunk_id": f"{document_id}_c{chunk_idx}",
                            "header": sec["header"],
                            "text": piece,
                            "metadata": {
                                **base_metadata,
                                "chunk_type": "table",
                                "has_table": True,
                            },
                        })
                        chunk_idx += 1
                else:
                    # 텍스트: pre-split 조각을 buf에 그룹핑
                    for piece in pieces:
                        piece = piece.strip()
                        if not piece:
                            continue
                        if buf_len + len(piece) > group_max_len and buf:
                            joined = "\n\n".join(buf).strip()
                            sec_chunks.append({
                                "chunk_id": f"{document_id}_c{chunk_idx}",
                                "header": sec["header"],
                                "text": joined,
                                "metadata": {
                                    **base_metadata,
                                    "chunk_type": "paragraph_group",
                                    "has_table": False,
                                },
                            })
                            chunk_idx += 1
                            buf = []
                            buf_len = 0
                        buf.append(piece)
                        buf_len += len(piece)

            # 남은 텍스트 buf flush
            if buf:
                joined = "\n\n".join(buf).strip()
                has_tbl = any(l.strip().startswith("|") for l in joined.splitlines())
                sec_chunks.append({
                    "chunk_id": f"{document_id}_c{chunk_idx}",
                    "header": sec["header"],
                    "text": joined,
                    "metadata": {
                        **base_metadata,
                        "chunk_type": "paragraph_group",
                        "has_table": has_tbl,
                    },
                })
                chunk_idx += 1

        # chunk_position 할당 (계층적 청킹 전환 시 parent 탐색에 활용)
        n = len(sec_chunks)
        for i, c in enumerate(sec_chunks):
            if n == 1:
                pos = "only"
            elif i == 0:
                pos = "first"
            elif i == n - 1:
                pos = "last"
            else:
                pos = "middle"
            c["metadata"]["chunk_position"] = pos

        all_chunks.extend(sec_chunks)

    return all_chunks
