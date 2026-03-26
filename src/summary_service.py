# src/summary_service.py
#
# 역할: 사내 문서 전체 AI 요약 생성 및 캐시 관리.
#
# 동작 방식:
#   - ChromaDB에서 파일별 청크를 수집
#   - Gemini 1회 호출로 전체 파일 요약 생성 (1M 토큰 컨텍스트 활용)
#   - 결과를 data/processed/summaries.json에 캐시
#   - 재호출 시 새 파일만 추가 생성, 삭제된 파일은 캐시에서 제거
#   - 청크 수 변화 감지로 파일 내용 변경 시 재요약 트리거
#
# 캐시 형식 (v2):
#   {"summaries": {"파일명.pdf": {"summary": "요약 내용", "chunk_count": 42}}}

import json
import logging
import os
from pathlib import Path

from google.genai import types

from src.llm_api import get_client
from src.rag_chain import RAG_CONFIG
from src.vector_db import get_or_create_collection

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH   = PROJECT_ROOT / "data" / "processed" / "summaries.json"

# 파일당 사용할 최대 청크 수 (토큰 한도 대비 안전 마진)
_MAX_CHUNKS_PER_FILE = 30

_SUMMARY_PROMPT = """\
아래 문서들을 각각 3문장으로 요약하라.
파일 이름은 그대로 사용하고, 아래 JSON 형식으로만 출력하라. 다른 설명은 하지 말라.

[
  {{"filename": "파일명.pdf", "summary": "요약 내용"}},
  ...
]

{documents}
"""


# =============================================================================
# ChromaDB 수집
# =============================================================================
def _get_file_chunks() -> dict[str, list[str]]:
    """ChromaDB에서 파일별 청크 텍스트를 수집한다."""
    col = get_or_create_collection(
        persist_dir=RAG_CONFIG["persist_dir"],
        collection_name=RAG_CONFIG["collection_name"],
        embedding_provider=RAG_CONFIG["embedding_provider"],
    )
    result    = col.get(include=["documents", "metadatas"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    file_chunks: dict[str, list[str]] = {}
    for doc, meta in zip(documents, metadatas):
        filename = (meta or {}).get("source_file", "알 수 없음")
        if doc:
            file_chunks.setdefault(filename, []).append(doc)

    return file_chunks


# =============================================================================
# 캐시 I/O
# =============================================================================
def _load_cache() -> dict[str, dict]:
    """
    캐시 로드. 반환 형식: {filename: {"summary": str, "chunk_count": int}}
    구 형식(str 값)은 자동으로 {"summary": ..., "chunk_count": -1}로 마이그레이션한다.
    chunk_count=-1은 "알 수 없음"을 의미하며, 실제 청크 수와 항상 불일치 → 재요약 트리거.
    """
    try:
        if CACHE_PATH.exists():
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            raw = data.get("summaries", {})
            # 구 형식(str) → 신 형식(dict) 마이그레이션
            migrated = {}
            for fname, val in raw.items():
                if isinstance(val, str):
                    migrated[fname] = {"summary": val, "chunk_count": -1}
                else:
                    migrated[fname] = val
            return migrated
    except Exception as e:
        logger.warning("캐시 로드 실패: %s", e)
    return {}


def _save_cache(summaries: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("요약 캐시 저장: %s", CACHE_PATH)


# =============================================================================
# Gemini 호출
# =============================================================================
def _generate_summaries(
    file_chunks: dict[str, list[str]],
    target_files: list[str],
) -> dict[str, dict]:
    """
    Gemini 1회 호출로 target_files 전체 요약을 생성한다.
    반환 형식: {filename: {"summary": str, "chunk_count": int}}
    """
    doc_blocks = []
    for filename in target_files:
        chunks  = file_chunks.get(filename, [])[:_MAX_CHUNKS_PER_FILE]
        content = "\n".join(chunks)
        doc_blocks.append(f"[파일: {filename}]\n{content}")

    prompt = _SUMMARY_PROMPT.format(documents="\n\n---\n\n".join(doc_blocks))

    resp = get_client().models.generate_content(
        model=os.getenv("GEMINI_RAG_MODEL", "gemini-2.0-flash"),
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )
    raw = (getattr(resp, "text", "") or "").strip()

    # 코드 펜스 제거 후 JSON 파싱
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        items = json.loads(raw.strip())

        # Gemini가 파일명을 변형(언더스코어→공백 등)할 수 있으므로
        # 응답 파일명을 정규화(공백→언더스코어, 소문자)해서 원본 파일명과 매칭한다.
        normalized_map = {
            f.replace(" ", "_").lower(): f for f in target_files
        }

        result = {}
        for item in items:
            gemini_name = item["filename"]
            # 1차: 완전 일치
            original_name = gemini_name if gemini_name in file_chunks else None
            # 2차: 정규화 후 매칭
            if original_name is None:
                original_name = normalized_map.get(gemini_name.replace(" ", "_").lower())
            if original_name is None:
                logger.warning("요약 응답 파일명 매칭 실패, 스킵: %r", gemini_name)
                continue
            result[original_name] = {
                "summary":     item["summary"],
                "chunk_count": len(file_chunks.get(original_name, [])),
            }

        # Gemini가 일부 파일을 누락한 경우 실패 처리
        for f in target_files:
            if f not in result:
                logger.warning("요약 응답에 누락된 파일, 실패 처리: %s", f)
                result[f] = {"summary": "요약 생성에 실패했습니다.", "chunk_count": len(file_chunks.get(f, []))}

        logger.info("요약 생성 완료: %d개 파일", len(result))
        return result
    except Exception as e:
        logger.error("요약 JSON 파싱 실패: %s | raw=%.300s", e, raw)
        return {
            f: {"summary": "요약 생성에 실패했습니다.", "chunk_count": len(file_chunks.get(f, []))}
            for f in target_files
        }


# =============================================================================
# 메인 진입점
# =============================================================================
def get_summaries() -> list[dict]:
    """
    전체 문서 요약 목록을 반환한다.

    캐시 전략:
      - 새 파일 추가 시: 해당 파일만 Gemini 호출 후 캐시에 추가
      - 파일 삭제 시: 캐시에서 제거
      - 파일 내용 변경 시: 청크 수 불일치 감지 → 재요약 트리거
      - 변화 없으면: Gemini 호출 없이 캐시 즉시 반환

    반환 형식: [{"index": 1, "filename": "...", "summary": "..."}]
    """
    file_chunks   = _get_file_chunks()
    current_files = set(file_chunks.keys())
    cached        = _load_cache()
    cached_files  = set(cached.keys())

    new_files     = current_files - cached_files
    removed_files = cached_files  - current_files

    # 청크 수 변화로 내용 변경 감지 (파일명은 동일하나 내용이 바뀐 경우)
    changed_files = {
        f for f in (current_files & cached_files)
        if len(file_chunks[f]) != cached[f].get("chunk_count", -1)
    }
    if changed_files:
        logger.info("내용 변경 감지 (청크 수 불일치) %d개: %s", len(changed_files), changed_files)

    files_to_summarize = sorted(new_files | changed_files)
    if files_to_summarize:
        logger.info("요약 생성 시작 — 신규: %s, 변경: %s", new_files, changed_files)
        new_summaries = _generate_summaries(file_chunks, files_to_summarize)
        cached.update(new_summaries)

    for f in removed_files:
        cached.pop(f, None)
        logger.info("삭제된 파일 캐시 제거: %s", f)

    if files_to_summarize or removed_files:
        _save_cache(cached)

    return [
        {"index": i + 1, "filename": f, "summary": cached[f]["summary"] if isinstance(cached.get(f), dict) else cached.get(f, "요약 없음")}
        for i, f in enumerate(sorted(current_files))
    ]
