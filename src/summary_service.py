# src/summary_service.py
#
# 역할: 사내 문서 전체 AI 요약 생성 및 캐시 관리.
#
# 동작 방식:
#   - ChromaDB에서 파일별 청크를 수집
#   - Gemini 1회 호출로 전체 파일 요약 생성 (1M 토큰 컨텍스트 활용)
#   - 결과를 data/processed/summaries.json에 캐시
#   - 재호출 시 새 파일만 추가 생성, 삭제된 파일은 캐시에서 제거

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
def _load_cache() -> dict[str, str]:
    try:
        if CACHE_PATH.exists():
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            return data.get("summaries", {})
    except Exception as e:
        logger.warning("캐시 로드 실패: %s", e)
    return {}


def _save_cache(summaries: dict[str, str]) -> None:
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
) -> dict[str, str]:
    """
    Gemini 1회 호출로 target_files 전체 요약을 생성한다.
    응답은 JSON 배열로 파싱해 {filename: summary} dict로 반환한다.
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
        config=types.GenerateContentConfig(temperature=0.0),
    )
    raw = (getattr(resp, "text", "") or "").strip()

    # 코드 펜스 제거 후 JSON 파싱
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        items = json.loads(raw.strip())
        result = {item["filename"]: item["summary"] for item in items}
        logger.info("요약 생성 완료: %d개 파일", len(result))
        return result
    except Exception as e:
        logger.error("요약 JSON 파싱 실패: %s | raw=%.300s", e, raw)
        return {f: "요약 생성에 실패했습니다." for f in target_files}


# =============================================================================
# 메인 진입점
# =============================================================================
def get_summaries() -> list[dict]:
    """
    전체 문서 요약 목록을 반환한다.

    캐시 전략:
      - 새 파일 추가 시: 해당 파일만 Gemini 호출 후 캐시에 추가
      - 파일 삭제 시: 캐시에서 제거
      - 변화 없으면: Gemini 호출 없이 캐시 즉시 반환

    반환 형식: [{"index": 1, "filename": "...", "summary": "..."}]
    """
    file_chunks   = _get_file_chunks()
    current_files = set(file_chunks.keys())
    cached        = _load_cache()
    cached_files  = set(cached.keys())

    new_files     = current_files - cached_files
    removed_files = cached_files  - current_files

    if new_files:
        logger.info("신규 파일 %d개 요약 생성 시작: %s", len(new_files), new_files)
        new_summaries = _generate_summaries(file_chunks, sorted(new_files))
        cached.update(new_summaries)

    for f in removed_files:
        cached.pop(f, None)
        logger.info("삭제된 파일 캐시 제거: %s", f)

    if new_files or removed_files:
        _save_cache(cached)

    return [
        {"index": i + 1, "filename": f, "summary": cached.get(f, "요약 없음")}
        for i, f in enumerate(sorted(current_files))
    ]
