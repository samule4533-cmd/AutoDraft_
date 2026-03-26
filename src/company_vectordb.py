"""
company_vectordb.py — 회사 문서 ChromaDB 적재 및 쿼리 테스트

역할:
  - data/processed/parsing_result_company/ 하위 모든 vector_chunks.json 수집
  - chunk_type별 라우팅:
      section / child → ChromaDB upsert
      parent          → data/processed/parent_index.json (원자적 저장)
  - 적재 후 테스트 쿼리 실행

이전 단계:
  company_ingest.py 실행 → PDF 파싱 완료 후 실행

사용법:
  cd src && uv run python company_vectordb.py
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

import parent_store
from chunker import split_markdown_into_chunks
from vector_db import get_chroma_dir, query_collection, print_query_summary, reset_collection, upsert_chunks_to_chroma

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPANY_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "parsing_result_company"
CHROMA_DIR = str(get_chroma_dir())

# =============================================================================
# Config (.env에서 읽음)
# =============================================================================
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "ninewatt_company")
# CHROMA_RESET=true 시 기존 컬렉션 삭제 후 재생성 (임베딩 모델 전환 시 사용)
CHROMA_RESET = os.getenv("CHROMA_RESET", "false").lower() == "true"


# =============================================================================
# Helpers
# =============================================================================
def collect_all_chunks() -> List[Dict[str, Any]]:
    """
    COMPANY_OUTPUT_ROOT 하위의 모든 parse_report.json + *.md를 읽어
    계층적 청킹을 수행하고 전체 청크를 반환한다.

    vector_chunks.json은 더 이상 사용하지 않는다.
    *.md가 파싱 완료의 유일한 기준이다.
    """
    all_chunks: List[Dict[str, Any]] = []
    report_files = sorted(COMPANY_OUTPUT_ROOT.rglob("parse_report.json"))

    if not report_files:
        logger.warning("parse_report.json 없음: %s", COMPANY_OUTPUT_ROOT)
        return all_chunks

    for report_path in report_files:
        output_dir = report_path.parent
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("[SKIP] parse_report.json 읽기 실패: %s | %s", report_path, e)
            continue

        document_id = report.get("document_id", output_dir.name)
        model_name  = report.get("model", "unknown")
        source_file = report.get("source_file", "")

        md_files = list(output_dir.glob("*.md"))
        if not md_files:
            logger.warning("[SKIP] *.md 없음: %s", output_dir)
            continue

        try:
            markdown_text = md_files[0].read_text(encoding="utf-8")
        except Exception as e:
            logger.error("[SKIP] *.md 읽기 실패: %s | %s", md_files[0], e)
            continue

        source_pdf = Path(source_file) if source_file else md_files[0].with_suffix(".pdf")

        try:
            chunks = split_markdown_into_chunks(
                markdown_text=markdown_text,
                document_id=document_id,
                source_pdf=source_pdf,
                model_name=model_name,
            )
        except Exception as e:
            logger.error("[FAIL] 청킹 실패: %s | %s", document_id, e)
            continue

        all_chunks.extend(chunks)
        logger.info("[CHUNK] %s → %d 청크", document_id[:40], len(chunks))

    return all_chunks


# =============================================================================
# 메인 적재 로직
# =============================================================================
def upsert_all(force_reset: bool = False) -> None:
    # -------------------------------------------------------------------------
    # 1. 모든 파싱 결과 수집
    # -------------------------------------------------------------------------
    all_chunks = collect_all_chunks()

    if not all_chunks:
        print("적재할 청크가 없습니다. 먼저 company_ingest.py를 실행하세요.")
        return

    # -------------------------------------------------------------------------
    # 2. chunk_type별 라우팅
    #    section / child  → ChromaDB (임베딩 대상)
    #    parent           → parent_index.json (intro_text + children 보관)
    #
    #    parent 청크는 "text" 필드가 없어 ChromaDB에 upsert해도 prepare_chroma_items()
    #    에서 자동 필터되지만, 명시적으로 분리해 의도를 명확히 한다.
    # -------------------------------------------------------------------------
    chroma_chunks = [
        c for c in all_chunks
        if c.get("chunk_type", "section") in ("section", "child")
    ]
    parent_chunks = [
        c for c in all_chunks
        if c.get("chunk_type") == "parent"
    ]
    child_map: Dict[str, Dict[str, Any]] = {
        c["chunk_id"]: c
        for c in all_chunks
        if c.get("chunk_type") == "child"
    }

    logger.info(
        "청크 라우팅: ChromaDB(section+child)=%d / parent_index=%d",
        len(chroma_chunks), len(parent_chunks),
    )

    # -------------------------------------------------------------------------
    # 2-1. parent_index.json 저장 (원자적) → parent_store 인메모리 재로드 포함
    # -------------------------------------------------------------------------
    if parent_chunks:
        parent_store.save_index(parent_chunks, child_map)
    else:
        logger.info("parent 청크 없음 (구 버전 청킹) → parent_index.json 저장 생략")

    # -------------------------------------------------------------------------
    # 2-2. ChromaDB upsert
    #      임베딩 모델 전환 시 기존 컬렉션 리셋 (CHROMA_RESET=true)
    #      local(384차원) → openai(1536차원) 등 차원이 달라지면 반드시 실행해야 함.
    # -------------------------------------------------------------------------
    if CHROMA_RESET or force_reset:
        logger.warning("컬렉션 '%s' 삭제 후 재생성 (force_reset=%s CHROMA_RESET=%s)", COLLECTION_NAME, force_reset, CHROMA_RESET)
        reset_collection(
            collection_name=COLLECTION_NAME,
            persist_dir=CHROMA_DIR,
            embedding_provider=EMBEDDING_PROVIDER,
        )

    logger.info("ChromaDB 적재 시작: %d 청크 → 컬렉션 '%s'", len(chroma_chunks), COLLECTION_NAME)
    upsert_chunks_to_chroma(
        chunks=chroma_chunks,
        collection_name=COLLECTION_NAME,
        persist_dir=CHROMA_DIR,
        embedding_provider=EMBEDDING_PROVIDER,
        default_doc_type="company",
    )

    # -------------------------------------------------------------------------
    # 3. 결과 요약 출력
    # -------------------------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"[적재 완료] ChromaDB {len(chroma_chunks)}개 / parent_index {len(parent_chunks)}개")
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  임베딩: {EMBEDDING_PROVIDER}")
    print(f"  저장 경로: {CHROMA_DIR}")
    print("=" * 55)

    # -------------------------------------------------------------------------
    # 4. 적재 sanity check — 컬렉션에 데이터가 실제로 들어갔는지 확인
    # -------------------------------------------------------------------------
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(COLLECTION_NAME)
    print(f"\n[sanity check] 컬렉션 '{COLLECTION_NAME}' 총 청크 수: {col.count()}")


if __name__ == "__main__":
    upsert_all()
