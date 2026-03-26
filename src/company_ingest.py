"""
company_ingest.py — 회사 문서 배치 인제스트 파이프라인

역할:
  - data/raw/company/ 하위 모든 PDF를 스캔
  - 이미 파싱된 파일(*.md 존재)은 파싱 스킵
  - 신규 파일만 Gemini API로 파싱 → *.md 저장
  - 파싱 직후 청킹(계층적) → ChromaDB + parent_index.json 직접 적재
  - 실패 시 최대 MAX_RETRIES회 재시도 (지수 백오프)
  - 최종 실패 파일은 failed_parse.log에 기록

사용법:
  cd src && uv run python company_ingest.py

신규 PDF 추가 시:
  data/raw/company/{폴더}/새파일.pdf 를 넣고 이 스크립트 재실행
  → 새 파일만 파싱+청킹+적재, 나머지는 스킵

청커 로직 변경 후 재청킹 (Gemini API 호출 없음):
  cd src && uv run python company_ingest.py --rechunk-only
  → *.md에서 재청킹 → ChromaDB 리셋 후 전체 재적재
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

import parent_store
from chunker import split_markdown_into_chunks
from pdf_parser import parse_single_pdf
from vector_db import get_chroma_dir, get_or_create_collection, reset_collection, upsert_chunks_to_chroma

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Paths / Config
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPANY_INPUT_DIR   = PROJECT_ROOT / "data" / "raw" / "company"
COMPANY_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "parsing_result_company"
FAILED_LOG_PATH     = COMPANY_OUTPUT_ROOT / "failed_parse.log"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
COLLECTION_NAME    = os.getenv("CHROMA_COLLECTION_NAME", "ninewatt_company")
CHROMA_DIR         = str(get_chroma_dir())

MAX_RETRIES      = 3
RETRY_BASE_DELAY = 5


# =============================================================================
# Helpers
# =============================================================================
def get_output_dir(pdf_path: Path) -> Path:
    relative_parent = pdf_path.relative_to(COMPANY_INPUT_DIR).parent
    return COMPANY_OUTPUT_ROOT / relative_parent / pdf_path.stem


def append_failed_log(pdf_path: Path, reason: str) -> None:
    FAILED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    relative  = pdf_path.relative_to(COMPANY_INPUT_DIR)
    with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] FAIL | {relative} | {reason}\n")


def _route_and_ingest(chunks: List[dict], force_reset: bool = False) -> None:
    """
    청크를 타입별로 라우팅해 적재한다.
      section / child → ChromaDB upsert
      parent          → parent_index.json 병합

    force_reset=True: ChromaDB 컬렉션을 삭제 후 재생성 (--rechunk-only 전체 재적재 시)
    """
    chroma_chunks = [c for c in chunks if c.get("chunk_type", "section") in ("section", "child")]
    parent_chunks = [c for c in chunks if c.get("chunk_type") == "parent"]
    child_map     = {c["chunk_id"]: c for c in chunks if c.get("chunk_type") == "child"}

    if force_reset:
        reset_collection(
            collection_name=COLLECTION_NAME,
            persist_dir=CHROMA_DIR,
            embedding_provider=EMBEDDING_PROVIDER,
        )

    upsert_chunks_to_chroma(
        chunks=chroma_chunks,
        collection_name=COLLECTION_NAME,
        persist_dir=CHROMA_DIR,
        embedding_provider=EMBEDDING_PROVIDER,
        default_doc_type="company",
    )

    if parent_chunks:
        if force_reset:
            # 전체 재적재: 마지막에 save_index로 한 번에 교체 (호출자가 처리)
            pass
        else:
            parent_store.merge_parents(parent_chunks, child_map)


async def parse_with_retry(pdf_path: Path) -> List:
    last_exc: Exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await parse_single_pdf(
                source_pdf=pdf_path,
                input_dir=COMPANY_INPUT_DIR,
                output_root=COMPANY_OUTPUT_ROOT,
            )
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning("[RETRY %d/%d] %s — %d초 후 재시도. 원인: %s",
                               attempt, MAX_RETRIES, pdf_path.name, delay, e)
                await asyncio.sleep(delay)
            else:
                logger.error("[FAIL] %s — %d회 시도 모두 실패. 원인: %s",
                             pdf_path.name, MAX_RETRIES, e)
    raise last_exc


# =============================================================================
# 일반 파싱 모드
# =============================================================================
async def parse_all() -> None:
    all_pdfs = sorted(COMPANY_INPUT_DIR.rglob("*.pdf"))
    if not all_pdfs:
        logger.warning("처리할 PDF 없음: %s", COMPANY_INPUT_DIR)
        return

    logger.info("총 %d개 PDF 발견", len(all_pdfs))

    parsed_files: List[str]        = []
    skipped_files: List[str]       = []
    failed_files: List[Tuple]      = []

    for pdf_path in all_pdfs:
        output_dir = get_output_dir(pdf_path)
        md_path    = output_dir / f"{pdf_path.stem}.md"

        # skip 기준: *.md 존재 여부 (파싱 완료 판단)
        if md_path.exists():
            skipped_files.append(pdf_path.name)
            logger.info("[SKIP] %s (*.md 존재)", pdf_path.name)
            continue

        logger.info("[PARSE] 시작: %s", pdf_path.name)
        try:
            chunks = await parse_with_retry(pdf_path)
            # 파싱 직후 바로 라우팅 + 적재 (단건 병합)
            _route_and_ingest(chunks, force_reset=False)
            parsed_files.append(pdf_path.name)
            logger.info("[DONE] %s → %d 청크 적재 완료", pdf_path.name, len(chunks))
        except Exception as e:
            failed_files.append((pdf_path.name, str(e)))
            append_failed_log(pdf_path, str(e))

    print("\n" + "=" * 55)
    print(f"[완료] 전체 {len(all_pdfs)}개 PDF")
    print(f"  신규 파싱+적재: {len(parsed_files)}개")
    print(f"  스킵(기존)    : {len(skipped_files)}개")
    print(f"  실패          : {len(failed_files)}개")
    if failed_files:
        print("\n[실패 목록]")
        for name, reason in failed_files:
            print(f"  x {name}: {reason}")
        print(f"\n  로그 위치: {FAILED_LOG_PATH}")
    print("=" * 55)


# =============================================================================
# 재청킹 모드 (--rechunk-only)
# =============================================================================
def rechunk_all() -> None:
    """
    기존 parse_report.json + *.md에서 재청킹한다. Gemini API 호출 없음.
    ChromaDB 컬렉션 전체 리셋 후 재적재, parent_index.json 전체 교체.
    """
    report_files = sorted(COMPANY_OUTPUT_ROOT.rglob("parse_report.json"))
    if not report_files:
        print("재청킹할 파싱 결과가 없습니다. 먼저 company_ingest.py를 실행하세요.")
        return

    logger.info("재청킹 대상: %d개 문서", len(report_files))

    all_chunks: List[dict] = []
    ok_count = fail_count = 0

    for report_path in report_files:
        output_dir = report_path.parent
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("[SKIP] parse_report.json 읽기 실패: %s | %s", report_path, e)
            fail_count += 1
            continue

        document_id = report.get("document_id", output_dir.name)
        model_name  = report.get("model", "unknown")
        source_file = report.get("source_file", "")

        md_files = list(output_dir.glob("*.md"))
        if not md_files:
            logger.warning("[SKIP] *.md 없음: %s", output_dir)
            fail_count += 1
            continue

        try:
            markdown_text = md_files[0].read_text(encoding="utf-8")
        except Exception as e:
            logger.error("[SKIP] *.md 읽기 실패: %s | %s", md_files[0], e)
            fail_count += 1
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
            fail_count += 1
            continue

        s = sum(1 for c in chunks if c.get("chunk_type", "section") == "section")
        p = sum(1 for c in chunks if c.get("chunk_type") == "parent")
        ch = sum(1 for c in chunks if c.get("chunk_type") == "child")
        logger.info("[RECHUNK] %s → %d청크 (section=%d parent=%d child=%d)",
                    document_id[:40], len(chunks), s, p, ch)
        all_chunks.extend(chunks)
        ok_count += 1

    print("\n" + "=" * 55)
    print(f"[재청킹 완료] 성공={ok_count}개 / 실패={fail_count}개")
    print("=" * 55)

    if ok_count == 0:
        print("\n재청킹 성공 파일 없음 — ChromaDB 적재 생략.")
        return

    # ChromaDB 전체 리셋 후 재적재
    print("\nChromaDB 전체 재적재 시작 (컬렉션 리셋)...")
    chroma_chunks = [c for c in all_chunks if c.get("chunk_type", "section") in ("section", "child")]
    parent_chunks = [c for c in all_chunks if c.get("chunk_type") == "parent"]
    child_map     = {c["chunk_id"]: c for c in all_chunks if c.get("chunk_type") == "child"}

    reset_collection(
        collection_name=COLLECTION_NAME,
        persist_dir=CHROMA_DIR,
        embedding_provider=EMBEDDING_PROVIDER,
    )
    upsert_chunks_to_chroma(
        chunks=chroma_chunks,
        collection_name=COLLECTION_NAME,
        persist_dir=CHROMA_DIR,
        embedding_provider=EMBEDDING_PROVIDER,
        default_doc_type="company",
    )

    # parent_index.json 전체 교체
    if parent_chunks:
        parent_store.save_index(parent_chunks, child_map)

    print(f"\n[완료] ChromaDB {len(chroma_chunks)}개 / parent_index {len(parent_chunks)}개")


# =============================================================================
# 진입점
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="회사 문서 인제스트 파이프라인")
    parser.add_argument(
        "--rechunk-only",
        action="store_true",
        help="기존 *.md에서 재청킹만 수행 (Gemini API 호출 없음). ChromaDB 리셋 후 전체 재적재.",
    )
    args = parser.parse_args()

    if args.rechunk_only:
        rechunk_all()
    else:
        asyncio.run(parse_all())


if __name__ == "__main__":
    main()
