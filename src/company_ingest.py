"""
company_ingest.py — 회사 문서 배치 파싱 파이프라인

역할:
  - data/raw/company/ 하위 모든 PDF를 스캔
  - 이미 파싱된 파일(vector_chunks.json 존재)은 파싱 스킵
  - 신규 파일만 Gemini API로 파싱 → vector_chunks.json 저장
  - 실패 시 최대 MAX_RETRIES회 재시도 (지수 백오프)
  - 최종 실패 파일은 failed_parse.log에 기록

다음 단계:
  company_vectordb.py 실행 → ChromaDB 적재

사용법:
  cd src && uv run python company_ingest.py

신규 PDF 추가 시:
  data/raw/company/{폴더}/새파일.pdf 를 넣고 이 스크립트 재실행
  → 새 파일만 파싱, 나머지는 스킵
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from pdf_parser import parse_single_pdf

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPANY_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "company"
COMPANY_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "parsing_result_company"

FAILED_LOG_PATH = COMPANY_OUTPUT_ROOT / "failed_parse.log"

# =============================================================================
# Retry 설정
# =============================================================================
MAX_RETRIES = 3       # 최초 시도 포함 총 3회
RETRY_BASE_DELAY = 5  # 첫 재시도 대기 시간(초) — 이후 2배씩 증가: 5s → 10s → 20s


# =============================================================================
# Helpers
# =============================================================================
def get_output_dir(pdf_path: Path) -> Path:
    """입력 경로의 상대구조를 유지한 출력 경로 반환 (폴더 생성 없음)"""
    relative_parent = pdf_path.relative_to(COMPANY_INPUT_DIR).parent
    return COMPANY_OUTPUT_ROOT / relative_parent / pdf_path.stem


def append_failed_log(pdf_path: Path, reason: str) -> None:
    """실패 파일 정보를 failed_parse.log에 누적 기록"""
    FAILED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    relative = pdf_path.relative_to(COMPANY_INPUT_DIR)
    line = f"[{timestamp}] FAIL | {relative} | {reason}\n"
    with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
    logger.info("실패 기록 저장: %s", FAILED_LOG_PATH)


async def parse_with_retry(pdf_path: Path) -> List:
    """
    parse_single_pdf를 최대 MAX_RETRIES회 시도.
    실패 시 지수 백오프(RETRY_BASE_DELAY * 2^n 초) 후 재시도.
    모든 시도가 실패하면 마지막 예외를 raise.
    """
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
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 5s, 10s, 20s
                logger.warning(
                    "[RETRY %d/%d] %s — %d초 후 재시도. 원인: %s",
                    attempt, MAX_RETRIES, pdf_path.name, delay, e,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "[FAIL] %s — %d회 시도 모두 실패. 원인: %s",
                    pdf_path.name, MAX_RETRIES, e,
                )

    raise last_exc


# =============================================================================
# 메인 파싱 로직
# =============================================================================
async def parse_all() -> None:
    # -------------------------------------------------------------------------
    # 1. 모든 PDF 스캔 (하위 폴더 포함)
    # -------------------------------------------------------------------------
    all_pdfs = sorted(COMPANY_INPUT_DIR.rglob("*.pdf"))

    if not all_pdfs:
        logger.warning("처리할 PDF 없음: %s", COMPANY_INPUT_DIR)
        return

    logger.info("총 %d개 PDF 발견", len(all_pdfs))
    for p in all_pdfs:
        logger.info("  - %s", p.relative_to(COMPANY_INPUT_DIR))

    # -------------------------------------------------------------------------
    # 2. 각 PDF 처리: 기존 파싱 결과 있으면 스킵, 없으면 신규 파싱
    # -------------------------------------------------------------------------
    parsed_files: List[str] = []
    skipped_files: List[str] = []
    failed_files: List[Tuple[str, str]] = []

    for pdf_path in all_pdfs:
        chunks_path = get_output_dir(pdf_path) / "vector_chunks.json"

        if chunks_path.exists():
            skipped_files.append(pdf_path.name)
            logger.info("[SKIP] %s (이미 파싱됨)", pdf_path.name)
            continue

        logger.info("[PARSE] 시작: %s", pdf_path.name)
        try:
            chunks = await parse_with_retry(pdf_path)
            parsed_files.append(pdf_path.name)
            logger.info("[PARSE] 완료: %s (%d 청크)", pdf_path.name, len(chunks))

        except Exception as e:
            failed_files.append((pdf_path.name, str(e)))
            append_failed_log(pdf_path, str(e))

    # -------------------------------------------------------------------------
    # 3. 결과 요약 출력
    # -------------------------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"[파싱 완료] 전체 {len(all_pdfs)}개 PDF")
    print(f"  신규 파싱 : {len(parsed_files)}개")
    print(f"  스킵(기존): {len(skipped_files)}개")
    print(f"  실패      : {len(failed_files)}개")

    if failed_files:
        print("\n[실패 목록]")
        for name, reason in failed_files:
            print(f"  x {name}: {reason}")
        print(f"\n  로그 위치: {FAILED_LOG_PATH}")

    print("\n다음 단계: uv run python company_vectordb.py")
    print("=" * 55)


def main():
    asyncio.run(parse_all())


if __name__ == "__main__":
    main()
