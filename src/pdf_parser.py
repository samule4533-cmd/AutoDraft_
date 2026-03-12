import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

import cleaning
import field_extract
from chunker import split_markdown_into_chunks
from image_parser import build_image_caption_chunks
from output_writer import build_document_json, save_outputs

# =============================================================================
# Logging / Env
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sample_notices"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "parsing_result_notices"

DEFAULT_PDF_NAME = os.getenv("DEFAULT_PDF_NAME", "sample1.pdf")
SOURCE_PDF = INPUT_DIR / DEFAULT_PDF_NAME

# =============================================================================
# Config
# =============================================================================
GEMINI_PDF_MODEL = os.getenv("GEMINI_PDF_MODEL", "gemini-3-flash-preview")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))

SAVE_FINAL_MD = True
SAVE_FINAL_JSON = True
SAVE_PARSE_REPORT = True
SAVE_FIELDS_JSON = True
SAVE_VECTOR_CHUNKS = True

PDF_PARSE_PROMPT = """
너는 RAG 파이프라인을 위한 최고 수준의 문서 파서야.
첨부된 PDF 문서를 읽고 레이아웃이 완벽하게 보존된 Markdown 형식으로 변환해줘.

[🔥 절대 준수 사항 - 위반 시 시스템 오류 발생]
1. 누락 금지: 문서 내의 텍스트와 '표(Table)'는 단 하나도 누락하거나 요약하지 말고 100% 모두 추출할 것.
2. 생략 금지: 표가 길거나 복잡하더라도 절대 임의로 생략(Skip)하거나 중단하지 말 것. 원본에 있는 모든 데이터를 있는 그대로 전부 출력해.
3. 문서의 제목, 목차, 본문 구조를 마크다운 헤더(#, ##, ###)로 정확히 계층화할 것.
4. 표는 마크다운 표 문법(|---|---|)을 사용하여 행과 열 구조를 원본과 동일하게 유지할 것.

[공고문/입찰문서 특화 지침]
5. 공고번호, 공고명, 기관명, 공사명/사업명, 일정(제출 개시/마감/개찰), 금액, 장소, 문의처는 특히 정확히 보존할 것.
6. 숫자, 날짜, 전화번호, 금액, 퍼센트, 고유명사는 임의로 바꾸거나 정리하지 말고 원문에 가깝게 유지할 것.
7. 본문과 붙임 문서(예: 청렴서약서, 확인서, 동의서, 서식)는 구분하여 각각 제목을 유지할 것.
8. 체크박스, 서명란, 날인란, 붙임 제목도 가능한 한 텍스트로 보존할 것.
9. 읽기 애매한 부분은 내용을 추정해 매끄럽게 바꾸기보다, 원문에 가깝게 보존할 것.

출력은 설명 없이 Markdown 본문만 반환할 것.
""".strip()


# =============================================================================
# Helpers
# =============================================================================
def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다.")
    return genai.Client(api_key=api_key)


def korean_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(re.findall(r"[가-힣]", s)) / max(len(s), 1)


def build_output_dir(source_pdf: Path) -> Path:
    out_dir = OUTPUT_ROOT / source_pdf.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def preflight_check(source_pdf: Path, output_dir: Path) -> Dict[str, Any]:
    report = {"ok": True, "checks": []}

    def _add(name: str, ok: bool, msg: str = ""):
        report["checks"].append({"name": name, "ok": ok, "msg": msg})
        if not ok:
            report["ok"] = False

    _add("pdf_exists", source_pdf.exists(), str(source_pdf))
    _add("gemini_api_key_present", bool(os.getenv("GEMINI_API_KEY", "").strip()), "env: GEMINI_API_KEY")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        testfile = output_dir / "__write_test__.tmp"
        testfile.write_text("ok", encoding="utf-8")
        testfile.unlink(missing_ok=True)
        _add("output_writable", True, str(output_dir))
    except Exception as e:
        _add("output_writable", False, f"{output_dir} / {e}")

    return report


def normalize_markdown_headings(markdown_text: str) -> str:
    """
    Markdown 정규화:
    - 줄 전체가 **제목** 형태이면 ## 제목으로 변환
    - 그 외 본문 내 **강조** 는 그대로 둔다
    """
    lines = (markdown_text or "").splitlines()
    normalized: List[str] = []

    for line in lines:
        s = line.strip()

        # 이미 markdown 헤더면 그대로 유지
        if re.match(r"^\s{0,3}#{1,6}\s+", s):
            normalized.append(line)
            continue

        # 줄 전체가 **제목** 형태인 경우만 헤더로 승격
        m = re.match(r"^\*\*([^*]{2,80})\*\*$", s)
        if m:
            normalized.append(f"## {m.group(1).strip()}")
            continue

        normalized.append(line)

    return "\n".join(normalized)


def build_final_pages_like(markdown_text: str) -> List[Dict[str, Any]]:
    """
    기존 field_extract.py를 최대한 건드리지 않기 위해
    final_pages 비슷한 구조를 한 페이지짜리로 만들어 재사용한다.
    """
    page_payload = {
        "page_number": 1,
        "engine_used": "gemini_file_api",
        "quality": {
            "text_len": len(markdown_text or ""),
            "korean_ratio": round(korean_ratio(markdown_text or ""), 4),
            "garbled": False,
        },
        "md_reason": "gemini_file_api_markdown",
        "final_text": markdown_text or "",
        "cleaned_text": "",
        "cleaned_for_fields": "",
        "cleaned_render_ocr_text": "",
        "cleaned_gemini_key_values": {},
        "docling_dict": {},
        "render_ocr_text": None,
        "gemini_page_vision": None,
        "gemini_key_values": {},
        "tables_normalized": [],
        "blocks": [],
        "images": [],
        "image_ocr": [],
        "needs_review": False,
        "needs_review_reasons": [],
        "warnings": [],
        "errors": [],
    }

    try:
        page_payload = cleaning.clean_page_payload(page_payload)
    except Exception as e:
        logger.warning("clean_page_payload 실패: %s", e)

    return [page_payload]


# =============================================================================
# Gemini File API
# =============================================================================
def _upload_and_wait(client: genai.Client, file_path: Path):
    """
    한글 파일명 업로드 이슈를 피하기 위해 영문 임시 파일명으로 복사 후 업로드
    """
    logger.info("PDF 업로드 시작: %s", file_path.name)

    suffix = file_path.suffix or ".pdf"

    with tempfile.TemporaryDirectory() as tmpdir:
        safe_path = Path(tmpdir) / f"upload_input{suffix}"
        shutil.copy2(file_path, safe_path)

        uploaded = client.files.upload(file=str(safe_path))

        while uploaded.state.name == "PROCESSING":
            logger.info("파일 처리 중... 2초 후 재확인")
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        if uploaded.state.name == "FAILED":
            raise RuntimeError(f"Gemini File API 처리 실패: {file_path.name}")

        logger.info("PDF 업로드 완료: %s", file_path.name)
        return uploaded


async def parse_pdf_to_markdown(source_pdf: Path) -> str:
    client = get_genai_client()

    uploaded_file = await asyncio.to_thread(_upload_and_wait, client, source_pdf)

    logger.info("Gemini 호출 시작: model=%s, file=%s", GEMINI_PDF_MODEL, source_pdf.name)
    response = await client.aio.models.generate_content(
        model=GEMINI_PDF_MODEL,
        contents=[uploaded_file, PDF_PARSE_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        ),
    )

    markdown_text = response.text or ""
    logger.info("Gemini 파싱 완료: %s (출력 %d자)", source_pdf.name, len(markdown_text))
    return markdown_text


# =============================================================================
# Main
# =============================================================================
async def async_main():
    source_pdf = SOURCE_PDF
    output_dir = build_output_dir(source_pdf)

    pf = preflight_check(source_pdf, output_dir)
    with open(output_dir / "preflight.json", "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)

    if not pf["ok"]:
        logger.error("❌ Pre-flight check 실패")
        for c in pf["checks"]:
            if not c["ok"]:
                logger.error("- %s: %s", c["name"], c["msg"])
        return

    t0 = time.perf_counter()

    markdown_text = await parse_pdf_to_markdown(source_pdf)

    # Markdown 정규화:
    # **입찰공고** 같은 줄 전체 볼드 제목을 ## 입찰공고 형태로 변환
    markdown_text = normalize_markdown_headings(markdown_text)

    elapsed_parse = time.perf_counter() - t0

    final_pages = build_final_pages_like(markdown_text)

    try:
        fields = {
            "bid_amount": field_extract.extract_bid_amount_from_final_pages(final_pages),
        }
    except Exception as e:
        logger.warning("fields 추출 실패: %s", e)
        fields = {}

    text_chunks = split_markdown_into_chunks(
        markdown_text=markdown_text,
        document_id=source_pdf.stem,
        source_pdf=source_pdf,
        model_name=GEMINI_PDF_MODEL,
    )

    image_chunks = await build_image_caption_chunks(
        source_pdf=source_pdf,
        output_dir=output_dir,
    )

    chunks = text_chunks + image_chunks
    elapsed_total = time.perf_counter() - t0

    document_json = build_document_json(
        source_pdf=source_pdf,
        markdown_text=markdown_text,
        chunks=chunks,
        elapsed_sec=elapsed_total,
        fields=fields,
        image_count=len(image_chunks),
        model_name=GEMINI_PDF_MODEL,
    )

    parse_report = {
        "source_file": str(source_pdf),
        "document_id": source_pdf.stem,
        "mode": "fast_gemini_file_api",
        "model": GEMINI_PDF_MODEL,
        "elapsed_parse_sec": round(elapsed_parse, 3),
        "elapsed_total_sec": round(elapsed_total, 3),
        "text_len": len(markdown_text or ""),
        "text_chunk_count": len(text_chunks),
        "image_chunk_count": len(image_chunks),
        "chunk_count": len(chunks),
        "field_keys": list(fields.keys()),
        "warnings": [],
        "errors": [],
    }

    save_outputs(
        output_dir=output_dir,
        source_pdf=source_pdf,
        markdown_text=markdown_text,
        document_json=document_json,
        parse_report=parse_report,
        fields=fields,
        chunks=chunks,
        save_final_md=SAVE_FINAL_MD,
        save_final_json=SAVE_FINAL_JSON,
        save_parse_report=SAVE_PARSE_REPORT,
        save_fields_json=SAVE_FIELDS_JSON,
        save_vector_chunks=SAVE_VECTOR_CHUNKS,
    )

    logger.info("🎉 완료! (Gemini File API + Markdown + JSON + Fields + Vector Chunks)")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()