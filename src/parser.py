import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

import cleaning
import field_extract

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

DEFAULT_PDF_NAME = os.getenv("DEFAULT_PDF_NAME", "sample2.pdf")
SOURCE_PDF = INPUT_DIR / DEFAULT_PDF_NAME

# =============================================================================
# Config
# =============================================================================
GEMINI_PDF_MODEL = os.getenv("GEMINI_PDF_MODEL", "gemini-3-flash-preview")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-flash-preview")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))

SAVE_FINAL_MD = True
SAVE_FINAL_JSON = True
SAVE_PARSE_REPORT = True
SAVE_FIELDS_JSON = True
SAVE_VECTOR_CHUNKS = True

ENABLE_IMAGE_CAPTIONS = True
SAVE_EXTRACTED_IMAGES = True

# 테스트 단계에서는 너무 크지 않게 잡는 게 좋음
IMAGE_MIN_WIDTH = 80
IMAGE_MIN_HEIGHT = 80
MAX_IMAGES_PER_DOC = 20

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

IMAGE_CAPTION_PROMPT = """
이 이미지를 공고문/RAG 검색용으로 설명하라.

규칙:
1. 반드시 1~2문장으로만 작성할 것.
2. 먼저 이미지 유형을 가능한 범위에서 밝힐 것: 도면, 위치도, 구조도, 현장사진, 서식, 표 이미지, 기타.
3. 문서에서 무엇을 설명하는 이미지인지 핵심 주제만 짧게 적을 것.
4. 문서 검색에 도움이 되도록 공사명, 사업명, 대상, 구조, 위치, 양식 목적이 보이면 포함할 것.
5. 추정은 최소화하고, 확실하지 않으면 과장하지 말 것.
6. 출력은 불필요한 수식 없이 평문만 반환할 것.
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


def split_markdown_into_chunks(markdown_text: str, document_id: str, source_pdf: Path) -> List[Dict[str, Any]]:
    """
    RAG용 최소 chunking:
    - 헤더(#, ##, ###) 기준 우선 분리
    - 너무 긴 chunk는 문단 단위로 추가 분리
    """
    text = (markdown_text or "").strip()
    if not text:
        return []

    lines = text.splitlines()
    sections: List[Dict[str, Any]] = []

    current_header = "ROOT"
    current_lines: List[str] = []

    def flush_section():
        nonlocal current_lines, current_header
        body = "\n".join(current_lines).strip()
        if body:
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

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 1

    for sec_idx, sec in enumerate(sections, start=1):
        sec_text = sec["text"].strip()

        base_metadata = {
            "document_id": document_id,
            "source_file": source_pdf.name,
            "section_order": sec_idx,
            "header": sec["header"],
            "source": "gemini_file_api_markdown",
            "model": GEMINI_PDF_MODEL,
            "mode": "fast_gemini_file_api",
        }

        if len(sec_text) <= 1500: # section이 짧으면 그대로 1개 chunk로 봄
            chunks.append(
                {
                    "chunk_id": f"{document_id}_c{chunk_idx}",
                    "header": sec["header"],
                    "text": sec_text,
                    "metadata": {
                        **base_metadata,
                        "chunk_type": "section",
                    },
                }
            )
            chunk_idx += 1
            continue

        paras = re.split(r"\n\s*\n", sec_text) # 길면 쪼개서 buf에 쌓아버림
        buf: List[str] = []
        buf_len = 0

        for para in paras:
            p = para.strip()
            if not p:
                continue

            if buf_len + len(p) > 1300 and buf: # buf가 길어지면 chunck로 저장함
                joined = "\n\n".join(buf).strip()
                chunks.append(
                    {
                        "chunk_id": f"{document_id}_c{chunk_idx}",
                        "header": sec["header"],
                        "text": joined,
                        "metadata": {
                            **base_metadata,
                            "chunk_type": "paragraph_group",
                        },
                    }
                )
                chunk_idx += 1
                buf = [p]
                buf_len = len(p)
            else:
                buf.append(p)
                buf_len += len(p)

        if buf:
            joined = "\n\n".join(buf).strip()
            chunks.append(
                {
                    "chunk_id": f"{document_id}_c{chunk_idx}",
                    "header": sec["header"],
                    "text": joined,
                    "metadata": {
                        **base_metadata,
                        "chunk_type": "paragraph_group",
                    },
                }
            )
            chunk_idx += 1

    return chunks


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


def build_document_json(
    source_pdf: Path,
    markdown_text: str,
    chunks: List[Dict[str, Any]],
    elapsed_sec: float,
    fields: Dict[str, Any],
    image_count: int,
) -> Dict[str, Any]:
    return {
        "source_file": str(source_pdf),
        "document_id": source_pdf.stem,
        "format": "pdf",
        "mode": "fast_gemini_file_api",
        "model": GEMINI_PDF_MODEL,
        "elapsed_sec": round(elapsed_sec, 3),
        "markdown_text": markdown_text,
        "fields": fields,
        "chunks": chunks,
        "stats": {
            "text_len": len(markdown_text or ""),
            "chunk_count": len(chunks),
            "korean_ratio": round(korean_ratio(markdown_text or ""), 4),
            "image_count": image_count,
        },
    }


def save_outputs(
    output_dir: Path,
    source_pdf: Path,
    markdown_text: str,
    document_json: Dict[str, Any],
    parse_report: Dict[str, Any],
    fields: Dict[str, Any],
    chunks: List[Dict[str, Any]],
):
    if SAVE_FINAL_MD:
        final_md_path = output_dir / f"{source_pdf.stem}.md"
        final_md_path.write_text(markdown_text, encoding="utf-8")

    if SAVE_FINAL_JSON:
        final_json_path = output_dir / f"{source_pdf.stem}.json"
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(document_json, f, ensure_ascii=False, indent=2)

    if SAVE_PARSE_REPORT:
        parse_report_path = output_dir / "parse_report.json"
        with open(parse_report_path, "w", encoding="utf-8") as f:
            json.dump(parse_report, f, ensure_ascii=False, indent=2)

    if SAVE_FIELDS_JSON:
        fields_path = output_dir / "fields.json"
        with open(fields_path, "w", encoding="utf-8") as f:
            json.dump(fields, f, ensure_ascii=False, indent=2)

    if SAVE_VECTOR_CHUNKS:
        chunks_path = output_dir / "vector_chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)


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
# Image Captioning
# =============================================================================
def extract_images_from_pdf(source_pdf: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """
    PDF에서 의미 있을 가능성이 있는 이미지를 추출한다.
    너무 작은 로고/아이콘은 제외한다.
    저장은 모두 PNG로 통일한다.
    """
    if not ENABLE_IMAGE_CAPTIONS:
        return []

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(source_pdf))
    extracted: List[Dict[str, Any]] = []
    image_counter = 1

    try:
        for page_index in range(doc.page_count):
            if len(extracted) >= MAX_IMAGES_PER_DOC:
                break

            page = doc.load_page(page_index)
            page_images = page.get_images(full=True)

            logger.info("page %d: raw images=%d", page_index + 1, len(page_images))

            seen_xrefs = set()

            for img in page_images:
                if len(extracted) >= MAX_IMAGES_PER_DOC:
                    break

                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    width, height = image.size

                    logger.info(
                        "candidate image(page=%d, xref=%s, size=%dx%d)",
                        page_index + 1,
                        xref,
                        width,
                        height,
                    )

                    if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
                        logger.info(
                            "skip small image(page=%d, xref=%s, size=%dx%d)",
                            page_index + 1,
                            xref,
                            width,
                            height,
                        )
                        continue

                    image_name = f"page_{page_index + 1}_img_{image_counter}.png"
                    image_path = images_dir / image_name

                    image.save(image_path, format="PNG")

                    extracted.append(
                        {
                            "page_number": page_index + 1,
                            "image_index": image_counter,
                            "image_path": str(image_path),
                            "width": width,
                            "height": height,
                        }
                    )
                    image_counter += 1

                except Exception:
                    logger.exception("이미지 추출 실패(page=%s, xref=%s)", page_index + 1, xref)

    finally:
        doc.close()

    logger.info("이미지 추출 완료: %d개", len(extracted))
    return extracted


async def generate_image_caption(image_path: str) -> str:
    client = get_genai_client()
    image_path_obj = Path(image_path)

    if not image_path_obj.exists():
        logger.warning("이미지 파일을 확인할 수 없습니다: %s", image_path)
        return ""

    def _load_image_part(path: Path):
        with open(path, "rb") as f:
            data = f.read()
        return types.Part.from_bytes(data=data, mime_type="image/png")

    image_part = await asyncio.to_thread(_load_image_part, image_path_obj)

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[IMAGE_CAPTION_PROMPT, image_part],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
        return (response.text or "").strip()
    except Exception as e:
        logger.warning("이미지 캡션 생성 실패(%s): %s", image_path, e)
        return ""


async def build_image_caption_chunks(source_pdf: Path, output_dir: Path) -> List[Dict[str, Any]]:
    if not ENABLE_IMAGE_CAPTIONS:
        return []

    image_infos = extract_images_from_pdf(source_pdf, output_dir)
    if not image_infos:
        return []

    chunks: List[Dict[str, Any]] = []

    for idx, info in enumerate(image_infos, start=1):
        caption = await generate_image_caption(info["image_path"])
        if not caption:
            continue

        chunks.append(
            {
                "chunk_id": f"{source_pdf.stem}_img_{idx}",
                "header": "## 이미지 설명",
                "text": caption,
                "metadata": {
                    "document_id": source_pdf.stem,
                    "source_file": source_pdf.name,
                    "chunk_type": "image_caption",
                    "source": "gemini_image_caption",
                    "model": GEMINI_IMAGE_MODEL,
                    "mode": "fast_gemini_file_api",
                    "page_number": info["page_number"],
                    "image_index": info["image_index"],
                    "image_path": info["image_path"],
                    "width": info["width"],
                    "height": info["height"],
                },
            }
        )

    logger.info("이미지 캡션 chunk 생성 완료: %d개", len(chunks))
    return chunks


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

    elapsed_parse = time.perf_counter() - t0

    final_pages = build_final_pages_like(markdown_text)

    try:
        fields = {
            "bid_amount": field_extract.extract_bid_amount_from_final_pages(final_pages),
        }
    except Exception as e:
        logger.warning("fields 추출 실패: %s", e)
        fields = {}

    text_chunks = split_markdown_into_chunks(markdown_text, source_pdf.stem, source_pdf)
    image_chunks = await build_image_caption_chunks(source_pdf, output_dir)

    chunks = text_chunks + image_chunks

    elapsed_total = time.perf_counter() - t0

    document_json = build_document_json(
        source_pdf=source_pdf,
        markdown_text=markdown_text,
        chunks=chunks,
        elapsed_sec=elapsed_total,
        fields=fields,
        image_count=len(image_chunks),
    )

    parse_report = {
        "source_file": str(source_pdf),
        "document_id": source_pdf.stem,
        "mode": "fast_gemini_file_api",
        "model": GEMINI_PDF_MODEL,
        "image_model": GEMINI_IMAGE_MODEL,
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
    )

    logger.info("🎉 완료! (Gemini File API + Markdown + JSON + Fields + Text/Image Vector Chunks)")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()