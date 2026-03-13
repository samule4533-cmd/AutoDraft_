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

from chunker import split_markdown_into_chunks
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

DOC_SOURCE_TYPE = os.getenv("DOC_SOURCE_TYPE", "company").strip().lower()
DEFAULT_PDF_SUBDIR = os.getenv("DEFAULT_PDF_SUBDIR", "").strip()
DEFAULT_PDF_NAME = os.getenv("DEFAULT_PDF_NAME", "sample_company.pdf").strip()

if DOC_SOURCE_TYPE == "company":
    INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "company"
    OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "parsing_result_company"
elif DOC_SOURCE_TYPE == "notice":
    INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sample_notices"
    OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "parsing_result_notices"
else:
    raise ValueError(f"DOC_SOURCE_TYPE 값이 올바르지 않습니다: {DOC_SOURCE_TYPE}")

if DEFAULT_PDF_SUBDIR:
    SOURCE_PDF = INPUT_DIR / DEFAULT_PDF_SUBDIR / DEFAULT_PDF_NAME
else:
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
첨부된 PDF 문서를 읽고 레이아웃이 최대한 보존된 Markdown 형식으로 변환해줘.

[절대 준수 사항]
1. 문서 내 텍스트, 표, 제목, 소제목, 항목, 번호, 캡션은 임의로 누락하거나 요약하지 말 것.
2. 문서의 구조를 가능한 한 유지하면서 Markdown 헤더(#, ##, ###)로 계층화할 것.
3. 표는 가능한 경우 Markdown 표 문법으로 보존하고, 표 구조가 복잡하면 행/열 의미가 유지되도록 텍스트로라도 충실히 옮길 것.
4. 숫자, 날짜, 등록번호, 특허번호, 인증번호, 기관명, 회사명, 제품명, 기술명, 문의처 등은 원문에 가깝게 보존할 것.
5. 본문 내 강조(**bold**), 목록, 표제어, 도면명, 도표 설명, 붙임 제목 등은 가능한 한 유지할 것.
6. 읽기 애매한 부분은 임의로 자연스럽게 바꾸지 말고 원문에 가깝게 남길 것.

[회사 자료 특화 지침]
7. 문서가 특허, 인증서, 회사소개서, 실적자료, 제안서, 기술소개서, 브로슈어 중 무엇에 가까운지 구조를 유지하며 파싱할 것.
8. 특허 문서의 경우 발명의 명칭, 기술분야, 배경기술, 해결과제, 해결수단, 효과, 청구항, 도면 설명을 특히 정확히 보존할 것.
9. 인증 문서의 경우 인증명, 인증기관, 인증번호, 등록일, 유효기간, 대상, 범위를 정확히 보존할 것.
10. 회사소개/실적 자료의 경우 사업 개요, 주요 기술, 제품/서비스 설명, 프로젝트명, 발주처, 기간, 성과, 수치 정보, 표와 목록을 정확히 보존할 것.
11. 목차가 있으면 목차 구조를 유지하고, 본문 섹션 제목이 분명하면 Markdown 헤더로 승격할 것.
12. 문서 검색과 질의응답에 도움이 되도록, 의미 있는 제목/소제목/섹션 구분이 사라지지 않게 할 것.
13. 특허 문서의 경우 페이지 상단 또는 하단에 기재된 등록특허번호, 공개번호, 출원번호, 문서 식별번호와 같은 식별 정보를 누락하지 말 것.
14. 특히 첫 페이지 및 각 페이지 상단 우측/좌측의 작은 텍스트라도 특허번호, 등록번호, 문서번호에 해당하면 반드시 본문에 보존할 것.
15. 페이지 머리글처럼 보이더라도 특허 식별번호는 문서 검색과 식별에 중요하므로 생략하지 말 것.
16. "중략", "생략", "요약", "...", "등" 등의 축약 표현을 임의로 삽입하지 말 것.
17. 응답은 문서에서 확인되는 내용을 처음부터 끝까지 순서대로 충실히 옮길 것.

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
    """
    입력 루트 아래 상대경로를 유지하면서 output dir 생성
    예:
    data/raw/company/certification_list_1/sample_company.pdf
    ->
    data/processed/parsing_result_company/certification_list_1/sample_company/
    """
    relative_parent = source_pdf.relative_to(INPUT_DIR).parent
    out_dir = OUTPUT_ROOT / relative_parent / source_pdf.stem
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

        if re.match(r"^\s{0,3}#{1,6}\s+", s):
            normalized.append(line)
            continue

        m = re.match(r"^\*\*([^*]{2,120})\*\*$", s)
        if m:
            normalized.append(f"## {m.group(1).strip()}")
            continue

        normalized.append(line)

    return "\n".join(normalized)


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

    # 줄 전체 볼드 제목을 markdown heading으로 보정
    markdown_text = normalize_markdown_headings(markdown_text)

    elapsed_parse = time.perf_counter() - t0

    # 회사 자료 단계에서는 field 추출 비활성화
    fields: Dict[str, Any] = {}

    text_chunks = split_markdown_into_chunks(
        markdown_text=markdown_text,
        document_id=source_pdf.stem,
        source_pdf=source_pdf,
        model_name=GEMINI_PDF_MODEL,
    )

    chunks = text_chunks
    elapsed_total = time.perf_counter() - t0

    document_json = build_document_json(
        source_pdf=source_pdf,
        markdown_text=markdown_text,
        chunks=chunks,
        elapsed_sec=elapsed_total,
        fields=fields,
        image_count=0,
        model_name=GEMINI_PDF_MODEL,
    )

    parse_report = {
        "source_file": str(source_pdf),
        "document_id": source_pdf.stem,
        "doc_source_type": DOC_SOURCE_TYPE,
        "input_dir": str(INPUT_DIR),
        "output_root": str(OUTPUT_ROOT),
        "mode": "fast_gemini_file_api",
        "model": GEMINI_PDF_MODEL,
        "elapsed_parse_sec": round(elapsed_parse, 3),
        "elapsed_total_sec": round(elapsed_total, 3),
        "text_len": len(markdown_text or ""),
        "text_chunk_count": len(text_chunks),
        "image_chunk_count": 0,
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

    logger.info("🎉 완료! (Gemini File API + Markdown + JSON + Vector Chunks)")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()