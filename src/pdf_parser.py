# pdf 파일 1개만 가능 (단독 테스트용)
# 배치 처리는 company_ingest.py 사용

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
# Paths (단독 실행용 기본값 — company_ingest에서는 경로를 직접 전달)
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
GEMINI_PDF_MODEL = os.getenv("GEMINI_PDF_MODEL", "gemini-2.0-flash")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))

SAVE_FINAL_MD = True
SAVE_FINAL_JSON = True
SAVE_PARSE_REPORT = True
SAVE_FIELDS_JSON = True
SAVE_VECTOR_CHUNKS = False  # vector_chunks.json 미저장 — *.md가 재청킹 소스

# =============================================================================
# 파싱 프롬프트 (모듈형)
# - _BASE_PARSE_PROMPT  : 문서 유형에 무관한 공통 지침
# - _PATENT_RULES       : 특허 문서 전용 구조화 지침
# - _CERT_RULES         : 인증서 문서 전용 구조화 지침 (향후 확장용)
# - _COMPANY_RULES      : 회사소개/실적자료 전용 구조화 지침 (향후 확장용)
# - build_parse_prompt() : doc_type에 따라 프롬프트를 조합하여 반환
# =============================================================================
_BASE_PARSE_PROMPT = """
너는 RAG 파이프라인을 위한 최고 수준의 문서 파서야.
첨부된 PDF 문서를 읽고 레이아웃이 최대한 보존된 Markdown 형식으로 변환해줘.

[절대 준수 사항]
1. 문서 내 텍스트, 표, 제목, 소제목, 항목, 번호, 캡션은 임의로 누락하거나 요약하지 말 것.
2. 문서의 구조를 가능한 한 유지하면서 Markdown 헤더(#, ##, ###)로 계층화할 것.
3. 표는 가능한 경우 Markdown 표 문법으로 보존하고, 표 구조가 복잡하면 행/열 의미가 유지되도록 텍스트로라도 충실히 옮길 것.
4. 숫자, 날짜, 등록번호, 특허번호, 인증번호, 기관명, 회사명, 제품명, 기술명, 문의처 등은 원문에 가깝게 보존할 것.
5. 본문 내 강조(**bold**), 목록, 표제어, 도면명, 도표 설명, 붙임 제목 등은 가능한 한 유지할 것.
6. 읽기 애매한 부분은 임의로 자연스럽게 바꾸지 말고 원문에 가깝게 남길 것.
7. 페이지 구분선(---, ===, ─── 등 구분 역할의 선)은 완전히 제거할 것. 헤더 바로 아래에도 구분선을 넣지 말 것.
   페이지 번호는 `[p.N]` 형식으로 단독 줄에 통일하여 보존할 것 (예: `[p.3]`).
   단, 특허 번호·등록번호·인증번호처럼 본문 내용에 해당하는 번호는 절대 건드리지 말 것.

[공통 구조화 지침]
8. 헤더(#, ##, ###)는 실질적인 내용(본문 2줄 이상)이 뒤따르는 섹션에만 붙일 것.
   단일 줄 값, 짧은 레이블, 번호 하나만 있는 항목에는 헤더를 붙이지 말 것.
9. "중략", "생략", "요약", "...", "등" 등의 축약 표현을 임의로 삽입하지 말 것.
10. 응답은 문서에서 확인되는 내용을 처음부터 끝까지 순서대로 충실히 옮길 것.
""".strip()

_PATENT_RULES = """
[특허 문서 구조화 지침]
11. 특허 서지정보 항목 [(19) 국가, (21) 출원번호, (22) 출원일, (24) 등록일, (45) 공고일,
    (51) IPC, (54) 발명의 명칭, (57) 요약, (71) 출원인, (72) 발명자, (73) 특허권자,
    (74) 대리인 등 번호로 시작하는 메타데이터 필드]는 각각을 별도 헤더(##)로 쪼개지 말 것.
    → "## 특허 서지정보" 하나의 섹션 안에 "- 항목명: 값" 목록으로 묶을 것.
    예시:
    ## 특허 서지정보
    - 등록번호: 10-2708831
    - 공고일자: 2024년 09월 24일
    - 등록일자: 2024년 09월 19일
    - 출원번호: 10-2023-0170792
    - 출원인: 주식회사 나인와트
    - 발명자: 홍길동
    - 대리인: 이미래특허법률사무소

12. 단독 등록번호·특허번호·출원번호처럼 값만 있는 한 줄짜리는 반드시 서지정보 섹션이나
    선행기술 섹션 등 가장 관련 있는 섹션의 내용으로 포함시킬 것.
    독립 헤더나 독립 단락으로 분리하지 말 것.

13. 발명의 명칭, 기술분야, 배경기술, 해결과제, 해결수단, 효과, 청구항, 도면 설명은 각각
    별도 섹션(##)으로 분리하여 정확히 보존할 것.
""".strip()

_CERT_RULES = """
[인증서 문서 구조화 지침]
11. 인증명, 인증기관, 인증번호, 등록일, 유효기간, 인증 범위 등 메타 항목은
    "## 인증 서지정보" 하나의 섹션 안에 "- 항목명: 값" 목록으로 묶을 것.
12. 인증 범위·조건·세부 내역은 별도 섹션으로 분리하여 보존할 것.
""".strip()

_COMPANY_RULES = """
[회사소개/실적 문서 구조화 지침]
11. 사업 개요, 주요 기술, 제품/서비스 설명, 프로젝트명, 발주처, 기간, 성과, 수치 정보,
    표와 목록을 정확히 보존할 것.
12. 목차가 있으면 목차 구조를 유지하고, 본문 섹션 제목이 분명하면 Markdown 헤더로 승격할 것.
""".strip()

_OUTPUT_INSTRUCTION = "\n\n출력은 설명 없이 Markdown 본문만 반환할 것."

_DOC_TYPE_RULES = {
    "patent":  _PATENT_RULES,
    "cert":    _CERT_RULES,
    "company": _COMPANY_RULES,
}


def build_parse_prompt(doc_type: str = "company") -> str:
    """
    doc_type에 맞는 파싱 프롬프트를 조합하여 반환.
    지원 doc_type: "patent" | "cert" | "company"
    알 수 없는 타입은 company 규칙으로 fallback.
    """
    type_rules = _DOC_TYPE_RULES.get(doc_type, _COMPANY_RULES)
    return _BASE_PARSE_PROMPT + "\n\n" + type_rules + _OUTPUT_INSTRUCTION


# 단독 실행 / 기본값용 — doc_type은 .env의 DOC_SOURCE_TYPE에서 결정
# 특허(patent) 문서가 대부분이면 "patent"로 지정
PDF_PARSE_PROMPT = build_parse_prompt(doc_type="patent")


# =============================================================================
# Helpers
# =============================================================================
def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다.")
    return genai.Client(api_key=api_key)


def build_output_dir(
    source_pdf: Path,
    input_dir: Path = None,
    output_root: Path = None,
) -> Path:
    """
    입력 루트 아래 상대경로를 유지하면서 output dir 생성.
    company_ingest에서 호출 시 input_dir, output_root를 명시적으로 전달.
    단독 실행 시에는 모듈 전역 INPUT_DIR, OUTPUT_ROOT 사용.
    """
    _input_dir = input_dir or INPUT_DIR
    _output_root = output_root or OUTPUT_ROOT
    relative_parent = source_pdf.relative_to(_input_dir).parent
    out_dir = _output_root / relative_parent / source_pdf.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def preflight_check(source_pdf: Path, output_dir: Path) -> Dict[str, Any]:
    """실행 전 필수 조건 검사 (파일 존재 / API 키 / 쓰기 권한)"""
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
    Gemini가 줄 전체를 **제목** 형태로 출력할 경우 ## 헤더로 보정.
    본문 내 인라인 **강조**는 그대로 유지.
    """
    lines = (markdown_text or "").splitlines()
    normalized: List[str] = []

    for line in lines:
        s = line.strip()

        # 이미 Markdown 헤더면 그대로
        if re.match(r"^\s{0,3}#{1,6}\s+", s):
            normalized.append(line)
            continue

        # 줄 전체가 **텍스트** 패턴이면 ## 헤더로 변환
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
    한글 파일명은 Gemini File API 업로드 시 오류 발생 가능.
    임시 디렉토리에 영문명(upload_input.pdf)으로 복사 후 업로드하여 우회.
    """
    logger.info("PDF 업로드 시작: %s", file_path.name)

    suffix = file_path.suffix or ".pdf"

    with tempfile.TemporaryDirectory() as tmpdir:
        safe_path = Path(tmpdir) / f"upload_input{suffix}"
        shutil.copy2(file_path, safe_path)  # 한글명 → 영문명 복사

        uploaded = client.files.upload(file=str(safe_path))

        # 파일 처리 완료될 때까지 폴링
        while uploaded.state.name == "PROCESSING":
            logger.info("파일 처리 중... 2초 후 재확인")
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        if uploaded.state.name == "FAILED":
            raise RuntimeError(f"Gemini File API 처리 실패: {file_path.name}")

        logger.info("PDF 업로드 완료: %s", file_path.name)
        return uploaded


async def parse_pdf_to_markdown(source_pdf: Path, parse_prompt: str = PDF_PARSE_PROMPT) -> str:
    """PDF를 Gemini File API에 업로드하고 Markdown 텍스트로 변환하여 반환"""
    client = get_genai_client()

    uploaded_file = await asyncio.to_thread(_upload_and_wait, client, source_pdf)

    logger.info("Gemini 호출 시작: model=%s, file=%s", GEMINI_PDF_MODEL, source_pdf.name)
    response = await client.aio.models.generate_content(
        model=GEMINI_PDF_MODEL,
        contents=[uploaded_file, parse_prompt],
        config=types.GenerateContentConfig(
            temperature=0.0,                         # 재현성을 위해 temperature 0
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        ),
    )

    markdown_text = response.text or ""
    logger.info("Gemini 파싱 완료: %s (출력 %d자)", source_pdf.name, len(markdown_text))
    return markdown_text


# =============================================================================
# 핵심 공개 함수 — company_ingest.py에서 import하여 사용
# =============================================================================
async def parse_single_pdf(
    source_pdf: Path,
    input_dir: Path,
    output_root: Path,
    doc_type: str = "patent",
) -> List[Dict[str, Any]]:
    """
    PDF 1개를 파싱하고 디스크에 저장한 뒤 청크 목록을 반환.

    Args:
        source_pdf:  처리할 PDF 경로
        input_dir:   raw 데이터 루트 (상대경로 계산 기준)
        output_root: 처리 결과를 저장할 루트

    Returns:
        vector_chunks 리스트 (company_ingest가 ChromaDB에 적재)
    """
    output_dir = build_output_dir(source_pdf, input_dir, output_root)

    # 실행 전 필수 조건 검사
    pf = preflight_check(source_pdf, output_dir)
    with open(output_dir / "preflight.json", "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)

    if not pf["ok"]:
        failed = [c for c in pf["checks"] if not c["ok"]]
        raise RuntimeError(f"Pre-flight check 실패 ({source_pdf.name}): {failed}")

    t0 = time.perf_counter()

    # Gemini File API로 PDF → Markdown 변환
    markdown_text = await parse_pdf_to_markdown(source_pdf, parse_prompt=build_parse_prompt(doc_type))

    # 볼드 전체줄을 헤더로 보정 (Gemini 출력 정규화)
    markdown_text = normalize_markdown_headings(markdown_text)

    elapsed_parse = time.perf_counter() - t0

    fields: Dict[str, Any] = {}  # 회사 자료는 field 추출 비활성화

    # 헤더 기반 청크 분리
    text_chunks = split_markdown_into_chunks(
        markdown_text=markdown_text,
        document_id=source_pdf.stem,
        source_pdf=source_pdf,
        model_name=GEMINI_PDF_MODEL,
    )

    elapsed_total = time.perf_counter() - t0

    document_json = build_document_json(
        source_pdf=source_pdf,
        markdown_text=markdown_text,
        chunks=text_chunks,
        elapsed_sec=elapsed_total,
        fields=fields,
        image_count=0,
        model_name=GEMINI_PDF_MODEL,
    )

    parse_report = {
        "source_file": str(source_pdf),
        "document_id": source_pdf.stem,
        "doc_source_type": "company",
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "mode": "fast_gemini_file_api",
        "model": GEMINI_PDF_MODEL,
        "elapsed_parse_sec": round(elapsed_parse, 3),
        "elapsed_total_sec": round(elapsed_total, 3),
        "text_len": len(markdown_text or ""),
        "text_chunk_count": len(text_chunks),
        "image_chunk_count": 0,
        "chunk_count": len(text_chunks),
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
        chunks=text_chunks,
        save_final_md=SAVE_FINAL_MD,
        save_final_json=SAVE_FINAL_JSON,
        save_parse_report=SAVE_PARSE_REPORT,
        save_fields_json=SAVE_FIELDS_JSON,
        save_vector_chunks=SAVE_VECTOR_CHUNKS,
    )

    logger.info("파싱 저장 완료: %s → %d 청크 (%.1f초)", source_pdf.name, len(text_chunks), elapsed_total)
    return text_chunks


# =============================================================================
# 단독 실행용 진입점 (단일 파일 테스트)
# =============================================================================
async def async_main():
    chunks = await parse_single_pdf(
        source_pdf=SOURCE_PDF,
        input_dir=INPUT_DIR,
        output_root=OUTPUT_ROOT,
    )
    logger.info("🎉 완료! 총 %d개 청크 생성", len(chunks))


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
