import asyncio
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-flash-preview")

ENABLE_IMAGE_CAPTIONS = False
SAVE_EXTRACTED_IMAGES = True
IMAGE_MIN_WIDTH = 80
IMAGE_MIN_HEIGHT = 80
MAX_IMAGES_PER_DOC = 20

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


def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다.")
    return genai.Client(api_key=api_key)


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