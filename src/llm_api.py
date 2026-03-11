# src/llm_api.py
import io
import json
import os
import re
from typing import Any, Dict, Optional

from PIL import Image
from google import genai
from google.genai import types


class GeminiAPIError(Exception):
    pass


class GeminiAuthError(GeminiAPIError):
    pass


class GeminiResponseParseError(GeminiAPIError):
    pass


_gemini_client: Optional[genai.Client] = None


def is_gemini_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiAuthError("GEMINI_API_KEY 환경변수가 없습니다.")
    return api_key


def get_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=get_api_key())
    return _gemini_client


def pil_to_png_part(pil_img: Image.Image) -> types.Part:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def safe_json_load(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()

    if not text:
        return {"key_values": {}, "ocr_text": "", "summary": "빈 응답"}

    # 1) 그대로 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) fenced json block
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # 3) 첫 { ~ 마지막 } 범위 복구
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {"key_values": {}, "ocr_text": text, "summary": "JSON 파싱 실패"}


def generate_image_json(
    pil_img: Image.Image,
    *,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    response_mime_type: str = "application/json",
) -> Dict[str, Any]:
    client = get_client()

    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=[prompt, pil_to_png_part(pil_img)],
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type=response_mime_type,
            ),
        )
    except Exception as e:
        msg = str(e)
        lowered = msg.lower()
        if "api key" in lowered or "api_key" in lowered or "401" in lowered or "403" in lowered:
            raise GeminiAuthError(f"Gemini 인증 실패: {msg}") from e
        raise GeminiAPIError(f"Gemini 호출 실패: {msg}") from e

    raw_text = getattr(resp, "text", "") or ""
    return safe_json_load(raw_text)