# 저장 포맷 책임 분리, 파일 저장 로직 재사용, DB저장 변환 용이
# 현재 코드는 로컬임베딩 추후에 openai api 가지고와서 임베딩 해야함!

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def korean_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(re.findall(r"[가-힣]", s)) / max(len(s), 1)


def build_document_json(
    source_pdf: Path,
    markdown_text: str,
    chunks: List[Dict[str, Any]],
    elapsed_sec: float,
    fields: Dict[str, Any],
    image_count: int,
    model_name: str,
) -> Dict[str, Any]:
    return {
        "source_file": str(source_pdf),
        "document_id": source_pdf.stem,
        "format": "pdf",
        "mode": "fast_gemini_file_api",
        "model": model_name,
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
    save_final_md: bool = True,
    save_final_json: bool = True,
    save_parse_report: bool = True,
    save_fields_json: bool = True,
    save_vector_chunks: bool = True,
):
    if save_final_md:
        final_md_path = output_dir / f"{source_pdf.stem}.md"
        final_md_path.write_text(markdown_text, encoding="utf-8")

    if save_final_json:
        final_json_path = output_dir / f"{source_pdf.stem}.json"
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(document_json, f, ensure_ascii=False, indent=2)

    if save_parse_report:
        parse_report_path = output_dir / "parse_report.json"
        with open(parse_report_path, "w", encoding="utf-8") as f:
            json.dump(parse_report, f, ensure_ascii=False, indent=2)

    if save_fields_json:
        fields_path = output_dir / "fields.json"
        with open(fields_path, "w", encoding="utf-8") as f:
            json.dump(fields, f, ensure_ascii=False, indent=2)

    if save_vector_chunks:
        chunks_path = output_dir / "vector_chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)