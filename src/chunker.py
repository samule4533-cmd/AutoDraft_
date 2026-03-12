import re
from pathlib import Path
from typing import Any, Dict, List


def split_markdown_into_chunks(
    markdown_text: str,
    document_id: str,
    source_pdf: Path,
    model_name: str,
    section_max_len: int = 1500,
    group_max_len: int = 1300,
) -> List[Dict[str, Any]]:
    """
    RAG용 최소 chunking:
    - Markdown 헤더(#, ##, ###) 기준 우선 분리
    - 너무 긴 chunk는 문단 단위로 추가 분리
    """
    text = (markdown_text or "").strip()
    if not text:
        return []

    lines = text.splitlines()
    sections: List[Dict[str, Any]] = []

    current_header = "ROOT"
    current_lines: List[str] = []

    def flush_section() -> None:
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
            "model": model_name,
            "mode": "fast_gemini_file_api",
        }

        # section 길이가 적당하면 그대로 1개 chunk
        if len(sec_text) <= section_max_len:
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

        # 너무 길면 빈 줄 기준으로 문단 그룹 분리
        paras = re.split(r"\n\s*\n", sec_text)
        buf: List[str] = []
        buf_len = 0

        for para in paras:
            p = para.strip()
            if not p:
                continue

            if buf_len + len(p) > group_max_len and buf:
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