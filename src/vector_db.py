# vector_chuncks.json파일을 읽어오고 그 안에 있는 text chunck, image chunk 목록 가져옴
#=

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import chromadb
except ImportError:
    chromadb = None


def load_chunks_from_json(chunks_path: Path) -> List[Dict[str, Any]]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_chroma_items(chunks: List[Dict[str, Any]]) -> Dict[str, List[Any]]: # 각 chunck에서 ids->chunck_id, documents->text, metadatas->metadata
    ids: List[str] = []                                                         # 즉, ChromaDB가 넣기 좋은 형태로 변환        
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        metadatas.append(chunk.get("metadata", {}))

    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }


def upsert_chunks_to_chroma(
    chunks: List[Dict[str, Any]],
    collection_name: str = "notice_chunks",
    persist_dir: str = "data/chroma",
):
    if chromadb is None:
        raise ImportError("chromadb가 설치되어 있지 않습니다. `uv add chromadb` 후 다시 실행하세요.")

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    items = prepare_chroma_items(chunks)

    collection.upsert(
        ids=items["ids"],
        documents=items["documents"],
        metadatas=items["metadatas"],
    )

    return collection


def load_and_upsert_chunks(
    chunks_path: Path,
    collection_name: str = "notice_chunks",
    persist_dir: str = "data/chroma",
):
    chunks = load_chunks_from_json(chunks_path)
    return upsert_chunks_to_chroma(
        chunks=chunks,
        collection_name=collection_name,
        persist_dir=persist_dir,
    )