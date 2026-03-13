import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Core Helpers
# =============================================================================
def load_chunks_from_json(chunks_path: Path) -> List[Dict[str, Any]]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChromaDB에 안전하게 들어가도록 metadata 정리.
    - None 제거
    - str/int/float/bool 유지
    - list는 내부 타입이 단순하면 유지
    - dict/복합 객체는 문자열로 변환
    """
    clean_meta: Dict[str, Any] = {}

    for k, v in metadata.items():
        if v is None:
            continue

        if isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v
        elif isinstance(v, list):
            if all(isinstance(x, (str, int, float, bool)) for x in v):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        else:
            clean_meta[k] = str(v)

    return clean_meta


def prepare_chroma_items(
    chunks: List[Dict[str, Any]],
    default_doc_type: Optional[str] = None,
    exclude_image_chunks: bool = True,
) -> Dict[str, List[Any]]:
    """
    base rules 반영:
    - 빈 텍스트 제외
    - image_caption 제외 옵션
    - metadata 정리
    - doc_type 기본값 보정
    """
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        metadata = dict(chunk.get("metadata", {}))

        # 현재 image_caption은 노이즈 가능성이 있어 기본 제외
        if exclude_image_chunks and metadata.get("chunk_type") == "image_caption":
            continue

        if default_doc_type and not metadata.get("doc_type"):
            metadata["doc_type"] = default_doc_type

        clean_meta = clean_metadata_for_chroma(metadata)

        ids.append(chunk["chunk_id"])
        documents.append(text)
        metadatas.append(clean_meta)

    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }


def get_notice_chunks_path_from_env() -> Path:
    default_pdf_name = os.getenv("DEFAULT_PDF_NAME", "sample1.pdf")
    document_id = Path(default_pdf_name).stem
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "parsing_result_notices"
        / document_id
        / "vector_chunks.json"
    )


def get_chroma_dir() -> Path:
    return PROJECT_ROOT / "data" / "vector_store" / "chroma"


# =============================================================================
# Local Embedding Function
# =============================================================================
def get_local_embedding_function(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
):
    """
    한국어 포함 다국어 대응 가능한 로컬 임베딩 모델
    """
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_or_create_collection(
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
):
    client = chromadb.PersistentClient(path=persist_dir)

    local_ef = get_local_embedding_function(model_name=embedding_model_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=local_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# =============================================================================
# Chroma Upsert
# =============================================================================
def upsert_chunks_to_chroma(
    chunks: List[Dict[str, Any]],
    collection_name: str = "ninewatt_bids_local",
    persist_dir: str = "data/vector_store/chroma",
    batch_size: int = 50,
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    default_doc_type: Optional[str] = None,
    exclude_image_chunks: bool = True,
):
    collection = get_or_create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
    )

    items = prepare_chroma_items(
        chunks=chunks,
        default_doc_type=default_doc_type,
        exclude_image_chunks=exclude_image_chunks,
    )

    total_chunks = len(items["ids"])
    logger.info("총 %d개의 청크를 DB에 적재 시작...", total_chunks)

    if total_chunks == 0:
        logger.warning("적재할 청크가 없습니다.")
        return collection

    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)

        batch_ids = items["ids"][i:end_idx]
        batch_docs = items["documents"][i:end_idx]
        batch_metas = items["metadatas"][i:end_idx]

        logger.info("Upserting batch %d ~ %d", i, end_idx)

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
        )

    logger.info("✅ ChromaDB 적재 완료!")
    logger.info("현재 collection count: %d", collection.count())
    return collection


def load_and_upsert_chunks(
    chunks_path: Path,
    collection_name: str = "ninewatt_bids_local",
    persist_dir: str = "data/vector_store/chroma",
    batch_size: int = 50,
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    default_doc_type: Optional[str] = None,
    exclude_image_chunks: bool = True,
):
    chunks = load_chunks_from_json(chunks_path)

    return upsert_chunks_to_chroma(
        chunks=chunks,
        collection_name=collection_name,
        persist_dir=persist_dir,
        batch_size=batch_size,
        embedding_model_name=embedding_model_name,
        default_doc_type=default_doc_type,
        exclude_image_chunks=exclude_image_chunks,
    )


# =============================================================================
# Query Helpers
# =============================================================================
def query_collection(
    query_text: str,
    collection_name: str = "ninewatt_bids_local",
    persist_dir: str = "data/vector_store/chroma",
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
):
    collection = get_or_create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
    )

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
    )
    return results


def print_query_summary(query_text: str, results: Dict[str, Any]) -> None:
    print(f"\n[질문] {query_text}")

    ids = results.get("ids", [[]])
    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])

    if not ids or not ids[0]:
        print("검색 결과 없음")
        return

    top_ids = ids[0]
    top_docs = docs[0] if docs else []
    top_metas = metas[0] if metas else []
    top_distances = distances[0] if distances else []

    for i, chunk_id in enumerate(top_ids):
        header = ""
        if i < len(top_metas) and top_metas[i]:
            header = top_metas[i].get("header", "")

        dist = top_distances[i] if i < len(top_distances) else None
        preview = top_docs[i][:180].replace("\n", " ") if i < len(top_docs) else ""

        print(f"\n  - rank {i+1}")
        print(f"    id       : {chunk_id}")
        print(f"    header   : {header}")
        print(f"    distance : {dist}")
        print(f"    preview  : {preview}...")


# =============================================================================
# Example CLI Entry
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    notice_chunks_path = get_notice_chunks_path_from_env()
    chroma_dir = get_chroma_dir()

    logger.info("현재 vector_chunks 경로: %s", notice_chunks_path)
    logger.info("현재 Chroma 저장 경로: %s", chroma_dir)

    if not notice_chunks_path.exists():
        logger.warning("vector_chunks.json 파일이 없습니다: %s", notice_chunks_path)
        return

    collection = load_and_upsert_chunks(
        chunks_path=notice_chunks_path,
        collection_name="ninewatt_bids_local",
        persist_dir=str(chroma_dir),
        batch_size=50,
        embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        default_doc_type="notice",
        exclude_image_chunks=True,
    )

    print("\n[컬렉션 적재 확인]")
    print("collection count:", collection.count())

    sample = collection.get(limit=3)
    print("sample ids:", sample.get("ids"))
    print("sample metadatas:", sample.get("metadatas"))

    test_queries = [
        "입찰시 어떤 서류를 제출해야 하나?",
        "입찰보증금은 어떻게 되나?",
        "입찰참가자격이 무엇인가?",
        "문의처 연락처는?",
        "기초금액은 얼마인가?",
    ]

    print("\n[질문 테스트 시작]")
    for q in test_queries:
        results = query_collection(
            query_text=q,
            collection_name="ninewatt_bids_local",
            persist_dir=str(chroma_dir),
            embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            n_results=5,
            where={"doc_type": "notice"},
        )
        print_query_summary(q, results)


if __name__ == "__main__":
    main()