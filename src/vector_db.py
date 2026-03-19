import json
import time
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
) -> Dict[str, List[Any]]:
    """
    - 빈 텍스트 제외
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


def get_chroma_dir() -> Path:
    return PROJECT_ROOT / "data" / "vector_store" / "chroma"


# =============================================================================
# Embedding Functions
# =============================================================================
def get_local_embedding_function(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
):
    """로컬 SentenceTransformer 임베딩 (API 비용 없음, 오프라인 가능)"""
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


class _OpenAIEmbeddingFunction:
    """
    OpenAI Embeddings API를 ChromaDB embedding_function 형식으로 감싼 클래스.
    text-embedding-3-small: 1536차원, 한국어 기술문서에서 로컬 모델보다 정밀도 높음.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def name(self) -> str:
        # ChromaDB가 컬렉션 생성/접근 시 임베딩 함수 충돌 검증에 사용
        return f"openai_{self._model}"

    def _embed(self, input: List[str]) -> List[List[float]]:
        # text-embedding-3-small 최대 8192 토큰 제한
        # 한국어 1글자 ≈ 2~3 토큰 기준으로 2500자를 안전 상한선으로 사용
        MAX_CHARS = 2500
        MAX_RETRIES = 3
        safe_input = []
        for text in input:
            if len(text) > MAX_CHARS:
                logger.warning("청크 길이 초과(%d자) → %d자로 truncate", len(text), MAX_CHARS)
                text = text[:MAX_CHARS]
            safe_input.append(text)

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.embeddings.create(input=safe_input, model=self._model)
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 10 * (attempt + 1)  # 10s, 20s
                    logger.warning("OpenAI 임베딩 실패, %d초 후 재시도 (%d/%d): %s", wait, attempt + 1, MAX_RETRIES, e)
                    time.sleep(wait)
                else:
                    logger.error("OpenAI 임베딩 최종 실패: %s", e)
                    raise

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)


def get_openai_embedding_function(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
) -> _OpenAIEmbeddingFunction:
    """OpenAI 임베딩 함수 반환. API 키는 인자 또는 OPENAI_API_KEY 환경변수에서 읽음."""
    key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
    return _OpenAIEmbeddingFunction(api_key=key, model=model)


def get_embedding_function(
    provider: str = "local",
    model: Optional[str] = None,
):
    """
    임베딩 함수 선택.
    provider: 'openai' → OpenAI API 사용
              'local'  → SentenceTransformer 사용 (기본값)
    """
    if provider == "openai":
        _model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return get_openai_embedding_function(model=_model)

    # local
    _model = model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return get_local_embedding_function(model_name=_model)


# =============================================================================
# ChromaDB 클라이언트 캐시
# =============================================================================
# PersistentClient는 SQLite 파일을 열고 내부 인덱스를 로딩하는 비용이 있다.
# 챗봇 환경에서는 쿼리마다 새 클라이언트를 만들면 응답 지연이 쌓인다.
# persist_dir 경로를 키로 최초 1회만 초기화하고 이후엔 재사용한다.
#
# [스레드 안전성]
# CPython의 GIL이 딕셔너리 읽기/쓰기를 원자적으로 보호한다.
# ChromaDB PersistentClient 자체도 내부적으로 동시 접근을 처리한다.
# Phase 1 단일 프로세스 환경에서는 추가 Lock 불필요.
_chroma_client_cache: Dict[str, chromadb.PersistentClient] = {}


def _get_or_init_client(persist_dir: str) -> chromadb.PersistentClient:
    """
    ChromaDB PersistentClient를 경로별로 캐싱해 반환한다.
    같은 persist_dir에 대해 프로세스 수명 동안 단 하나의 클라이언트만 유지한다.
    """
    if persist_dir not in _chroma_client_cache:
        logger.debug("ChromaDB 클라이언트 초기화: %s", persist_dir)
        _chroma_client_cache[persist_dir] = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client_cache[persist_dir]


# =============================================================================
# Collection Management
# =============================================================================
def get_or_create_collection(
    persist_dir: str,
    collection_name: str,
    embedding_provider: str = "local",
    embedding_model: Optional[str] = None,
):
    """
    ChromaDB 컬렉션 가져오거나 생성.
    embedding_provider가 달라지면 반드시 다른 collection_name 사용 또는
    reset_collection()으로 기존 컬렉션을 먼저 삭제해야 함 (차원 불일치 방지).
    """
    client = _get_or_init_client(persist_dir)
    ef = get_embedding_function(provider=embedding_provider, model=embedding_model)

    # hnsw:space="cosine" → 벡터 간 거리 계산 방식을 코사인 거리로 지정.
    # 유클리드 거리(l2)와 달리 벡터 크기(길이)를 무시하고 방향(각도)만 비교하므로,
    # 청크 길이가 들쑥날쑥한 한국어 문서 임베딩에 더 안정적.
    # 컬렉션 생성 시에만 적용되며, 이후 변경 불가(바꾸려면 reset_collection() 필요).
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def reset_collection(
    collection_name: str,
    persist_dir: str,
    embedding_provider: str = "openai",
    embedding_model: Optional[str] = None,
):
    """
    컬렉션 삭제 후 재생성.
    임베딩 모델(provider)을 교체할 때 반드시 호출해야 함.
    기존 벡터와 새 벡터는 차원이 달라 같은 컬렉션에 공존 불가.
    """
    # 캐싱된 클라이언트를 재사용한다.
    # reset_collection에서 별도 클라이언트를 만들면
    # 같은 persist_dir에 대해 2개의 인스턴스가 생겨 파일 lock 충돌 위험이 있다.
    client = _get_or_init_client(persist_dir)

    try:
        client.delete_collection(name=collection_name)
        logger.info("기존 컬렉션 삭제 완료: %s", collection_name)
    except Exception:
        logger.info("삭제할 컬렉션 없음 (신규 생성): %s", collection_name)

    return get_or_create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )


# =============================================================================
# Upsert
# =============================================================================
def upsert_chunks_to_chroma(
    chunks: List[Dict[str, Any]],
    collection_name: str = "ninewatt_company",
    persist_dir: str = "data/vector_store/chroma",
    batch_size: int = 50,
    embedding_provider: str = "local",
    embedding_model: Optional[str] = None,
    default_doc_type: Optional[str] = None,
):
    """
    청크 목록을 ChromaDB에 upsert (중복 chunk_id는 덮어씀).
    배치 단위로 처리하여 메모리 부하 방지.
    """
    collection = get_or_create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )

    items = prepare_chroma_items(chunks=chunks, default_doc_type=default_doc_type)

    total_chunks = len(items["ids"])
    logger.info("총 %d개의 청크를 DB에 적재 시작...", total_chunks)

    if total_chunks == 0:
        logger.warning("적재할 청크가 없습니다.")
        return collection

    # 배치 단위로 나눠서 upsert (OpenAI API rate limit 고려)
    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)

        collection.upsert(
            ids=items["ids"][i:end_idx],
            documents=items["documents"][i:end_idx],
            metadatas=items["metadatas"][i:end_idx],
        )
        logger.info("Upsert 완료: %d ~ %d / %d", i + 1, end_idx, total_chunks)

    logger.info("ChromaDB 적재 완료! 현재 컬렉션 총 청크 수: %d", collection.count())
    return collection


# =============================================================================
# Query
# =============================================================================
def query_collection(
    query_text: str,
    collection_name: str = "ninewatt_company",
    persist_dir: str = "data/vector_store/chroma",
    embedding_provider: str = "local",
    embedding_model: Optional[str] = None,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
):
    """
    자연어 쿼리로 유사 청크 검색.
    where: ChromaDB 메타데이터 필터 (예: {"doc_type": "company"})
    """
    collection = get_or_create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
    )
    return results


def print_query_summary(query_text: str, results: Dict[str, Any]) -> None:
    """검색 결과를 rank/id/header/distance/본문 형식으로 출력"""
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
        header = top_metas[i].get("header", "") if i < len(top_metas) and top_metas[i] else ""
        dist = top_distances[i] if i < len(top_distances) else None
        full_text = top_docs[i] if i < len(top_docs) else ""

        print(f"\n  - rank {i+1}")
        print(f"    id       : {chunk_id}")
        print(f"    header   : {header}")
        print(f"    distance : {dist:.4f}" if dist is not None else "    distance : -")
        print("    full_text:")
        print(full_text)
        print("    " + "-" * 80)


# =============================================================================
# CLI Entry (쿼리 테스트용 단독 실행)
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    chroma_dir = str(get_chroma_dir())

    # .env에서 임베딩 설정 읽기
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "ninewatt_company")

    logger.info("Chroma 저장 경로: %s", chroma_dir)
    logger.info("임베딩 provider: %s | 컬렉션: %s", embedding_provider, collection_name)

    try:
        # CLI 확인 용도도 캐싱 클라이언트를 사용해 인스턴스를 통일한다
        client = _get_or_init_client(chroma_dir)
        col = client.get_collection(collection_name)
        print(f"\n[컬렉션 현황] {collection_name}: {col.count()}개 청크")
    except Exception as e:
        print(f"컬렉션을 찾을 수 없습니다: {e}")
        print("먼저 company_ingest.py를 실행하여 데이터를 적재하세요.")
        return

    # 쿼리 테스트는 RAG 구현 후 별도 스크립트에서 진행
    print("\n[sanity check 완료] 적재 확인 후 rag_chain.py에서 end-to-end 테스트 진행")


if __name__ == "__main__":
    main()
