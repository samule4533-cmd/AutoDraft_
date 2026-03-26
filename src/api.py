# src/api.py
#
# FastAPI 앱 진입점.
# rag_chain.ask()를 HTTP로 노출하는 얇은 레이어.
# 비즈니스 로직은 rag_chain.py에서 처리하고, 이 파일은 입출력 변환만 담당한다.

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import src.bm25_retriever as bm25_retriever
import src.parent_store as parent_store
# import src.reranker as reranker  # GPU 서버 구축 후 활성화
from src.pdf_parser import parse_single_pdf
from src.rag_chain import RAG_CONFIG, RagResult, ask
from src.summary_service import get_summaries
from src.vector_db import get_or_create_collection, upsert_chunks_to_chroma

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan — 서버 시작/종료 이벤트
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("warm-up 시작: ChromaDB 클라이언트 + 임베딩 모델 로딩")
    try:
        col = get_or_create_collection(
            persist_dir=RAG_CONFIG["persist_dir"],
            collection_name=RAG_CONFIG["collection_name"],
            embedding_provider=RAG_CONFIG["embedding_provider"],
        )
        logger.info("warm-up 완료: ChromaDB")
    except Exception as e:
        col = None
        logger.warning("ChromaDB warm-up 실패 (서버는 계속 실행됨): %s", e)

    # BM25 인덱스 빌드 — ChromaDB 청크 전체를 읽어 인메모리 인덱스 생성
    if col is not None:
        try:
            bm25_retriever.build_index(col)
        except Exception as e:
            logger.warning("BM25 인덱스 빌드 실패 (서버는 계속 실행됨): %s", e)

    # parent_store 로드 — parent_index.json → 인메모리 인덱스 빌드
    try:
        parent_store.load()
    except Exception as e:
        logger.warning("parent_store 로드 실패 (서버는 계속 실행됨): %s", e)

    # jina reranker 모델 로드 (~280MB, 최초 실행 시 HuggingFace에서 다운로드)
    # try:
    #     reranker.load_model()
    # except Exception as e:
    #     logger.warning("reranker 모델 로드 실패 (서버는 계속 실행됨): %s", e)

    yield
    logger.info("서버 종료")


# =============================================================================
# FastAPI 앱
# =============================================================================
app = FastAPI(
    title="AutoDraft RAG API",
    description="사내 문서 기반 질의응답 API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 설정.
# 개발 단계에서는 allow_origins=["*"]로 전체 허용.
# 운영 배포 시 allow_origins에 실제 프론트엔드 도메인만 지정해야 한다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 입력 스키마
# =============================================================================
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    chat_history: list[ChatMessage] | None = None
    filters: dict | None = None


# =============================================================================
# 엔드포인트
# =============================================================================
@app.get("/health")
def health():
    """
    서버 상태 확인용 엔드포인트.
    ChromaDB 컬렉션 접근 가능 여부와 적재된 청크 수를 반환한다.
    """
    try:
        col = get_or_create_collection(
            persist_dir=RAG_CONFIG["persist_dir"],
            collection_name=RAG_CONFIG["collection_name"],
            embedding_provider=RAG_CONFIG["embedding_provider"],
        )
        return {
            "status": "ok",
            "collection": RAG_CONFIG["collection_name"],
            "chunk_count": col.count(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ChromaDB 접근 실패: {e}")


@app.post("/chat", response_model=RagResult)
def chat(req: ChatRequest):
    """
    사용자 질문을 받아 RAG 답변을 반환한다.
    chat_history가 있으면 이전 대화 맥락을 반영하고,
    지시어가 포함된 질문은 자동으로 reformulation을 시도한다.
    """
    logger.info(
        "POST /chat | query=%.60s | history_len=%d",
        req.query,
        len(req.chat_history or []),
    )

    history = [m.model_dump() for m in req.chat_history] if req.chat_history else None

    try:
        result = ask(query=req.query, chat_history=history, filters=req.filters)
    except Exception as e:
        logger.error("ask() 예외: %s", e)
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

    return result


# =============================================================================
# Drive 연동 엔드포인트 (AutoDraft_ingest → AutoDraft_clean)
# =============================================================================
@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    file_id: str = Form(...),
    file_name: str = Form(...),
):
    """
    AutoDraft_ingest 서비스에서 Drive 파일을 전달받아
    파싱 → 청킹 → ChromaDB 적재까지 수행한다.
    같은 file_id로 재전송 시 기존 청크를 덮어쓴다 (멱등성).
    """
    logger.info("POST /ingest | file_id=%s | file_name=%s", file_id, file_name)

    raw_bytes = await file.read()

    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / "data" / "raw" / "company" / "drive"
    output_root = project_root / "data" / "processed" / "parsing_result_company"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    pdf_path = input_dir / file_name
    pdf_path.write_bytes(raw_bytes)

    try:
        chunks = await parse_single_pdf(
            source_pdf=pdf_path,
            input_dir=input_dir,
            output_root=output_root,
            doc_type="patent",
        )
    except Exception as e:
        logger.error("파싱 실패: %s | %s", file_name, e)
        raise HTTPException(status_code=500, detail=f"파싱 실패: {e}")

    # file_id를 각 청크 메타데이터에 추가 (삭제 시 필터링 기준)
    for chunk in chunks:
        chunk.setdefault("metadata", {})["file_id"] = file_id

    # chunk_type별 라우팅 — parent는 ChromaDB 제외, parent_index.json으로 분리
    chroma_chunks  = [c for c in chunks if c.get("chunk_type", "section") in ("section", "child")]
    new_parents    = [c for c in chunks if c.get("chunk_type") == "parent"]
    child_map      = {c["chunk_id"]: c for c in chunks if c.get("chunk_type") == "child"}

    # 기존 청크 삭제 후 upsert (재처리 시 중복 방지)
    col = get_or_create_collection(
        persist_dir=RAG_CONFIG["persist_dir"],
        collection_name=RAG_CONFIG["collection_name"],
        embedding_provider=RAG_CONFIG["embedding_provider"],
    )
    try:
        col.delete(where={"file_id": file_id})
    except Exception:
        pass  # 기존 청크 없으면 무시

    upsert_chunks_to_chroma(
        chunks=chroma_chunks,
        collection_name=RAG_CONFIG["collection_name"],
        persist_dir=RAG_CONFIG["persist_dir"],
        embedding_provider=RAG_CONFIG["embedding_provider"],
        default_doc_type="company",
    )

    # ChromaDB 적재 완료 후 raw PDF 삭제.
    pdf_path.unlink(missing_ok=True)

    # parent_index.json 병합 + parent_store 인메모리 재로드
    if new_parents:
        try:
            parent_store.merge_parents(new_parents, child_map)
        except Exception as e:
            logger.warning("parent_store 병합 실패 (context 확장 불가): %s", e)

    # BM25 인덱스 재빌드 — 새 청크가 추가됐으므로 인메모리 인덱스를 갱신한다.
    try:
        bm25_retriever.rebuild_index(col)
    except Exception as e:
        logger.warning("BM25 재빌드 실패 (검색은 이전 인덱스로 동작): %s", e)

    logger.info(
        "적재 완료: file_id=%s | chroma=%d parent=%d",
        file_id, len(chroma_chunks), len(new_parents),
    )
    return {"chunk_count": len(chroma_chunks), "file_id": file_id, "file_name": file_name}


class RenameRequest(BaseModel):
    file_name: str = Field(min_length=1)


@app.patch("/ingest/{file_id}")
def rename_ingest(file_id: str, req: RenameRequest):
    """
    Drive에서 파일 제목만 변경된 경우 ChromaDB 메타데이터의 source_file만 업데이트한다.
    재파싱/재임베딩 없이 파일명만 교체하므로 API 비용이 발생하지 않는다.
    """
    logger.info("PATCH /ingest/%s | new_name=%s", file_id, req.file_name)
    try:
        col = get_or_create_collection(
            persist_dir=RAG_CONFIG["persist_dir"],
            collection_name=RAG_CONFIG["collection_name"],
            embedding_provider=RAG_CONFIG["embedding_provider"],
        )
        result = col.get(where={"file_id": file_id}, include=["metadatas"])
        ids = result["ids"]

        if not ids:
            raise HTTPException(status_code=404, detail=f"file_id={file_id}에 해당하는 청크 없음")

        updated_metadatas = []
        for meta in result["metadatas"]:
            updated_meta = dict(meta)
            updated_meta["source_file"] = req.file_name
            updated_metadatas.append(updated_meta)

        col.update(ids=ids, metadatas=updated_metadatas)
        logger.info("메타데이터 업데이트 완료: file_id=%s | chunks=%d", file_id, len(ids))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("메타데이터 업데이트 실패: %s | %s", file_id, e)
        raise HTTPException(status_code=500, detail=f"메타데이터 업데이트 실패: {e}")

    # BM25 재빌드 — _chunks의 source_file 메타데이터를 최신 상태로 갱신
    try:
        bm25_retriever.rebuild_index(col)
    except Exception as e:
        logger.warning("BM25 재빌드 실패 (이전 인덱스로 동작): %s", e)

    return {"updated_chunks": len(ids), "file_id": file_id, "file_name": req.file_name}


@app.delete("/ingest/{file_id}")
def delete_ingest(file_id: str):
    """
    Drive에서 파일이 삭제됐을 때 ChromaDB에서 해당 파일의 청크를 제거한다.
    """
    logger.info("DELETE /ingest/%s", file_id)
    try:
        col = get_or_create_collection(
            persist_dir=RAG_CONFIG["persist_dir"],
            collection_name=RAG_CONFIG["collection_name"],
            embedding_provider=RAG_CONFIG["embedding_provider"],
        )
        col.delete(where={"file_id": file_id})
    except Exception as e:
        logger.error("청크 삭제 실패: %s | %s", file_id, e)
        raise HTTPException(status_code=500, detail=f"삭제 실패: {e}")

    # parent_index.json에서 해당 문서 parent 항목 제거 + 인메모리 재로드
    try:
        parent_store.remove_by_document(file_id)
    except Exception as e:
        logger.warning("parent_store 제거 실패 (이전 상태로 동작): %s", e)

    # BM25 재빌드 — 삭제된 청크가 검색 결과에 계속 등장하는 것을 방지
    try:
        col = get_or_create_collection(
            persist_dir=RAG_CONFIG["persist_dir"],
            collection_name=RAG_CONFIG["collection_name"],
            embedding_provider=RAG_CONFIG["embedding_provider"],
        )
        bm25_retriever.rebuild_index(col)
    except Exception as e:
        logger.warning("BM25 재빌드 실패 (이전 인덱스로 동작): %s", e)

    return {"deleted": True, "file_id": file_id}


# =============================================================================
# 출력 스키마
# =============================================================================
class SummaryItem(BaseModel):
    index:    int
    filename: str
    summary:  str


# =============================================================================
# 요약 엔드포인트
# =============================================================================
@app.get("/summaries", response_model=list[SummaryItem])
def summaries():
    """
    전체 사내 문서 AI 요약 목록을 반환한다.
    캐시가 있으면 즉시 반환하고, 새 파일이 추가된 경우에만 Gemini를 호출한다.
    """
    logger.info("GET /summaries")
    try:
        return get_summaries()
    except Exception as e:
        logger.error("요약 생성 실패: %s", e)
        raise HTTPException(status_code=500, detail="요약 생성 중 오류가 발생했습니다.")


# =============================================================================
# 직접 실행
# =============================================================================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
