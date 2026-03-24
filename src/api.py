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

from src.pdf_parser import parse_single_pdf
from src.rag_chain import RAG_CONFIG, RagResult, ask
from src.summary_service import get_summaries
from src.vector_db import get_or_create_collection, upsert_chunks_to_chroma

load_dotenv()

# uvicorn CLI로 실행할 때도 로그가 보이도록 모듈 로딩 시점에 설정한다.
# if __name__ == "__main__" 안에 두면 python api.py 직접 실행 시에만 적용된다.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan — 서버 시작/종료 이벤트
# =============================================================================
# FastAPI 권장 방식: @app.on_event 대신 lifespan 컨텍스트 매니저 사용.
# yield 이전 = 시작 시 실행 / yield 이후 = 종료 시 실행.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 임베딩 모델과 ChromaDB 클라이언트를 미리 로딩한다.
    # 첫 요청 시 로딩하면 SentenceTransformer 모델 로딩으로 수 초 지연이 발생한다.
    # 여기서 미리 호출하면 get_or_create_collection() 내부에서
    # ChromaDB 클라이언트(_chroma_client_cache)와 임베딩 함수가 초기화된다.
    logger.info("warm-up 시작: ChromaDB 클라이언트 + 임베딩 모델 로딩")
    try:
        get_or_create_collection(
            persist_dir=RAG_CONFIG["persist_dir"],
            collection_name=RAG_CONFIG["collection_name"],
            embedding_provider=RAG_CONFIG["embedding_provider"],
        )
        logger.info("warm-up 완료")
    except Exception as e:
        # 데이터가 아직 적재되지 않은 환경에서도 서버는 뜰 수 있어야 한다.
        # 오류를 로그로 남기고 서버 구동은 계속 진행한다.
        logger.warning("warm-up 실패 (서버는 계속 실행됨): %s", e)
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
    # role은 "user" 또는 "assistant"만 허용.
    # 그 외 값이 들어오면 FastAPI가 422 Unprocessable Entity를 자동 반환한다.
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    # min_length=1: 빈 문자열 차단 (환각 유발 방지)
    # max_length=1000: 지나치게 긴 입력이 reformulation + retrieve 두 번의 LLM 호출을 유발하는 것을 방어
    query: str = Field(min_length=1, max_length=1000)

    # chat_history: 이전 대화 맥락. 없으면 단발성 질문으로 처리.
    # list 자체도 None 허용이므로 채팅 첫 시작 시 생략 가능하다.
    chat_history: list[ChatMessage] | None = None

    # filters: ChromaDB where 조건. Phase 3 메타데이터 필터링용.
    # 지금은 None으로 두면 되고, 나중에 {"doc_type": "company"} 형태로 확장한다.
    filters: dict | None = None


# =============================================================================
# 엔드포인트
# =============================================================================
@app.get("/health")
def health():
    """
    서버 상태 확인용 엔드포인트.
    ChromaDB 컬렉션 접근 가능 여부와 적재된 청크 수를 반환한다.
    배포 후 서버가 실제로 쿼리를 처리할 준비가 됐는지 즉시 확인할 수 있다.
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
        # 503: 서버는 살아있지만 DB에 접근할 수 없는 상태
        raise HTTPException(status_code=503, detail=f"ChromaDB 접근 실패: {e}")


@app.post("/chat", response_model=RagResult)
def chat(req: ChatRequest):
    """
    사용자 질문을 받아 RAG 답변을 반환한다.
    chat_history가 있으면 이전 대화 맥락을 반영하고,
    지시어가 포함된 질문은 자동으로 reformulation을 시도한다.

    응답의 fallback 필드가 True이면 정상 답변이 아님을 의미한다.
    fallback_reason으로 원인을 구분할 수 있다:
      - "no_docs": threshold를 통과한 관련 문서 없음
      - "retrieval_error": ChromaDB 검색 실패
      - "llm_error": Gemini 호출 실패
    """
    # API 진입점 로그: 어떤 요청이 들어왔는지 기록한다.
    # query는 앞 60자만 출력 (긴 질문이 로그를 도배하지 않도록)
    logger.info(
        "POST /chat | query=%.60s | history_len=%d",
        req.query,
        len(req.chat_history or []),
    )

    # Pydantic 모델 → dict 변환.
    # ask()는 내부에서 m.get("role"), m.get("content")로 dict에 접근하므로 변환이 필요하다.
    history = [m.model_dump() for m in req.chat_history] if req.chat_history else None

    try:
        result = ask(query=req.query, chat_history=history, filters=req.filters)
    except Exception as e:
        # ask() 내부에서 이미 fallback 처리를 하므로 여기까지 오는 경우는 드물다.
        # 예상치 못한 예외가 발생했을 때 500으로 반환한다.
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
        )
    except Exception as e:
        logger.error("파싱 실패: %s | %s", file_name, e)
        raise HTTPException(status_code=500, detail=f"파싱 실패: {e}")

    # file_id를 각 청크 메타데이터에 추가 (삭제 시 필터링 기준)
    for chunk in chunks:
        chunk.setdefault("metadata", {})["file_id"] = file_id

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
        chunks=chunks,
        collection_name=RAG_CONFIG["collection_name"],
        persist_dir=RAG_CONFIG["persist_dir"],
        embedding_provider=RAG_CONFIG["embedding_provider"],
        default_doc_type="company",
    )

    # ChromaDB 적재 완료 후 raw PDF 삭제.
    # 원본은 Drive에 있으므로 재처리 필요 시 다시 다운로드하면 된다.
    pdf_path.unlink(missing_ok=True)

    logger.info("적재 완료: file_id=%s | chunks=%d", file_id, len(chunks))
    return {"chunk_count": len(chunks), "file_id": file_id, "file_name": file_name}


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
    # reload=True: 코드 변경 시 서버 자동 재시작 (개발용)
    # 운영 배포 시 reload=False, workers=N으로 변경한다.
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
