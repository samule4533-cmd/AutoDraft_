# src/rag_chain.py
#
# 역할: QueryContext를 받아 RAG를 실행하고 RagResult를 반환한다.
# 질문 전처리(이해/재작성/라우팅)는 query_processor.py가 담당한다.
# 이 파일은 검색 → 필터 → 생성 → 출처 인용에만 집중한다.

import logging
import os
import re
import uuid
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

from src.llm_api import GeminiAPIError, get_client
from src.query_processor import QueryContext, QueryType, process_query, trim_history
from src.vector_db import get_or_create_collection, query_collection

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Config
# =============================================================================
RAG_CONFIG = {
    "collection_name":    os.getenv("CHROMA_COLLECTION_NAME", "ninewatt_company"),
    "persist_dir":        str(PROJECT_ROOT / "data" / "vector_store" / "chroma"),
    "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "local"),
    "gemini_model":       os.getenv("GEMINI_RAG_MODEL", "gemini-2.0-flash"),
    "n_results":          10,
    "distance_threshold": 0.55,
    "max_context_chars":  6000,
}

# fallback 메시지
_MSG_NO_DOCS         = "사내 문서에서 해당 내용을 찾을 수 없습니다."
_MSG_RETRIEVAL_ERROR = "문서 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
_MSG_LLM_ERROR       = "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

SYSTEM_PROMPT = """\
당신은 회사 내부 문서를 기반으로 질문에 답하는 어시스턴트입니다.

[규칙]
1. 반드시 아래 [참고 문서]에 있는 내용에만 근거하여 답하라.
2. 문서에 없는 내용은 절대 추측하거나 지어내지 말고, "해당 내용이 문서에 없습니다"라고 답하라.
3. 답변 내 사실마다 [출처: {헤더명}] 형태로 표기하라.
4. 표, 수치, 날짜는 원문 그대로 인용하라.
5. 한국어로 답하라.
6. 핵심만 간결하게 3~5문장 이내로 답하라. 불필요한 배경 설명이나 서론은 생략한다.
"""


# =============================================================================
# 반환 타입
# =============================================================================
class Citation(BaseModel):
    chunk_id:    str
    header:      str
    source_file: str
    distance:    float | None


class RagResult(BaseModel):
    answer:             str
    citations:          list[Citation]
    query_type:         str            # "meta" | "existence" | "content"
    used_query:         str            # 실제 검색에 사용된 쿼리
    reformulated_query: str | None     # 지시어 재작성 결과 (디버깅)
    understood_query:   str | None     # 검색 최적화 변환 결과 (디버깅)
    retrieved_count:    int
    passed_threshold:   int
    top_distance:       float | None
    fallback:           bool
    fallback_reason:    str | None     # "no_docs" | "retrieval_error" | "llm_error"


def _make_fallback(
    ctx: QueryContext,
    reason: str,
    message: str,
    retrieved_count: int = 0,
    passed_threshold: int = 0,
    top_distance: float | None = None,
) -> RagResult:
    """fallback RagResult 생성 헬퍼. 반복 코드를 줄인다."""
    return RagResult(
        answer=             message,
        citations=          [],
        query_type=         ctx.query_type,
        used_query=         ctx.search_query,
        reformulated_query= ctx.reformulated,
        understood_query=   ctx.understood,
        retrieved_count=    retrieved_count,
        passed_threshold=   passed_threshold,
        top_distance=       top_distance,
        fallback=           True,
        fallback_reason=    reason,
    )


# =============================================================================
# Meta / Existence 핸들러
# =============================================================================
def handle_meta_query(ctx: QueryContext) -> RagResult:
    """
    "파일 몇 개야?", "어떤 문서들 있어?" 같은 DB 현황 질문을 처리한다.
    ChromaDB 메타데이터를 직접 조회해 파일 목록과 청크 수를 반환한다.
    벡터 검색을 거치지 않으므로 빠르고 정확하다.
    """
    try:
        col = get_or_create_collection(
            persist_dir=RAG_CONFIG["persist_dir"],
            collection_name=RAG_CONFIG["collection_name"],
            embedding_provider=RAG_CONFIG["embedding_provider"],
        )
        result = col.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        file_counts: dict[str, int] = {}
        for m in metadatas:
            f = (m or {}).get("source_file", "알 수 없음")
            file_counts[f] = file_counts.get(f, 0) + 1

        total_chunks = col.count()
        lines = [f"현재 {len(file_counts)}개 문서가 등록되어 있습니다.\n"]
        for i, fname in enumerate(sorted(file_counts.keys()), 1):
            lines.append(f"{i}. {fname}")

        return RagResult(
            answer=             "\n".join(lines),
            citations=          [],
            query_type=         ctx.query_type,
            used_query=         ctx.search_query,
            reformulated_query= ctx.reformulated,
            understood_query=   ctx.understood,
            retrieved_count=    total_chunks,
            passed_threshold=   total_chunks,
            top_distance=       None,
            fallback=           False,
            fallback_reason=    None,
        )
    except Exception as e:
        logger.error("meta query 처리 실패: %s", e)
        return _make_fallback(ctx, "retrieval_error", _MSG_RETRIEVAL_ERROR)


def handle_existence_query(ctx: QueryContext, req_id: str) -> RagResult:
    """
    "에너지 관련 특허 있어?" 같은 존재 확인 질문을 처리한다.
    벡터 검색으로 관련 문서를 찾고, threshold 통과 여부로 존재/부재를 판단한다.
    """
    try:
        raw = query_collection(
            query_text=ctx.search_query,
            collection_name=RAG_CONFIG["collection_name"],
            persist_dir=RAG_CONFIG["persist_dir"],
            embedding_provider=RAG_CONFIG["embedding_provider"],
            n_results=RAG_CONFIG["n_results"],
        )
        distances = (raw.get("distances") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]

        passed = [
            (metadatas[i], distances[i])
            for i in range(len(distances))
            if distances[i] is not None and distances[i] <= RAG_CONFIG["distance_threshold"]
        ]

        if not passed:
            return RagResult(
                answer=             "관련 문서가 등록되어 있지 않습니다.",
                citations=          [],
                query_type=         ctx.query_type,
                used_query=         ctx.search_query,
                reformulated_query= ctx.reformulated,
                understood_query=   ctx.understood,
                retrieved_count=    len(distances),
                passed_threshold=   0,
                top_distance=       distances[0] if distances else None,
                fallback=           False,
                fallback_reason=    None,
            )

        # 관련 파일 목록 중복 제거
        files = []
        seen = set()
        for meta, _ in passed:
            f = (meta or {}).get("source_file", "")
            if f and f not in seen:
                seen.add(f)
                files.append(f)

        lines = [f"관련 문서 {len(files)}개가 있습니다.\n"]
        for i, f in enumerate(files, 1):
            lines.append(f"{i}. {f}")

        return RagResult(
            answer=             "\n".join(lines),
            citations=          [],
            query_type=         ctx.query_type,
            used_query=         ctx.search_query,
            reformulated_query= ctx.reformulated,
            understood_query=   ctx.understood,
            retrieved_count=    len(distances),
            passed_threshold=   len(passed),
            top_distance=       distances[0] if distances else None,
            fallback=           False,
            fallback_reason=    None,
        )
    except Exception as e:
        logger.error("[%s] existence query 처리 실패: %s", req_id, e)
        return _make_fallback(ctx, "retrieval_error", _MSG_RETRIEVAL_ERROR)


# =============================================================================
# Content RAG 파이프라인
# =============================================================================
def retrieve(
    query: str,
    req_id: str,
    n_results: int = RAG_CONFIG["n_results"],
    filters: dict | None = None,
) -> list[dict]:
    """ChromaDB에서 유사 청크를 검색하고 평탄화된 list[dict]로 반환한다."""
    cfg = RAG_CONFIG
    raw = query_collection(
        query_text=query,
        collection_name=cfg["collection_name"],
        persist_dir=cfg["persist_dir"],
        embedding_provider=cfg["embedding_provider"],
        n_results=n_results,
        where=filters,
    )

    ids       = raw.get("ids",       [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    chunks = []
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        chunks.append({
            "chunk_id":    chunk_id,
            "text":        documents[i] if i < len(documents) else "",
            "metadata":    meta,
            "distance":    distances[i] if i < len(distances) else None,
            "header":      (meta or {}).get("header", ""),
            "source_file": (meta or {}).get("source_file", ""),
        })

    logger.info("[%s] retrieve 완료: %d개 청크 반환", req_id, len(chunks))
    return chunks


def filter_by_threshold(
    chunks: list[dict],
    req_id: str,
    threshold: float = RAG_CONFIG["distance_threshold"],
) -> list[dict]:
    """distance > threshold인 청크를 제거한다."""
    passed = [c for c in chunks if c["distance"] is not None and c["distance"] <= threshold]
    logger.info(
        "[%s] threshold=%.2f 적용 → %d/%d 청크 통과",
        req_id, threshold, len(passed), len(chunks),
    )
    return passed


def build_context_block(
    chunks: list[dict],
    req_id: str,
    max_chars: int = RAG_CONFIG["max_context_chars"],
) -> str:
    """통과한 청크들을 LLM 컨텍스트 문자열로 조합한다. 상위 rank 우선 보존."""
    blocks = []
    total  = 0
    for i, chunk in enumerate(chunks):
        clean_header = re.sub(r"^#{1,6}\s*", "", chunk["header"]).strip()
        block = (
            f"[문서 {i + 1}]\n"
            f"출처: {clean_header} | {chunk['source_file']}\n"
            f"{chunk['text']}\n"
        )
        if total + len(block) > max_chars:
            logger.info("[%s] max_chars 도달 → %d/%d 청크만 포함", req_id, i, len(chunks))
            break
        blocks.append(block)
        total += len(block)
    return "\n---\n".join(blocks)


def build_prompt(query: str, context: str, chat_history: list | None = None) -> str:
    """시스템 지시 + 이전 대화 + 참고 문서 + 질문을 하나의 프롬프트로 조합한다."""
    history_block = ""
    if chat_history:
        lines = [
            f"{'사용자' if m.get('role') == 'user' else '어시스턴트'}: {m.get('content', '')}"
            for m in chat_history
        ]
        history_block = "\n[이전 대화]\n" + "\n".join(lines) + "\n"

    return (
        f"{SYSTEM_PROMPT}"
        f"{history_block}\n"
        f"[참고 문서]\n{context}\n\n"
        f"[질문]\n{query}"
    )


def generate_answer(prompt: str, req_id: str, max_retries: int = 3) -> str:
    """
    Gemini에 프롬프트를 전달하고 텍스트 답변을 반환한다.
    429(rate limit) 에러는 지수 백오프로 최대 max_retries회 재시도한다.
    """
    client = get_client()
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=RAG_CONFIG["gemini_model"],
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            answer = (getattr(resp, "text", "") or "").strip()
            if not answer:
                raise GeminiAPIError("LLM이 빈 응답을 반환했습니다.")
            logger.info("[%s] LLM 응답 수신: %d자", req_id, len(answer))
            return answer
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_rate_limit = (
                "429" in str(e) or "quota" in err_str
                or "rate" in err_str or "resource_exhausted" in err_str
            )
            if is_rate_limit:
                logger.error("[%s] Gemini rate limit 초과 → 즉시 종료: %s", req_id, e)
                raise
            raise

    raise last_exc  # type: ignore[misc]


def format_citations(chunks: list[dict]) -> list[Citation]:
    """(header, source_file) 기준 중복 제거 후 Citation 리스트 반환."""
    seen, result = set(), []
    for c in chunks:
        key = (c["header"], c["source_file"])
        if key not in seen:
            seen.add(key)
            result.append(Citation(
                chunk_id=    c["chunk_id"],
                header=      re.sub(r"^#{1,6}\s*", "", c["header"]).strip(),
                source_file= c["source_file"],
                distance=    c["distance"],
            ))
    return result


def _handle_content(
    ctx: QueryContext,
    req_id: str,
    filters: dict | None,
    chat_history: list | None = None,
) -> RagResult:
    """일반 RAG 파이프라인. retrieve → filter → build → generate → cite."""
    # 검색
    try:
        chunks = retrieve(ctx.search_query, req_id, filters=filters)
    except Exception as e:
        logger.error("[%s] ChromaDB 검색 실패: %s", req_id, e)
        return _make_fallback(ctx, "retrieval_error", _MSG_RETRIEVAL_ERROR)

    # 필터
    passed = filter_by_threshold(chunks, req_id)
    if not passed:
        logger.info("[%s] 통과 청크 0개 → fallback", req_id)
        return _make_fallback(
            ctx, "no_docs", _MSG_NO_DOCS,
            retrieved_count=len(chunks),
            top_distance=chunks[0]["distance"] if chunks else None,
        )

    # 컨텍스트 + 프롬프트 (원본 질문으로 자연스럽게 답변, chat_history 반영)
    context = build_context_block(passed, req_id)
    history = trim_history(chat_history) if chat_history else None
    prompt  = build_prompt(ctx.original_query, context, history)

    # LLM 호출
    try:
        answer = generate_answer(prompt, req_id)
    except Exception as e:
        logger.error("[%s] LLM 호출 실패: %s", req_id, e)
        return _make_fallback(
            ctx, "llm_error", _MSG_LLM_ERROR,
            retrieved_count=len(chunks),
            passed_threshold=len(passed),
            top_distance=passed[0]["distance"],
        )

    result = RagResult(
        answer=             answer,
        citations=          format_citations(passed),
        query_type=         ctx.query_type,
        used_query=         ctx.search_query,
        reformulated_query= ctx.reformulated,
        understood_query=   ctx.understood,
        retrieved_count=    len(chunks),
        passed_threshold=   len(passed),
        top_distance=       passed[0]["distance"],
        fallback=           False,
        fallback_reason=    None,
    )
    logger.info(
        "[%s] 완료 | type=%s retrieved=%d passed=%d top_dist=%.4f",
        req_id, ctx.query_type, result.retrieved_count,
        result.passed_threshold, result.top_distance or 0.0,
    )
    return result


# =============================================================================
# 메인 진입점
# =============================================================================
def ask(
    query: str,
    chat_history: list | None = None,
    filters: dict | None = None,
) -> RagResult:
    """
    사용자 질문을 받아 RagResult를 반환한다.

    처리 흐름:
      1. process_query() → QueryContext (이해/재작성/라우팅)
      2. query_type에 따라 분기:
           meta      → handle_meta_query()
           existence → handle_existence_query()
           content   → _handle_content()
    """
    req_id = uuid.uuid4().hex[:8]
    logger.info("[%s] 질문 수신: %.80s", req_id, query)

    ctx = process_query(query, chat_history, req_id)
    logger.info(
        "[%s] QueryContext | type=%s reformulated=%s understood=%s search_query=%.60s",
        req_id, ctx.query_type,
        ctx.reformulated is not None,
        ctx.understood is not None,
        ctx.search_query,
    )

    if ctx.query_type == "meta":
        return handle_meta_query(ctx)
    if ctx.query_type == "existence":
        return handle_existence_query(ctx, req_id)
    return _handle_content(ctx, req_id, filters, chat_history)


# =============================================================================
# CLI — 단독 실행 테스트
# =============================================================================
def _print_result(result: RagResult) -> None:
    print("\n" + "=" * 60)
    print(f"[query_type: {result.query_type}]")
    if result.fallback:
        print(f"[FALLBACK: {result.fallback_reason}]")
    if result.reformulated_query:
        print(f"[Reformulation] {result.reformulated_query}")
    if result.understood_query:
        print(f"[Understanding] {result.understood_query}")
    print(f"\n[답변]\n{result.answer}")
    print(f"\n[검색 통계]")
    print(f"  사용된 query:    {result.used_query}")
    print(f"  검색된 청크:     {result.retrieved_count}개")
    print(f"  threshold 통과:  {result.passed_threshold}개")
    if result.top_distance is not None:
        print(f"  최근접 distance: {result.top_distance:.4f}")
    if result.citations:
        print("\n[출처]")
        for c in result.citations:
            dist = f" (dist={c.distance:.4f})" if c.distance is not None else ""
            print(f"  - {c.header} | {c.source_file}{dist}")
    print("=" * 60)


_TEST_CASES = [
    ("메타 질문",               "파일 몇 개 들어있어?"),
    ("메타 질문",               "어떤 문서들 있어?"),
    ("존재 확인",               "에너지 관련 특허 있어?"),
    ("존재 확인",               "RAG 관련 자료 있나요?"),
    ("문서에 있는 질문",         "건물 에너지 모델링 자동화 시스템은 무엇이야?"),
    ("수치/날짜 질문",           "건물 에너지 모델링 자동화 시스템 특허의 출원일은 언제야?"),
    ("구어체 질문",             "출원일 알려줘"),
    ("문서에 없는 질문",         "직원 복지 정책이 나와있어?"),
    ("엉뚱한 질문",             "서울날씨 알려줘"),
]


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        for label, q in _TEST_CASES:
            print(f"\n{'#' * 60}")
            print(f"[케이스] {label}")
            print(f"[질문]   {q}")
            _print_result(ask(q))
    else:
        q = sys.argv[1] if len(sys.argv) > 1 else "회사 주요 특허는 무엇인가요?"
        print(f"\n질문: {q}")
        _print_result(ask(q))
