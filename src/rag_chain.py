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
import src.bm25_retriever as bm25_retriever
# import src.reranker as reranker  # GPU 서버 구축 후 활성화

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
    # ── hybrid 검색 설정 ──────────────────────────────────────────────────────
    "bm25_top_k":         10,    # BM25에서 가져올 후보 수
    "vector_top_k":       10,    # 벡터 검색에서 가져올 후보 수
    "rrf_top_n":           10,    # RRF 후 LLM에 전달할 최종 청크 수
    "rrf_k":              60,    # RRF 상수. 표준값 60. 낮추면 상위 랭크 boost 강해짐
    # ── 품질 게이트 ───────────────────────────────────────────────────────────
    # 벡터 검색 결과의 최솟값(best_distance)이 이 값 초과면 관련 문서 없음으로 판단.
    # reranker 활성화 시: min_rerank_score(0.1) 기반 게이트로 교체 가능.
    "distance_threshold": 0.65,
    "max_context_chars":  6000,
}

# 응답 메시지 상수
_MSG_RETRIEVAL_ERROR = "문서 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
_MSG_LLM_ERROR       = "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
_MSG_NO_DOCS         = "사내 문서에서 해당 내용을 찾을 수 없습니다."

# 사내 문서 RAG 답변용 — 관련 청크가 있을 때 사용
SYSTEM_PROMPT = """\
당신은 회사 내부 문서를 기반으로 질문에 답하는 어시스턴트입니다.

[규칙]
1. 반드시 아래 [참고 문서]에 있는 내용에만 근거하여 답하라.
2. 문서에 없는 내용은 절대 추측하거나 지어내지 말고, "해당 내용이 문서에 없습니다"라고 답하라.
3. 본문에는 출처를 쓰지 말고, 답변 마지막에 아래 형식으로 출처를 모아서 표기하라.

📎 출처
- {파일명} > {헤더명}
- {파일명} > {헤더명}

4. 표, 수치, 날짜는 원문 그대로 인용하라.
5. 한국어로 답하라.
6. 핵심만 간결하게 3~5문장 이내로 답하라. 불필요한 배경 설명이나 서론은 생략한다.
7. "발표용", "기술적 관점", "사업성 관점", "쉽게 설명", "간단히" 등 형식·관점 지시가 포함된 경우,
   문서 내용을 그 형식에 맞게 재구성하여 답하라. 이는 내용 검색이 아니라 형식 변환 요청이다.
"""

# 인사·잡담용 — query_type == "greeting"일 때만 사용
_CHITCHAT_SYSTEM_PROMPT = """\
당신은 회사 내부 문서 어시스턴트입니다.
인사나 잡담에는 자연스럽고 친근하게 짧게 답하라.
한국어로 답하라.
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
    retrieved_count:    int            # BM25+벡터 union 후보 수 (RRF 입력 전 unique 청크 수)
    passed_threshold:   int            # 품질 게이트 통과 후 LLM 전달 수
    top_distance:       float | None   # 벡터 1위 청크의 cosine distance (있을 경우)
    top_rrf_score:      float | None   # RRF 1위 청크의 점수 (디버깅)
    debug_chunk_ids:    list[str]      # 품질 게이트 통과 전체 chunk_id (디버깅용 원본)
    fallback:           bool
    fallback_reason:    str | None     # "no_docs" | "low_confidence" | "retrieval_error" | "llm_error"


def _make_fallback(
    ctx: QueryContext,
    reason: str,
    message: str,
    retrieved_count: int = 0,
    passed_threshold: int = 0,
    top_distance: float | None = None,
    top_rrf_score: float | None = None,
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
        top_rrf_score=      top_rrf_score,
        debug_chunk_ids=    [],
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
            _m = m or {}
            f = _m.get("source_file") or _m.get("file_name", "알 수 없음")
            file_counts[f] = file_counts.get(f, 0) + 1

        total_chunks = col.count()
        count_only_keywords = ("갯수만", "개수만", "몇 개만", "몇개만", "숫자만", "수만 알")
        query_lower = ctx.original_query.lower()
        is_count_only = any(kw in query_lower for kw in count_only_keywords)

        if is_count_only:
            answer = f"현재 {len(file_counts)}개 문서가 등록되어 있습니다."
        else:
            lines = [f"현재 {len(file_counts)}개 문서가 등록되어 있습니다.\n"]
            for i, fname in enumerate(sorted(file_counts.keys()), 1):
                lines.append(f"{i}. {fname}")
            answer = "\n".join(lines)

        return RagResult(
            answer=             answer,
            citations=          [],
            query_type=         ctx.query_type,
            used_query=         ctx.search_query,
            reformulated_query= ctx.reformulated,
            understood_query=   ctx.understood,
            retrieved_count=    total_chunks,
            passed_threshold=   total_chunks,
            top_distance=       None,
            top_rrf_score=      None,
            debug_chunk_ids=    [],
            fallback=           False,
            fallback_reason=    None,
        )
    except Exception as e:
        logger.error("meta query 처리 실패: %s", e)
        return _make_fallback(ctx, "retrieval_error", _MSG_RETRIEVAL_ERROR)


def handle_existence_query(
    ctx: QueryContext,
    req_id: str,
    filters: dict | None = None,
) -> RagResult:
    """
    "에너지 관련 특허 있어?" 같은 존재 확인 질문을 처리한다.

    retrieve() + quality_gate()를 재사용해 BM25+벡터 하이브리드로 관련 문서를 탐색한다.
    quality_gate 통과 청크의 source_file을 추출해 파일 목록을 반환한다.
    LLM을 호출하지 않으므로 빠르다.
    """
    try:
        chunks, union_count = retrieve(ctx.search_query, req_id, filters=filters)
    except Exception as e:
        logger.error("[%s] existence query 검색 실패: %s", req_id, e)
        return _make_fallback(ctx, "retrieval_error", _MSG_RETRIEVAL_ERROR)

    passed = quality_gate(chunks, req_id)
    top_rrf = chunks[0].get("rrf_score") if chunks else None
    top_distance = chunks[0].get("distance") if chunks else None

    if not passed:
        return RagResult(
            answer=             "관련 문서가 등록되어 있지 않습니다.",
            citations=          [],
            query_type=         ctx.query_type,
            used_query=         ctx.search_query,
            reformulated_query= ctx.reformulated,
            understood_query=   ctx.understood,
            retrieved_count=    union_count,
            passed_threshold=   0,
            top_distance=       top_distance,
            top_rrf_score=      top_rrf,
            debug_chunk_ids=    extract_debug_chunk_ids(chunks),
            fallback=           False,
            fallback_reason=    "no_docs",
        )

    # 통과 청크에서 파일명 추출 (순서 보존 dedup)
    files = list(dict.fromkeys(
        c["source_file"] for c in passed if c.get("source_file")
    ))
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
        retrieved_count=    union_count,
        passed_threshold=   len(passed),
        top_distance=       passed[0].get("distance"),
        top_rrf_score=      top_rrf,
        debug_chunk_ids=    extract_debug_chunk_ids(passed),
        fallback=           False,
        fallback_reason=    None,
    )


# =============================================================================
# Content RAG 파이프라인
# =============================================================================
def _eval_filter(metadata: dict, where: dict) -> bool:
    """
    ChromaDB where절을 재귀적으로 평가한다.

    지원 연산자: $and, $or, $eq, $ne, $in, $nin
    단순 키-값 {"key": "value"}도 $eq와 동일하게 처리한다.

    예시:
        {"doc_type": "patent"}
        {"$and": [{"doc_type": "patent"}, {"year": "2024"}]}
        {"year": {"$in": ["2023", "2024"]}}
    """
    for key, val in where.items():
        if key == "$and":
            if not all(_eval_filter(metadata, clause) for clause in val):
                return False
        elif key == "$or":
            if not any(_eval_filter(metadata, clause) for clause in val):
                return False
        else:
            # key는 필드명, val은 직접 값이거나 {"$op": value} 형태
            meta_val = metadata.get(key)
            if isinstance(val, dict):
                op, operand = next(iter(val.items()))
                if op == "$eq"  and meta_val != operand:   return False
                if op == "$ne"  and meta_val == operand:   return False
                if op == "$in"  and meta_val not in operand: return False
                if op == "$nin" and meta_val in operand:   return False
            else:
                if meta_val != val:
                    return False
    return True


def _apply_bm25_filters(hits: list[dict], filters: dict | None) -> list[dict]:
    """
    BM25 결과에 ChromaDB where절 필터를 적용한다.

    _eval_filter()로 중첩 필터($and/$or/$eq 등)까지 완전하게 평가한다.
    Phase 2 메타데이터 필터링에서 벡터 검색 결과와 BM25 결과가 다른 범위를
    커버하는 문제를 방지한다.
    """
    if not filters or not hits:
        return hits
    return [h for h in hits if _eval_filter(h.get("metadata") or {}, filters)]


def _rrf_merge(
    bm25_hits: list[dict],
    vector_hits: list[dict],
    top_n: int,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion으로 BM25와 벡터 검색 결과를 합산한다.

    각 청크의 RRF 점수 = 1/(k + bm25_rank) + 1/(k + vector_rank)
      - 양쪽에 모두 등장한 청크: 두 점수 합산 → 자동 부스트
      - 한 쪽만 등장한 청크:    해당 점수만 반영
      - k=60: 표준값. 낮추면 상위 랭크 boost 강해짐

    chunk_id를 key로 점수를 누적하므로 dedup이 자연스럽게 처리된다.
    벡터 chunk를 base로 우선 사용하는 이유: distance 필드 보존.

    # TODO: GPU 서버 구축 후 이 함수 대신 reranker 사용:
    #   reranked = reranker.rerank(query, union, top_n=top_n)
    """
    scores: dict[str, float] = {}
    vector_map: dict[str, dict] = {c["chunk_id"]: c for c in vector_hits}
    bm25_map:   dict[str, dict] = {c["chunk_id"]: c for c in bm25_hits}

    for rank, chunk in enumerate(bm25_hits):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    for rank, chunk in enumerate(vector_hits):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    results = []
    for cid in sorted_ids[:top_n]:
        # 벡터 chunk 우선(distance 필드 있음), 없으면 BM25 chunk
        base = vector_map.get(cid) or bm25_map[cid]
        c = dict(base)
        c["rrf_score"] = round(scores[cid], 6)
        results.append(c)

    return results


def retrieve(
    query: str,
    req_id: str,
    filters: dict | None = None,
) -> tuple[list[dict], int]:
    """
    하이브리드 검색: BM25 + 벡터 검색 → RRF → top_n 반환.

    흐름:
    1. BM25 검색 (top_k=10) + 메타데이터 필터 적용
       → 키워드 exact match에 강함 (특허번호, 날짜, 고유명사)

    2. 벡터 검색 (top_k=10) + ChromaDB where 필터 적용
       → 의미 유사성에 강함 (paraphrase, 설명형 질문)

    3. RRF (Reciprocal Rank Fusion)
       → BM25 순위 + 벡터 순위를 수식으로 합산
       → chunk_id 기준 dedup 포함
       → 양쪽 모두 등장한 청크 자동 부스트
       → 최종 rrf_top_n(5)개 반환

    Returns:
        (merged_chunks, union_size)
        merged_chunks : rrf_score 필드가 추가된 상위 top_n 청크 리스트
        union_size    : RRF 입력 전 BM25+벡터 unique 후보 수 (retrieved_count 용)
    """
    cfg = RAG_CONFIG

    # ── Step 1: BM25 검색 + 필터 ───────────────────────────────────────────
    bm25_raw  = bm25_retriever.search(query, top_k=cfg["bm25_top_k"])
    bm25_hits = _apply_bm25_filters(bm25_raw, filters)
    logger.info("[%s] BM25 검색: %d개 (필터 후 %d개)", req_id, len(bm25_raw), len(bm25_hits))

    # ── Step 2: 벡터 검색 ──────────────────────────────────────────────────
    raw = query_collection(
        query_text=query,
        collection_name=cfg["collection_name"],
        persist_dir=cfg["persist_dir"],
        embedding_provider=cfg["embedding_provider"],
        n_results=cfg["vector_top_k"],
        where=filters,
    )
    ids       = raw.get("ids",       [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    vector_hits = []
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        _m   = meta or {}
        page = _m.get("page_number")
        vector_hits.append({
            "chunk_id":    chunk_id,
            "text":        documents[i] if i < len(documents) else "",
            "metadata":    meta,
            "distance":    distances[i] if i < len(distances) else None,
            "header":      _m.get("header") or (f"p.{page}" if page is not None else ""),
            "source_file": _m.get("source_file") or _m.get("file_name", ""),
        })
    logger.info("[%s] 벡터 검색: %d개", req_id, len(vector_hits))

    # ── Step 3: RRF ────────────────────────────────────────────────────────
    # union_size = RRF 입력 전 unique 후보 수. retrieved_count에 기록한다.
    union_size = len({c["chunk_id"] for c in bm25_hits} | {c["chunk_id"] for c in vector_hits})
    merged = _rrf_merge(bm25_hits, vector_hits, top_n=cfg["rrf_top_n"], k=cfg["rrf_k"])
    logger.info(
        "[%s] RRF 완료: union=%d → top%d | 최고점=%.6f",
        req_id, union_size, len(merged),
        merged[0]["rrf_score"] if merged else 0.0,
    )
    return merged, union_size


def quality_gate(
    chunks: list[dict],
    req_id: str,
) -> list[dict]:
    """
    벡터 distance 기반 품질 게이트. 확실히 무관한 결과만 걸러낸다.

    RRF 결과 중 distance 필드가 있는 청크의 최솟값(best_distance)을 확인한다.
    best_distance > distance_threshold 이면 관련 문서 없음으로 판단 → fallback.

    distance가 없는 청크(BM25 전용)만 있는 경우 통과시킨다.
    BM25 키워드 매칭 자체가 관련성의 신호이기 때문이다.

    # TODO: reranker 활성화 시 이 함수를 rerank_score 기반으로 교체:
    #   top_score = chunks[0].get("rerank_score") or 0.0
    #   if top_score < RAG_CONFIG["min_rerank_score"]: return []

    Returns:
        통과한 청크 리스트 (비어 있으면 fallback으로 연결)
    """
    if not chunks:
        return []

    threshold = RAG_CONFIG["distance_threshold"]
    distances = [c["distance"] for c in chunks if c.get("distance") is not None]

    if distances:
        best_distance = min(distances)
        if best_distance > threshold:
            logger.info(
                "[%s] 품질 게이트 미통과: best_distance=%.4f > %.4f",
                req_id, best_distance, threshold,
            )
            return []

    if distances:
        logger.info(
            "[%s] 품질 게이트 통과: best_distance=%.4f top_rrf=%.6f",
            req_id, min(distances), chunks[0].get("rrf_score") or 0.0,
        )
    else:
        # 벡터 거리 없음 = BM25 전용 청크만 남은 상태. 키워드 매칭이 관련성 신호이므로 통과.
        # distance 기반 품질 검증이 불가하므로 WARNING으로 기록.
        logger.warning(
            "[%s] 품질 게이트 통과(BM25 only): distance 없음 → 키워드 매칭 신뢰. top_rrf=%.6f",
            req_id, chunks[0].get("rrf_score") or 0.0,
        )
    return chunks


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

    재시도 정책:
      - 503 등 일시적 서버 오류: 지수 백오프(1s → 2s)로 최대 max_retries회 재시도
      - 429 / rate limit / quota 초과: 재시도 없이 즉시 예외를 던진다
        (재시도해도 쿼터가 회복되지 않으므로 호출자가 빠르게 실패를 인지해야 함)
    """
    client = get_client()
    last_exc: Exception | None = None

    import time as _time

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
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s
                logger.warning("[%s] LLM 호출 실패 (%d/%d), %d초 후 재시도: %s", req_id, attempt + 1, max_retries, wait, e)
                _time.sleep(wait)
            else:
                raise

    raise last_exc  # type: ignore[misc]


def format_citations(chunks: list[dict]) -> list[Citation]:
    """
    사용자 노출용 Citation 리스트를 반환한다.

    (header, source_file) 기준으로 중복 제거해 깔끔한 출처 목록을 만든다.
    같은 섹션의 여러 청크가 답변에 기여했더라도 사용자에게는 하나의 출처로 표시된다.

    청크 단위 원본 추적이 필요하면 debug_chunk_ids(RagResult 필드)를 사용한다.
    """
    seen: set[tuple] = set()
    result: list[Citation] = []
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


def extract_debug_chunk_ids(chunks: list[dict]) -> list[str]:
    """
    디버깅용 chunk_id 전체 목록을 반환한다.

    format_citations()는 (header, source_file) 기준 dedup을 하므로
    같은 섹션의 여러 청크가 합쳐져 어느 청크가 실제로 답변에 기여했는지
    추적이 어렵다. 이 함수는 dedup 없이 전체 chunk_id를 보존한다.
    """
    return [c["chunk_id"] for c in chunks]


def _handle_chitchat(
    ctx: QueryContext,
    req_id: str,
    chat_history: list | None,
) -> RagResult:
    """
    query_type == "greeting" 일 때 호출된다.
    인사·잡담에 LLM이 자유롭게 답한다.

    no-docs 경로(품질 게이트 미통과)에서는 호출하지 않는다.
    관련 문서가 없으면 LLM 호출 없이 _MSG_NO_DOCS를 즉시 반환한다.
    """
    history = trim_history(chat_history) if chat_history else None
    history_block = ""
    if history:
        lines = [
            f"{'사용자' if m.get('role') == 'user' else '어시스턴트'}: {m.get('content', '')}"
            for m in history
        ]
        history_block = "\n[이전 대화]\n" + "\n".join(lines) + "\n"

    prompt = (
        f"{_CHITCHAT_SYSTEM_PROMPT}"
        f"{history_block}\n"
        f"[질문]\n{ctx.original_query}"
    )

    try:
        answer = generate_answer(prompt, req_id)
    except Exception as e:
        logger.error("[%s] chit-chat LLM 호출 실패: %s", req_id, e)
        return _make_fallback(ctx, "llm_error", _MSG_LLM_ERROR)

    logger.info("[%s] chit-chat 응답 완료: %d자", req_id, len(answer))
    return RagResult(
        answer=             answer,
        citations=          [],
        query_type=         ctx.query_type,
        used_query=         ctx.search_query,
        reformulated_query= ctx.reformulated,
        understood_query=   ctx.understood,
        retrieved_count=    0,
        passed_threshold=   0,
        top_distance=       None,
        top_rrf_score=      None,
        debug_chunk_ids=    [],
        fallback=           False,
        fallback_reason=    None,
    )


def _handle_content(
    ctx: QueryContext,
    req_id: str,
    filters: dict | None,
    chat_history: list | None = None,
) -> RagResult:
    """일반 RAG 파이프라인. retrieve → quality_gate → build → generate → cite."""
    # 검색
    try:
        chunks, union_count = retrieve(ctx.search_query, req_id, filters=filters)
    except Exception as e:
        logger.error("[%s] 검색 실패: %s", req_id, e)
        return _make_fallback(ctx, "retrieval_error", _MSG_RETRIEVAL_ERROR)

    # 품질 게이트: 관련 문서 없으면 LLM 호출 없이 즉시 반환
    passed = quality_gate(chunks, req_id)
    top_rrf = chunks[0].get("rrf_score") if chunks else None
    if not passed:
        logger.info("[%s] 품질 게이트 미통과 → no_docs 응답 반환", req_id)
        return RagResult(
            answer=             _MSG_NO_DOCS,
            citations=          [],
            query_type=         ctx.query_type,
            used_query=         ctx.search_query,
            reformulated_query= ctx.reformulated,
            understood_query=   ctx.understood,
            retrieved_count=    union_count,
            passed_threshold=   0,
            top_distance=       chunks[0].get("distance") if chunks else None,
            top_rrf_score=      top_rrf,
            debug_chunk_ids=    extract_debug_chunk_ids(chunks),
            fallback=           False,
            fallback_reason=    "no_docs",
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
            retrieved_count=union_count,
            passed_threshold=len(passed),
            top_distance=passed[0].get("distance"),
            top_rrf_score=top_rrf,
        )

    result = RagResult(
        answer=             answer,
        citations=          format_citations(passed),
        query_type=         ctx.query_type,
        used_query=         ctx.search_query,
        reformulated_query= ctx.reformulated,
        understood_query=   ctx.understood,
        retrieved_count=    union_count,
        passed_threshold=   len(passed),
        top_distance=       passed[0].get("distance"),
        top_rrf_score=      top_rrf,
        debug_chunk_ids=    extract_debug_chunk_ids(passed),
        fallback=           False,
        fallback_reason=    None,
    )
    logger.info(
        "[%s] 완료 | type=%s retrieved=%d passed=%d top_dist=%s top_rrf=%.6f chunk_ids=%s",
        req_id, ctx.query_type, result.retrieved_count,
        result.passed_threshold,
        f"{result.top_distance:.4f}" if result.top_distance is not None else "N/A",
        result.top_rrf_score or 0.0,
        result.debug_chunk_ids,
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
           greeting  → _handle_chitchat() (LLM 자유 답변)
           meta      → handle_meta_query() (DB 현황 조회)
           existence → handle_existence_query() (문서 존재 확인)
           content   → _handle_content()
                         └ 품질 게이트 통과 → LLM 답변 + 출처
                         └ 품질 게이트 미통과 → _MSG_NO_DOCS 즉시 반환 (LLM 호출 없음)
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

    if ctx.query_type == "greeting":
        return _handle_chitchat(ctx, req_id, chat_history)
    if ctx.query_type == "meta":
        return handle_meta_query(ctx)
    if ctx.query_type == "existence":
        return handle_existence_query(ctx, req_id, filters)
    return _handle_content(ctx, req_id, filters, chat_history)
