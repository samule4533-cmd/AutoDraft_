# src/rag_chain.py
#
# Phase 1 — RAG 체인 코어
# 역할: 질문 → 검색 → 필터 → 생성 → 출처 포함 답변 반환
# 이 파일은 순수 오케스트레이션 레이어다.
# ChromaDB / Gemini를 직접 건드리지 않고 vector_db / llm_api 함수만 호출한다.

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from google.genai import types

from llm_api import GeminiAPIError, get_client
from vector_db import query_collection

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Config
# =============================================================================
# 모든 튜닝 포인트를 한 곳에 모아둔다.
# threshold: cosine distance 기준. 낮을수록 유사.
#   - 0.0 = 완전 동일 / 1.0 = 무관
#   - 0.5가 일반적 시작점. 검색 결과 top_distance를 보면서 조정.
# max_context_chars: LLM에 넘길 컨텍스트 총 글자 수 상한.
#   - 초과 시 하위 rank 청크부터 제외.
#   - 한국어 1자 ≈ 2~3 토큰 기준 6000자 ≈ 2000~3000 토큰.

RAG_CONFIG = {
    "collection_name":    os.getenv("CHROMA_COLLECTION_NAME", "ninewatt_company"),
    "persist_dir":        str(PROJECT_ROOT / "data" / "vector_store" / "chroma"),
    "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "local"),
    "gemini_model":       os.getenv("GEMINI_RAG_MODEL", "gemini-2.0-flash"),
    "n_results":          10,
    "distance_threshold": 0.5,
    "max_context_chars":  6000,
}

# fallback 메시지를 상수로 분리해두면 나중에 한 곳만 수정하면 된다.
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
"""


# =============================================================================
# Return types
# =============================================================================
# dict 대신 dataclass를 쓰는 이유:
#   - IDE 자동완성 + 오타를 컴파일 타임에 잡음
#   - 반환 구조가 코드 자체로 문서화됨
#   - 나중에 Pydantic BaseModel로 교체하면 JSON 직렬화까지 자동 (서버 연동 시)

@dataclass
class Citation:
    chunk_id:    str
    header:      str
    source_file: str
    distance:    float | None


@dataclass
class RagResult:
    answer:           str
    citations:        list[Citation]
    retrieved_count:  int          # ChromaDB가 반환한 청크 수
    passed_threshold: int          # distance threshold 통과 수
    top_distance:     float | None # 가장 유사한 청크의 distance (낮을수록 좋음)
    fallback:         bool         # True = 정상 답변 아님
    fallback_reason:  str | None   # "no_docs" | "retrieval_error" | "llm_error"


# =============================================================================
# Step 1 — retrieve
# =============================================================================
def retrieve(
    query: str,
    req_id: str,
    n_results: int = RAG_CONFIG["n_results"],
    filters: dict | None = None,
) -> list[dict]:
    """
    ChromaDB에서 유사 청크를 검색하고 정규화된 list[dict]로 반환한다.

    ChromaDB raw 결과는 중첩 리스트 구조라 그대로 쓰기 불편하다.
    이 함수가 평탄화 책임을 전담한다.
    """
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


# =============================================================================
# Step 2 — filter
# =============================================================================
def filter_by_threshold(
    chunks: list[dict],
    req_id: str,
    threshold: float = RAG_CONFIG["distance_threshold"],
) -> list[dict]:
    """
    distance > threshold인 청크를 제거한다.

    ChromaDB는 관련 없어도 무조건 n개를 반환한다.
    이 필터가 없으면 엉뚱한 문서가 LLM 컨텍스트로 들어간다.
    """
    passed = [
        c for c in chunks
        if c["distance"] is not None and c["distance"] <= threshold
    ]
    logger.info(
        "[%s] threshold=%.2f 적용 → %d/%d 청크 통과",
        req_id, threshold, len(passed), len(chunks),
    )
    return passed


# =============================================================================
# Step 3 — context 조합
# =============================================================================
def build_context_block(
    chunks: list[dict],
    req_id: str,
    max_chars: int = RAG_CONFIG["max_context_chars"],
) -> str:
    """
    통과한 청크들을 LLM에 넘길 컨텍스트 문자열로 조합한다.

    각 청크 앞에 헤더/출처를 붙이는 이유:
      LLM이 어느 문서에서 가져온 내용인지 인식해야 [출처: ...] 인용이 정확해진다.

    max_chars 초과 시 하위 rank 청크부터 제외하는 이유:
      상위 rank(가장 유사한 것)를 우선 보존하기 위해.
    """
    blocks = []
    total  = 0

    for i, chunk in enumerate(chunks):
        # 헤더에서 Markdown 기호(#)를 제거한다.
        # chunk['header']에는 "## 출원일자" 같이 # 기호가 그대로 담겨 있다.
        # LLM이 이걸 그대로 인용하면 "[출처: ## 출원일자]"처럼 # 이 노출된다.
        # LLM에 넘기는 context에서는 사람이 읽기 좋은 텍스트만 전달한다.
        clean_header = re.sub(r"^#{1,6}\s*", "", chunk["header"]).strip()
        block = (
            f"[문서 {i + 1}]\n"
            f"출처: {clean_header} | {chunk['source_file']}\n"
            f"{chunk['text']}\n"
        )
        if total + len(block) > max_chars:
            logger.info(
                "[%s] max_chars(%d) 도달 → %d/%d 청크만 컨텍스트에 포함",
                req_id, max_chars, i, len(chunks),
            )
            break
        blocks.append(block)
        total += len(block)

    return "\n---\n".join(blocks)


# =============================================================================
# Step 4 — 프롬프트 조합
# =============================================================================
def build_prompt(
    query: str,
    context: str,
    chat_history: list | None = None,  # Phase 2에서 활성화
) -> str:
    """
    시스템 지시 + (이전 대화) + 참고 문서 + 질문을 하나의 프롬프트로 조합한다.

    chat_history 형식: [{"role": "user"|"assistant", "content": str}, ...]
    Phase 1에서는 None으로 전달 → history_block 생략.
    """
    history_block = ""
    if chat_history:
        lines = []
        for m in chat_history:
            role = "사용자" if m.get("role") == "user" else "어시스턴트"
            lines.append(f"{role}: {m.get('content', '')}")
        history_block = "\n[이전 대화]\n" + "\n".join(lines) + "\n"

    return (
        f"{SYSTEM_PROMPT}"
        f"{history_block}\n"
        f"[참고 문서]\n{context}\n\n"
        f"[질문]\n{query}"
    )


# =============================================================================
# Step 5 — LLM 호출
# =============================================================================
def generate_answer(prompt: str, req_id: str, max_retries: int = 3) -> str:
    """
    Gemini에 프롬프트를 전달하고 텍스트 답변을 반환한다.
    llm_api.get_client() 싱글턴을 재사용한다.

    temperature=0.0: 사실 기반 문서 답변이므로 창의성을 최소화.
    429(rate limit) 에러는 지수 백오프로 최대 max_retries회 재시도한다.
      - RPM(분당) 제한: Gemini는 1분 윈도우 기반이므로 첫 대기를 60초로 시작.
      - RPD(일일) 제한: 재시도해도 해소되지 않으며 당일 할당량 초과를 의미.
        이 경우 로그에 "RESOURCE_EXHAUSTED" 또는 "daily" 키워드가 포함됨.
    """
    client = get_client()
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=RAG_CONFIG["gemini_model"],
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            answer = (getattr(resp, "text", "") or "").strip()
            # Gemini가 빈 문자열을 반환하는 경우가 있다.
            # (안전 필터 차단, 모델 오작동 등)
            # 빈 채로 반환하면 챗봇 화면에 아무것도 안 보이므로 오류로 처리한다.
            # ask()의 except 블록이 잡아서 _MSG_LLM_ERROR를 반환한다.
            if not answer:
                raise GeminiAPIError("LLM이 빈 응답을 반환했습니다.")
            logger.info("[%s] LLM 응답 수신: %d자", req_id, len(answer))
            return answer
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_rate_limit = (
                "429" in str(e)
                or "quota" in err_str
                or "rate" in err_str
                or "resource_exhausted" in err_str
            )
            if is_rate_limit:
                # RPD(일일 할당량) 초과는 재시도가 무의미하므로 즉시 중단
                if "daily" in err_str or ("resource_exhausted" in err_str and "per_day" in err_str):
                    logger.error(
                        "[%s] Gemini 일일 할당량 초과 (RPD). 재시도 불가: %s",
                        req_id, e,
                    )
                    raise
                # RPM(분당) 제한: 지수 백오프 60s → 120s → 180s
                wait = 60 * (attempt + 1)
                logger.warning(
                    "[%s] Gemini 429 rate limit, %d초 후 재시도 (%d/%d) — %s",
                    req_id, wait, attempt + 1, max_retries, e,
                )
                time.sleep(wait)
                continue
            raise

    raise last_exc  # type: ignore[misc]


# =============================================================================
# Step 6 — citation 포맷
# =============================================================================
def format_citations(chunks: list[dict]) -> list[Citation]:
    """
    (header, source_file) 기준 중복 제거 후 Citation 리스트를 반환한다.

    chunk_id는 모든 청크가 고유하므로 chunk_id로는 중복이 절대 걸리지 않는다.
    같은 섹션에서 복수의 청크가 threshold를 통과했을 때
    사용자에게 같은 출처가 여러 번 표시되는 걸 막기 위해
    (header, source_file) 조합을 dedup 키로 사용한다.

    header에서 Markdown 기호(#)를 제거한다.
    build_context_block에서도 동일하게 제거하므로
    LLM 인라인 인용과 Citation 목록의 헤더 표기가 일치한다.
    """
    seen   = set()
    result = []
    for c in chunks:
        # 같은 섹션(헤더+파일)이면 청크가 여러 개여도 출처는 하나로 묶는다
        key = (c["header"], c["source_file"])
        if key not in seen:
            seen.add(key)
            result.append(Citation(
                chunk_id=    c["chunk_id"],
                # # 기호 제거: "## 출원일자" → "출원일자"
                # LLM에 전달한 context와 표기 방식을 통일한다
                header=      re.sub(r"^#{1,6}\s*", "", c["header"]).strip(),
                source_file= c["source_file"],
                distance=    c["distance"],
            ))
    return result


# =============================================================================
# 메인 진입점
# =============================================================================
def ask(
    query: str,
    chat_history: list | None = None,  # Phase 2
    filters: dict | None = None,       # Phase 3 (metadata pre-filtering)
) -> RagResult:
    """
    사용자 질문을 받아 RagResult를 반환한다.

    반환값에 retrieved_count / passed_threshold / top_distance를 포함하는 이유:
      답변이 이상할 때 검색 문제인지(retrieved/passed 수치) 생성 문제인지를
      숫자만 보고 즉시 판별할 수 있게 하기 위해.

    fallback_reason을 분리하는 이유:
      "문서 없음"과 "시스템 오류"를 구분해야 원인별 대응이 가능하기 때문.
    """
    req_id = uuid.uuid4().hex[:8]
    logger.info("[%s] 질문 수신: %.80s", req_id, query)

    # ── 1. 검색 ──────────────────────────────────────────────────────────────
    try:
        chunks = retrieve(query, req_id, filters=filters)
    except Exception as e:
        logger.error("[%s] ChromaDB 검색 실패: %s", req_id, e)
        return RagResult(
            answer=           _MSG_RETRIEVAL_ERROR,
            citations=        [],
            retrieved_count=  0,
            passed_threshold= 0,
            top_distance=     None,
            fallback=         True,
            fallback_reason=  "retrieval_error",
        )

    # ── 2. 필터 ──────────────────────────────────────────────────────────────
    passed = filter_by_threshold(chunks, req_id)

    # ── 3. Fallback 분기 (관련 문서 없음) ────────────────────────────────────
    if not passed:
        logger.info("[%s] 통과 청크 0개 → LLM 호출 생략, fallback 반환", req_id)
        return RagResult(
            answer=           _MSG_NO_DOCS,
            citations=        [],
            retrieved_count=  len(chunks),
            passed_threshold= 0,
            top_distance=     chunks[0]["distance"] if chunks else None,
            fallback=         True,
            fallback_reason=  "no_docs",
        )

    # ── 4. 컨텍스트 조합 ──────────────────────────────────────────────────────
    context = build_context_block(passed, req_id)

    # ── 5. 프롬프트 조합 ──────────────────────────────────────────────────────
    prompt = build_prompt(query, context, chat_history)

    # ── 6. LLM 호출 ───────────────────────────────────────────────────────────
    try:
        answer = generate_answer(prompt, req_id)
    except (GeminiAPIError, Exception) as e:
        logger.error("[%s] LLM 호출 실패: %s", req_id, e)
        return RagResult(
            answer=           _MSG_LLM_ERROR,
            citations=        [],
            retrieved_count=  len(chunks),
            passed_threshold= len(passed),
            top_distance=     passed[0]["distance"],
            fallback=         True,
            fallback_reason=  "llm_error",
        )

    # ── 7. 결과 조합 ──────────────────────────────────────────────────────────
    result = RagResult(
        answer=           answer,
        citations=        format_citations(passed),
        retrieved_count=  len(chunks),
        passed_threshold= len(passed),
        top_distance=     passed[0]["distance"],
        fallback=         False,
        fallback_reason=  None,
    )
    logger.info(
        "[%s] 완료 | retrieved=%d passed=%d top_dist=%.4f",
        req_id,
        result.retrieved_count,
        result.passed_threshold,
        result.top_distance or 0.0,
    )
    return result


# =============================================================================
# CLI — 단독 실행 테스트
# =============================================================================
def _print_result(result: RagResult) -> None:
    print("\n" + "=" * 60)
    if result.fallback:
        print(f"[FALLBACK: {result.fallback_reason}]")
    print(f"\n[답변]\n{result.answer}")
    print(f"\n[검색 통계]")
    print(f"  검색된 청크: {result.retrieved_count}개")
    print(f"  threshold 통과: {result.passed_threshold}개")
    print(f"  최근접 distance: {result.top_distance:.4f}" if result.top_distance is not None else "  최근접 distance: -")
    if result.citations:
        print("\n[출처]")
        for c in result.citations:
            print(f"  - {c.header} | {c.source_file} (dist={c.distance:.4f})" if c.distance is not None else f"  - {c.header} | {c.source_file}")
    print("=" * 60)


_TEST_CASES = [
    # (케이스 설명, 질문)
    ("문서에 있는 질문",       "건물 에너지 모델링 자동화 시스템 및 이를 이용한 방법은 무엇이야??"),
    ("문서에 있는 질문",    "에너지 효율화 대상 건물을 선정하는 서버 및 이를 이용한 에너지 효율화 대상 건물 선정 방법은 무엇이야?"),
    ("문서에 없는 질문",            "건물 에너지 모델링 자동화 시스템 및 이를 이용한 방법 특허에 인사평가 제도가 나와있어?"),
    ("문서에 없는 질문",            "에너지 분석 플랫폼의 정보검색증강 기반 자연아 질의응답 서비스를 제공하는 방법 및 시스템 특허에 직원 복지 정책이 나와있어?"),
    ("수치/날짜가 포함된 질문",       "건물 에너지 모델링 자동화 시스템 및 이를 이용한 방법 특허의 출원일은 언제야?"),
    ("수치/날짜가 포함된 질문",       "인공지능 모델 기반 건물의 에너지 사용량 및 절약 방법 추론 솔루션 제공 방법 장치 및 시스템 특허의 등록번호를 알려줘"),
    ("모호한 질문(키워드)",    "에너지 사용량"),
    ("엉뚱한 질문",  "서울날씨 알려줘"),
]


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 배치 테스트: 케이스 전체 실행
        for label, q in _TEST_CASES:
            print(f"\n{'#' * 60}")
            print(f"[케이스] {label}")
            print(f"[질문]   {q}")
            result = ask(q)
            _print_result(result)
    else:
        query = sys.argv[1] if len(sys.argv) > 1 else "회사 주요 특허는 무엇인가요?"
        print(f"\n질문: {query}")
        result = ask(query)
        _print_result(result)
