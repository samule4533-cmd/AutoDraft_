# src/bm25_retriever.py
#
# 역할: BM25 키워드 검색 인덱스를 관리한다.
#
# ── 저장 구조 ─────────────────────────────────────────────────────────────────
# BM25 인덱스는 디스크가 아닌 메모리에만 존재한다.(휘발성)
# 서버 시작 시 ChromaDB에서 전체 청크를 읽어 인메모리 인덱스를 빌드한다.
# 새 문서 ingest 후에는 rebuild_index()로 갱신한다.
#
# ── BM25 동작 원리 ────────────────────────────────────────────────────────────
# BM25Okapi(corpus)로 인덱스 생성.
# corpus = [토큰 리스트, 토큰 리스트, ...] (청크 하나 = 토큰 리스트 하나)
# 검색 시 query도 같은 방식으로 토크나이즈 → 각 청크와의 BM25 점수 계산.
#
# BM25 점수 = 각 query 토큰에 대한 (TF × IDF) 합산
#   TF  : 해당 토큰이 그 청크에 얼마나 자주 나오는가 (많을수록 높음, 포화 보정 있음)
#   IDF : 해당 토큰이 전체 문서에서 얼마나 희귀한가 (희귀할수록 높음)
# → 흔한 단어(조사, 접속사)는 IDF가 낮아 점수에 별 영향 없음
# → 특허번호처럼 드문 단어는 IDF가 높아 점수에 강하게 반영됨
#
# ── 동시성 설계 ───────────────────────────────────────────────────────────────
# _bm25와 _chunks는 전역 변수다.
# build/rebuild(쓰기)와 search(읽기)가 동시에 실행되면 두 가지 문제가 생긴다:
#
# 문제 1 — _bm25와 _chunks 불일치
#   _bm25 = new_bm25 과 _chunks = new_chunks 사이에 search()가 끼어들면
#   새 인덱스 + 구버전 청크 조합 → scores[i]와 _chunks[i]가 다른 문서를 가리킴.
#
# 문제 2 — 빌드 도중 읽기
#   _chunks = [] 초기화 직후 search()가 실행되면 항상 빈 결과 반환.
#
# 해결:
#   - build/rebuild: 로컬에서 완전히 빌드 → _index_lock 안에서 한 번에 swap
#   - search: _index_lock 안에서 레퍼런스 스냅샷만 복사 → 즉시 lock 해제
#             이후 읽기는 스냅샷으로 진행 (성능 유지)

import logging
import re
import threading
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_bm25: BM25Okapi | None = None
_chunks: list[dict[str, Any]] = []
_index_lock = threading.Lock()

# ── Kiwi 형태소 분석기 싱글톤 ────────────────────────────────────────────────
# Kiwi 초기화는 무겁기 때문에 최초 호출 시 한 번만 생성한다.
_kiwi = None
_kiwi_lock = threading.Lock()

# BM25에 보존할 품사 태그
# 명사류: NNG(일반명사) NNP(고유명사) NNB(의존명사) NR(수사) NP(대명사)
# 용언류: VV(동사) — VA(형용사) 제외: 특허 문서에서 형용사 어간("크", "있")은 노이즈
# 기타: XR(어근) SL(외국어) SH(한자) SN(숫자) W_SERIAL(일련번호/날짜) MM(관형사) MAG(일반부사)
_KEEP_POS = {
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV",
    "XR", "SL", "SH", "SN", "W_SERIAL", "MM", "MAG",
}

# 특허/기업 문서에서 모든 청크에 빈번히 등장하는 불용어
# IDF가 낮아 점수 기여 없이 노이즈만 됨
_DOMAIN_STOPWORDS = {
    # VV — 특허 관용 동사구 어간
    "관하",   # "…에 관한 것이다"
    "따르",   # "…에 따르면"
    "위하",   # "…을 위하여"
    "통하",   # "…을 통하여"
    "의하",   # "…에 의하여"
    "이루",   # "…로 이루어진"
    "포함하", # "…을 포함하는"
    "구성하", # "…으로 구성된"
    "수행하", # "…를 수행하는"
    "제공하", # "…를 제공하는"
    # NNB — 특허 문서 의존명사 노이즈
    "것",    # "…에 관한 것이다"
    "수",    # "…할 수 있다"
    "년",    # "2024년" — SN "2024"로 충분
    "호",    # "제10-2708831호" — W_SERIAL로 충분
    "일",    # "출원일" 등에서 분리되는 의존명사
    "등",    # "등" (etc.)
    # NNG/MM — 특허 지시어 (모든 문서에 등장, 검색 신호 없음)
    "상기",  # "상기 발명은" — 한국 특허 관용어
    "해당",  # "해당 청구항"
    "본",    # "본 발명" (MM 관형사)
}


# 특허 도메인 복합어 사용자 사전
# kiwi 기본 모델이 분리하는 복합어를 단일 명사로 등록한다.
# (word, tag, score) — score가 낮을수록 우선순위 높음
_PATENT_USER_WORDS: list[tuple[str, str, float]] = [
    ("청구항", "NNG", -3.0),   # 청구 + 항 → 단일 토큰
    ("출원인", "NNG", -3.0),   # 출원 + 인
    ("발명자", "NNG", -3.0),   # 발명 + 자
    ("특허청", "NNG", -3.0),   # 특허 + 청
    ("등록번호", "NNG", -3.0), # 등록 + 번호
    ("출원번호", "NNG", -3.0), # 출원 + 번호
    ("출원일",   "NNG", -3.0), # 출원 + 일
    ("등록일",   "NNG", -3.0), # 등록 + 일
    ("기준면",   "NNG", -3.0), # 기준 + 면
    ("풋프린트", "NNG", -3.0), # foot + print (복합어)
]


def _get_kiwi():
    """Kiwi 싱글톤을 반환한다. 스레드 세이프."""
    global _kiwi
    if _kiwi is None:
        with _kiwi_lock:
            if _kiwi is None:
                try:
                    from kiwipiepy import Kiwi
                    kiwi = Kiwi()
                    for word, tag, score in _PATENT_USER_WORDS:
                        kiwi.add_user_word(word, tag, score)
                    _kiwi = kiwi
                    logger.info("Kiwi 형태소 분석기 초기화 완료 (사용자 사전 %d개)", len(_PATENT_USER_WORDS))
                except Exception as e:
                    logger.warning("Kiwi 초기화 실패, 폴백 토크나이저 사용: %s", e)
                    _kiwi = False  # 실패 표시 (None과 구분)
    return _kiwi if _kiwi is not False else None


def _fallback_tokenize(text: str) -> list[str]:
    """Kiwi 불가 시 공백 분리 폴백."""
    text = text.lower()
    text = re.sub(r"[^\w\s\-\.]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def _tokenize(text: str) -> list[str]:
    """
    BM25용 토크나이저. kiwipiepy로 한국어 형태소를 분리한다.

    처리 순서:
    1. Kiwi 형태소 분석 → 보존 품사(명사/동사어간/수사/외국어/W_SERIAL 등)만 추출
       - 조사(JK*, JX, JC), 어미(E*), 접속부사(MAJ) 제거
       - "회사는"/"회사가"/"회사를" → 모두 "회사"로 통일
       - 특허번호·날짜는 W_SERIAL 태그로 그대로 보존
    2. 도메인 불용어 제거 (특허 관용 동사구·지시어 등)
    3. 소문자 변환 (영문/외국어 토큰)
    4. ASCII 단독 문자 제거 (한국어 1음절 단어 "딥" 등은 보존)

    Kiwi 불가 시 _fallback_tokenize()로 동작한다.
    """
    kiwi = _get_kiwi()
    if kiwi is None:
        return _fallback_tokenize(text)

    try:
        morphs = kiwi.tokenize(text, normalize_coda=True)
    except Exception as e:
        logger.debug("Kiwi 분석 오류, 폴백: %s", e)
        return _fallback_tokenize(text)

    tokens: list[str] = []

    # 3단계: 보존 품사만 추출
    for token in morphs:
        pos = token.tag.name if hasattr(token.tag, "name") else str(token.tag)
        pos_base = pos.split("+")[0]  # 복합 태그 처리 (드문 경우)
        if pos_base not in _KEEP_POS:
            continue
        form = token.form.lower()
        if form in _DOMAIN_STOPWORDS:
            continue
        # ASCII 단독 문자만 길이 필터 (한국어 1음절 단어 "딥", "글" 등은 보존)
        if form.isascii() and len(form) <= 1 and not form.isdigit():
            continue
        tokens.append(form)

    return tokens


def debug_tokenize(text: str) -> None:
    """
    토크나이저 동작을 육안으로 확인하기 위한 디버그 함수.
    kiwi 원시 형태소 결과와 최종 BM25 토큰을 나란히 출력한다.
    """
    print("=" * 80)
    print("원문:")
    print(text)

    kiwi = _get_kiwi()
    if kiwi is None:
        print("\n[Kiwi 없음] fallback 토큰:")
        print(_fallback_tokenize(text))
        return

    print("\n[Kiwi 원시 형태소 결과]")
    raw_tokens = kiwi.tokenize(text, normalize_coda=True)
    for tok in raw_tokens:
        print(f"  {tok.form:<20} | {tok.tag}")

    print("\n[최종 BM25 토큰]")
    print(_tokenize(text))


def build_index(collection) -> None:
    """
    ChromaDB 컬렉션에서 전체 청크를 로드하고 BM25 인덱스를 빌드한다.

    로컬 변수로 완전히 빌드한 뒤 _index_lock 안에서 한 번에 swap한다.
    → 빌드 도중 search()는 이전 인덱스로 안전하게 동작.
    """
    global _bm25, _chunks

    result = collection.get(include=["documents", "metadatas"])
    ids   = result.get("ids",       [])
    docs  = result.get("documents", [])
    metas = result.get("metadatas", [])

    if not ids:
        logger.warning("BM25 인덱스 빌드 실패: ChromaDB에 청크 없음")
        with _index_lock:
            _bm25   = None
            _chunks = []
        return

    new_chunks: list[dict[str, Any]] = []
    tokenized_corpus: list[list[str]] = []

    for i, chunk_id in enumerate(ids):
        text = docs[i]  if i < len(docs)  else ""
        meta = metas[i] if i < len(metas) else {}

        new_chunks.append({
            "chunk_id":    chunk_id,
            "text":        text,
            "metadata":    meta,
            "header":      meta.get("header",      ""),
            "source_file": meta.get("source_file", ""),
            "distance":    None,
        })
        tokenized_corpus.append(_tokenize(text))

    new_bm25 = BM25Okapi(tokenized_corpus)

    with _index_lock:
        _bm25   = new_bm25
        _chunks = new_chunks

    logger.info("BM25 인덱스 빌드 완료: %d개 청크", len(new_chunks))


def search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    BM25로 쿼리와 가장 관련 있는 청크를 검색한다.

    _index_lock으로 스냅샷을 획득한 뒤 즉시 해제.
    이후 읽기는 스냅샷으로 진행해 rebuild와 충돌하지 않는다.
    """
    with _index_lock:
        bm25   = _bm25
        chunks = _chunks

    if bm25 is None or not chunks:
        logger.warning("BM25 인덱스 없음 → 빈 결과 반환")
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            break
        chunk = dict(chunks[idx])
        chunk["bm25_score"] = float(scores[idx])
        results.append(chunk)

    logger.debug("BM25 검색 완료: query='%s...' → %d개", query[:20], len(results))
    return results


def fetch_by_claim_numbers(numbers: list[int]) -> list[dict]:
    """
    청구항 번호에 해당하는 청크를 header 기반으로 직접 반환한다.

    BM25 점수 계산을 거치지 않으므로 Kiwi IDF 희석 문제를 완전히 우회한다.
    "청구항 N"이 header에 포함된 청크를 모두 반환한다.

    반환 청크의 bm25_score는 float("inf")로 설정한다.
    → _rrf_merge에서 rank 0 취급되어 최상위 RRF 점수를 받는다.
    """
    if not numbers:
        return []

    with _index_lock:
        chunks = _chunks

    target_patterns = {f"청구항 {n}" for n in numbers}
    results: list[dict] = []
    for chunk in chunks:
        header = chunk.get("header") or ""
        if any(pat in header for pat in target_patterns):
            c = dict(chunk)
            c["bm25_score"] = float("inf")
            results.append(c)

    logger.debug(
        "fetch_by_claim_numbers: %s → %d개 청크",
        list(numbers), len(results),
    )
    return results


def rebuild_index(collection) -> None:
    """
    새 문서 ingest 후 BM25 인덱스를 갱신한다.

    호출 시점: api.py POST/PATCH/DELETE /ingest 완료 후
    """
    logger.info("BM25 인덱스 재빌드 시작")
    build_index(collection)
