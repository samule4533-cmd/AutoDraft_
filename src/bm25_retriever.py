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


def _tokenize(text: str) -> list[str]:
    """
    BM25용 토크나이저. 한국어/영어 혼합 텍스트를 처리한다.

    처리 순서:
    1. 소문자 변환 → 대소문자 불일치 방지
    2. 특수문자 제거, 단 하이픈(-), 점(.)은 보존
       이유: 특허번호(10-2708831), 날짜(2024.09.24) 구조 유지
    3. 공백 기준 분리
    4. 길이 1 이하 토큰 제거 (조사, 단독 기호 등 노이즈 제거)
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\-\.]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


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


def rebuild_index(collection) -> None:
    """
    새 문서 ingest 후 BM25 인덱스를 갱신한다.

    호출 시점: api.py POST/PATCH/DELETE /ingest 완료 후
    """
    logger.info("BM25 인덱스 재빌드 시작")
    build_index(collection)
