# src/reranker.py
#
# 역할: jina cross-encoder로 후보 청크를 재정렬한다.
#
# ── jina-reranker-v2-base-multilingual 선택 이유 ─────────────────────────────
# - 최대 8192 토큰 → 우리 최대 청크 1495자를 잘림 없이 처리
# - 모델 크기 ~280MB → 빠른 로드, 낮은 메모리
# - 한국어 포함 100개 이상 언어 지원
# - Apache 2.0 라이선스
#
# ── 점수 해석 ─────────────────────────────────────────────────────────────────
# CrossEncoder.predict()는 raw logit을 반환한다.
# activation_fn=Sigmoid()를 지정하면 [0, 1] 범위로 변환된다.
# 0.5 이상 → 모델이 관련 있다고 판단
# 0.3 미만 → 거의 관련 없다고 판단
# 절대값보다 상대적 순서(ranking)가 더 중요하다.
#
# ── reranker 입력 포맷 ────────────────────────────────────────────────────────
# text만 넘기면 모델이 "어떤 문서의 어떤 섹션인지" 맥락 없이 판단한다.
# 문서명 + 섹션명(헤더)을 앞에 붙이면 relevance 판단 정확도가 올라간다.
# 포맷: "문서: {source_file}\n섹션: {header}\n\n{text}"
#
# ── 동시성 ────────────────────────────────────────────────────────────────────
# _model은 load_model() 1회만 쓰고 이후 read-only.
# _load_lock으로 중복 로드 방지. rerank()의 predict()는 읽기 전용이라 Lock 불필요.
#
# ── 모델 캐시 위치 ────────────────────────────────────────────────────────────
# 최초 실행 시 HuggingFace Hub에서 다운로드 (~280MB).
# 이후는 OS 캐시 디렉토리에서 로드:
#   Windows: C:\Users\{USER}\.cache\huggingface\hub\
#   Linux:   ~/.cache/huggingface/hub/

import logging
import re
import threading
from typing import Any

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"

_model: CrossEncoder | None = None
_load_lock = threading.Lock()


def load_model() -> None:
    """
    jina-reranker 모델을 로드한다.
    _load_lock으로 중복 호출을 방지한다.
    """
    global _model
    with _load_lock:
        if _model is not None:
            return
        logger.info("reranker 모델 로딩 시작: %s", MODEL_NAME)
        # CPU 환경에서는 float16이 지원되지 않아 "could not execute a primitive" 오류 발생.
        # GPU 여부를 직접 확인해 dtype을 명시적으로 결정한다.
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _model = CrossEncoder(
            MODEL_NAME,
            model_kwargs={"torch_dtype": dtype},
            trust_remote_code=True,
            activation_fn=torch.nn.Sigmoid(),
        )
        logger.info("reranker 모델 로딩 완료: %s", MODEL_NAME)


def _build_reranker_input(chunk: dict[str, Any]) -> str:
    """
    reranker 입력 텍스트를 조합한다.
    문서명 + 섹션명 + 본문을 함께 넘겨 relevance 판단 정확도를 높인다.

    예시:
        문서: 건물 에너지 모델링 자동화 시스템 및 이를 이용한 방법.pdf
        섹션: 청구항 1

        서버를 통해 건물의 건물 에너지 모델링 자동화 방법에 있어서 ...
    """
    source = (chunk.get("source_file") or "").strip()
    header = re.sub(r"^#{1,6}\s*", "", chunk.get("header") or "").strip()
    text   = (chunk.get("text") or "").strip()

    prefix_parts = []
    if source:
        prefix_parts.append(f"문서: {source}")
    if header:
        prefix_parts.append(f"섹션: {header}")

    if prefix_parts:
        return "\n".join(prefix_parts) + "\n\n" + text
    return text


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    cross-encoder로 후보 청크들을 재정렬하고 상위 top_n을 반환한다.

    모델 미로드 시: rerank_score=None을 추가한 복사본 top_n개 반환 (안전 fallback).
    """
    if _model is None:
        logger.warning("reranker 모델 미로드 → 원본 순서 유지")
        return [
            {**chunk, "rerank_score": None}
            for chunk in chunks[:top_n]
        ]

    if not chunks:
        return []

    pairs = [(query, _build_reranker_input(chunk)) for chunk in chunks]
    scores = _model.predict(pairs, convert_to_numpy=True)

    scored_chunks = sorted(
        zip(scores.tolist(), chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, chunk in scored_chunks[:top_n]:
        c = dict(chunk)
        c["rerank_score"] = round(float(score), 4)
        results.append(c)

    logger.debug(
        "rerank 완료: %d → top%d | 최고점=%.4f 최저점=%.4f",
        len(chunks), top_n,
        results[0]["rerank_score"] if results else 0.0,
        results[-1]["rerank_score"] if results else 0.0,
    )
    return results
