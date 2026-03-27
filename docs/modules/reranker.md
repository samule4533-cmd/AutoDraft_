# reranker.py

Cross-encoder 기반 재순위화 모듈. Jina Reranker v2 (다국어)를 사용해 BM25+벡터 하이브리드 검색 결과를 재정렬한다.

> **현재 상태**: RAG 파이프라인에서 주석 처리되어 있음. GPU 서버 준비 완료 시 `api.py`의 `reranker.load_model()` 주석 해제 후 활성화.

---

## 개요

하이브리드 검색(BM25+벡터+RRF)으로 top-N 청크를 가져온 후, cross-encoder가 `(query, chunk)` 쌍을 직접 평가해 최종 순위를 재조정한다. bi-encoder(벡터 검색)보다 정확하지만 연산 비용이 높아 검색 후 단계에 배치한다.

---

## 모델 정보

| 항목 | 값 |
|------|-----|
| 모델 | `jinaai/jina-reranker-v2-base-multilingual` |
| 최대 토큰 | 8192 |
| 크기 | ~280MB |
| 라이센스 | Apache 2.0 |
| 지원 언어 | 100+ (한국어 포함) |

---

## 주요 함수

### `load_model() -> None`
thread-safe 지연 로딩. 디바이스 자동 감지:

| 디바이스 | dtype |
|---------|-------|
| CUDA | float16 |
| MPS (Apple Silicon) | float16 |
| CPU | float32 |

---

### `rerank(query, chunks, top_n=5) -> list[dict]`
cross-encoder 점수 계산 → top_n 반환.

- `_build_reranker_input()`: `"문서: {source_file}\n섹션: {header}\n\n{text}"` 형식으로 메타데이터 포함
- logit → sigmoid → `[0, 1]` 확률값 (`rerank_score` 필드 추가)
- 모델 미로드 시 원본 순서 유지 + `score=None` 반환 (safe fallback)

---

## 활성화 방법

```python
# api.py lifespan 내
# reranker.load_model()  # 주석 해제

# rag_chain._handle_content() 내
# passed = reranker.rerank(ctx.search_query, passed, top_n=5)  # 주석 해제
```

---

## 의존성

### 외부 라이브러리
- `sentence_transformers.CrossEncoder`
- `torch`, `threading`, `logging`

---
최종 수정: 2026-03-27
관련 파일: `src/reranker.py`
---
