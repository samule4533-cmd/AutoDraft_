# rag_chain.py

RAG 오케스트레이션 핵심 모듈. 하이브리드 검색(BM25+벡터) → RRF 병합 → quality gate → 컨텍스트 확장 → LLM 답변 생성까지 전 과정을 조율한다.

## 개요

`rag_chain.py`는 `query_processor.py`가 생성한 `QueryContext`를 받아, 질문 유형에 따라 처리 경로를 분기하고 최종 답변을 생성하는 RAG 체인의 핵심 오케스트레이션 모듈이다.

- **검색 계층** (`bm25_retriever` + `vector_db`) 과 **생성 계층** (Gemini LLM) 사이를 연결
- 질문 유형(greeting/meta/existence/content) 기반 4-way 라우팅
- 청구항 번호 직접 검색, 누락 청구항 처리, 환각 방지

---

## 설정값: `RAG_CONFIG`

| 키 | 기본값 | 의미 |
|----|--------|------|
| `collection_name` | `"ninewatt_company"` | ChromaDB 컬렉션 |
| `persist_dir` | `data/vector_store/chroma` | ChromaDB 저장 경로 |
| `embedding_provider` | `"local"` | `"local"` \| `"openai"` |
| `gemini_model` | `"gemini-2.0-flash"` | LLM 모델 |
| `bm25_top_k` | `10` | BM25 검색 결과 수 |
| `vector_top_k` | `10` | 벡터 검색 결과 수 |
| `rrf_top_n` | `10` | RRF 병합 후 유지 수 |
| `rrf_k` | `60` | RRF 감쇠 상수 |
| `distance_threshold` | `0.65` | quality gate 코사인 거리 상한 |
| `max_context_chars` | `12000` | LLM 컨텍스트 최대 길이 |

---

## 반환 타입

### `Citation`
```python
class Citation(BaseModel):
    chunk_id: str
    header: str
    source_file: str
    distance: float | None
```

### `RagResult`

| 필드 | 타입 | 설명 |
|------|------|------|
| `answer` | `str` | 최종 답변 텍스트 |
| `citations` | `list[Citation]` | 출처 목록 (no-docs 시 빈 리스트) |
| `query_type` | `str` | `"greeting"` \| `"meta"` \| `"existence"` \| `"content"` |
| `used_query` | `str` | 실제 검색에 사용된 쿼리 |
| `reformulated_query` | `str \| None` | 지시어 재작성 결과 (디버그) |
| `understood_query` | `str \| None` | 이해 변환 결과 (디버그) |
| `retrieved_count` | `int` | BM25+벡터 합산 검색 수 |
| `passed_threshold` | `int` | quality gate 통과 수 |
| `top_distance` | `float \| None` | 최상위 코사인 거리 |
| `top_rrf_score` | `float \| None` | 최상위 RRF 점수 |
| `debug_chunk_ids` | `list[str]` | 통과한 전체 청크 ID |
| `fallback` | `bool` | True = 오류/no-docs 응답 |
| `fallback_reason` | `str \| None` | `"no_docs"` \| `"retrieval_error"` \| `"llm_error"` |

---

## 처리 경로 (4-way 라우팅)

```
ask()
  ├── query_type == "greeting"   → _handle_chitchat()  (문서 검색 없음, 잡담 응답)
  ├── query_type == "meta"       → handle_meta_query()  (ChromaDB 직접, LLM 없음)
  ├── query_type == "existence"  → handle_existence_query()  (벡터 검색, LLM 없음)
  └── query_type == "content"   → _handle_content()   (하이브리드 검색 + LLM)
```

---

## 주요 함수

### `handle_meta_query(ctx) -> RagResult`
"파일 몇 개야?", "어떤 문서들 있어?" 처리. ChromaDB `source_file` 메타데이터를 직접 조회해 LLM 없이 파일 목록을 반환한다.

---

### `handle_existence_query(ctx, req_id, filters) -> RagResult`
"RAG 관련 자료 있나?" 처리. 벡터+BM25 검색 후 quality gate 통과 여부로 존재/부재를 판단한다. LLM 호출 없음.

---

### `retrieve(query, req_id, filters, claim_numbers) -> tuple[list[dict], int]`
하이브리드 검색 파이프라인.

```
1. BM25 검색 (top_k=10) + where 필터 적용
2. 청구항 번호 직접 주입 (claim_numbers → fetch_by_claim_numbers)
   └─ bm25_score=inf 설정 → RRF에서 최상위 순위 보장
3. 벡터 검색 (top_k=10) + where 필터 적용
4. RRF 병합 → top_n 반환
```

`claim_numbers`가 있으면 BM25 점수 계산을 우회하고 헤더 패턴 매칭으로 청구항 청크를 직접 삽입한다.

---

### `quality_gate(chunks, req_id) -> list[dict]`
`distance_threshold(0.65)` 기준으로 관련 없는 청크를 제거한다.

- BM25 전용 청크 (`distance` 필드 없음): 항상 통과
- 최상위 거리가 threshold 초과: 전체 청크 탈락 (fallback 트리거)

---

### `_expand_context(passed, req_id) -> list[dict]`
child 청크를 parent 도입부 + 인접 형제 청크로 확장한다.

- section 청크: 그대로 반환
- child 청크: `parent_store`에서 parent intro + 전후 child 조회 → 합산
- parent_store 비어있으면 원본 child 그대로 반환 (graceful degradation)

---

### `build_context_block(chunks, req_id, max_chars) -> str`
LLM 입력용 문맥 문자열 생성. `[문서 N]\n출처: {header} | {source_file}\n{text}` 형식. `max_chars` 초과 시 하위 순위 청크 제외.

---

### `generate_answer(prompt, req_id, max_retries) -> str`
Gemini 호출. temperature=0.0. rate limit(429) 발생 시 재시도 없이 즉시 raise → `llm_error` fallback.

---

### `format_citations(chunks) -> list[Citation]`
`(header, source_file)` 기준 중복 제거. 같은 섹션의 여러 청크는 단일 citation으로 합산.

---

### `_handle_content(ctx, req_id, filters, chat_history) -> RagResult`
전체 RAG 파이프라인.

```
retrieve()
  ↓
quality_gate()  → 통과 없음: fallback("no_docs")
  ↓
청구항 번호 검증
  ├── 모두 없음: fallback("no_docs", "청구항 N을(를) 문서에서 찾을 수 없습니다.")
  └── 일부 없음: missing_note 주입 ("청구항 N은(는) 문서에서 찾을 수 없습니다. 명시하라.")
  ↓
_expand_context()
  ↓
build_context_block() + build_prompt()
  ↓
generate_answer()
  ↓
no-docs 신호 감지 (answer에 "찾을 수 없", "없습니다" 포함 시)
  ├── citations 비움
  └── answer의 "📎 출처" 이후 텍스트 제거
  ↓
RagResult 반환
```

---

### `_handle_chitchat(ctx, req_id, chat_history) -> RagResult`
greeting 타입 처리. 문서 검색 없이 `_CHITCHAT_SYSTEM_PROMPT`로 Gemini 호출. 회사 사실 지어내기 방지 프롬프트 포함.

---

### `ask(query, chat_history, filters) -> RagResult`
메인 진입점. `process_query()` → `QueryContext` 생성 → 4-way 라우팅.

---

## 설계 포인트

### 1. 하이브리드 검색 + RRF
BM25는 특허번호·날짜 등 키워드에, 벡터는 의미 유사도에 강하다. RRF(`score = 1/(60+rank)` 합산)로 두 결과를 병합해 단일 검색 방식의 단점을 보완한다.

### 2. 청구항 번호 직접 검색
Kiwi 형태소 분석 후 "청구항"이 전 청크에 등장해 IDF가 낮아지는 문제 → `fetch_by_claim_numbers()`로 헤더 패턴 직접 매칭, BM25 점수 계산을 우회한다.

### 3. 누락 청구항 처리
청구항 234가 특허 원문에서 삭제된 경우 → 파싱 결과에도 없음. 모두 없으면 즉시 "찾을 수 없습니다" 반환, 일부 없으면 LLM에 missing_note를 주입해 답변 시 명시하도록 한다.

### 4. No-docs 출처 억제
LLM 응답에 "없습니다" 계열 문구가 있으면 citations를 비우고 answer에서 "📎 출처" 이후를 잘라낸다. 없는 정보에 출처가 붙는 혼란을 방지한다.

### 5. Quality gate (0.65)
검색 결과를 그대로 LLM에 넘기지 않고 최소 관련성을 검증한다. 비용 절감 + 환각 방지. threshold는 한국 특허 문서 + MiniLM 임베딩 실측 기반 결정값.

### 6. Rate limit 즉시 반환
429 발생 시 60초 대기 없이 llm_error 반환 → 사용자 응답 지연 방지.

---

## 의존성

### 내부
- `query_processor.process_query`, `QueryContext`
- `bm25_retriever.search`, `bm25_retriever.fetch_by_claim_numbers`
- `vector_db.query_collection`
- `parent_store.get_adjacent_child_ids`, `parent_store.get_child_text`
- `llm_api.get_client`, `llm_api.GeminiAPIError`

### 외부 라이브러리
- `pydantic`, `google.genai`, `logging`, `os`, `re`, `uuid`, `pathlib`

---
최종 수정: 2026-03-27
관련 파일: `src/rag_chain.py`
---
