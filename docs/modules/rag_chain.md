# rag_chain.py
2026-03-18;
Phase 1 코어 위에 Phase 2(대화형 UX, Query Reformulation, FastAPI 연동 대비)가 추가된 RAG 오케스트레이션 핵심 모듈.

## 개요
`rag_chain.py`는 사용자 질문을 받아, (Query Reformulation →) 벡터 검색 → threshold 필터링 → 컨텍스트 조합 → 프롬프트 생성 → LLM 호출 → 출처 정리까지 수행하는 RAG 체인의 핵심 오케스트레이션 모듈이다.

현재 구현은 **Phase 2** 수준의 RAG 체인에 해당하며, 단발성 질의응답뿐만 아니라 대화 맥락을 반영한 연속 질문도 처리할 수 있다.

이 파일의 핵심 목적은 다음과 같다.

- 이전 대화 맥락(chat_history)을 최근 N턴으로 관리
- 지시어가 포함된 질문을 LLM으로 재작성(Query Reformulation)해 검색 품질 향상
- 사용자 질문을 기준으로 관련 문서 청크 검색
- 검색 결과 중 관련성 낮은 청크 제거
- 관련 문서가 없을 경우 fallback 처리
- LLM에 전달할 문맥 구성
- 문서 기반 답변 생성
- 출처와 검색 통계 포함 결과 반환

즉, 이 모듈은 **검색 계층(vector_db)** 과 **생성 계층(LLM)** 사이를 연결하는 동시에, 대화 맥락 처리와 query 재작성까지 담당하는 중심 레이어이다.

---

## 역할
이 파일은 RAG 질의응답 흐름 전체를 조합하고 제어하는 역할을 한다.

구체적으로 다음 책임을 가진다.

- chat_history 수신 및 최근 N턴 슬라이싱
- 지시어 감지 및 조건부 Query Reformulation
- 사용자 질문 수신
- ChromaDB 검색 결과 정규화
- distance threshold 기반 필터링
- 문서 없음 fallback 분기
- LLM 입력용 context block 생성
- 시스템 프롬프트와 질문 결합
- Gemini 호출 및 재시도 처리
- citation 정리
- 최종 결과 구조화

---

## 현재 구현 범위
현재 이 파일은 **Phase 2** 수준의 RAG 체인 코어를 구현한 상태이다.

### Phase 1에서 포함된 요소
- 질문 입력
- 벡터 검색
- threshold 필터링
- fallback 분기
- context 조합
- prompt 생성
- LLM 호출
- 출처 정리
- 결과 구조화
- 테스트 케이스 기반 CLI 실행

### Phase 2에서 추가된 요소
- `chat_history` 수신 및 최근 N턴 슬라이싱 (`_trim_history`)
- 지시어 기반 조건부 Query Reformulation (`should_reformulate`, `reformulate_query`)
- `RagResult` → Pydantic BaseModel 전환 (FastAPI JSON 직렬화 자동화)
- `used_query`, `reformulated_query` 필드 추가 (디버깅용 재작성 추적)
- `history_max_turns` RAG_CONFIG 항목 추가

### 아직 본격적으로 확장되지 않은 요소
- metadata 사전 필터 자동 추출
- reranking
- answer post-processing 고도화
- 응답 품질 평가 자동화
- citation 세분화

---

## 이 파일이 필요한 이유
문서 검색과 LLM 호출을 각각 따로 구현해도, 실제 질의응답 시스템으로 동작시키려면 다음과 같은 중간 계층이 필요하다.

- 검색 결과를 그대로 LLM에 던져도 되는지 판단해야 함
- 관련 없는 결과를 걸러야 함
- 통과 청크가 없으면 LLM 호출을 생략해야 함
- 여러 청크를 하나의 문맥으로 조립해야 함
- 출처와 답변을 함께 정리해야 함
- 검색 문제와 생성 문제를 구분할 수 있어야 함
- 연속 질문에서 이전 맥락을 반영해야 함
- "그건", "거기" 같은 지시어를 독립적인 질문으로 재작성해야 함

`rag_chain.py`는 이 역할을 맡는 파일이다.

---

## 주요 구성 요소

### 설정값: `RAG_CONFIG`
RAG 파이프라인의 주요 튜닝 포인트를 한 곳에 모아둔 설정 딕셔너리이다.

#### 포함 항목
- `collection_name`
- `persist_dir`
- `embedding_provider`
- `gemini_model`
- `n_results`
- `distance_threshold`
- `max_context_chars`
- `history_max_turns` ← Phase 2 추가

#### 의미
- `n_results`: 벡터 검색에서 가져올 top-k 개수
- `distance_threshold`: 관련 문서로 인정할 최대 distance (현재 **0.55**)
- `max_context_chars`: LLM에 넘길 컨텍스트 최대 길이
- `history_max_turns`: chat_history 보존 최대 턴 수 (1턴 = user+assistant 쌍, 현재 5턴)

이 구조 덕분에 검색/생성 파라미터를 코드 여러 곳에서 수정하지 않고 한 곳에서 제어할 수 있다.

> **threshold 0.55 선택 근거:**
> 한국 특허 문서의 `【청구항 N】` 같은 특수 괄호 표기를 MiniLM 임베딩이 의미적으로 잘 처리하지 못해 distance가 0.50~0.55 구간에 분포한다.
> 완전 무관한 질문(서울날씨 등)은 0.65 이상으로 0.55에서도 차단된다.
> 0.50은 이 구간 질문을 전부 차단, 0.60은 노이즈 구간에 가까워 0.55가 적정값으로 결정됨.

---

### fallback 메시지 상수
- `_MSG_NO_DOCS`
- `_MSG_RETRIEVAL_ERROR`
- `_MSG_LLM_ERROR`

#### 목적
fallback 응답 메시지를 상수로 분리하여, 나중에 문구를 바꾸거나 정책을 수정할 때 한 곳만 수정하면 되도록 하기 위함이다.

---

### `SYSTEM_PROMPT`
문서 기반 질의응답을 위한 시스템 프롬프트이다.

#### 핵심 규칙
- 반드시 참고 문서에 근거하여 답변
- 문서에 없는 내용은 추측 금지
- 답변 내 사실마다 `[출처: 헤더명]` 표기
- 표/수치/날짜는 원문 그대로 인용
- 한국어 답변

---

### `_REFORMULATION_PROMPT`
Query Reformulation 전용 프롬프트로, SYSTEM_PROMPT와 분리되어 있다.

#### 분리 이유
역할이 다르기 때문이다. 이 프롬프트는 "질문을 고쳐라"이고 SYSTEM_PROMPT는 "문서를 보고 답하라"다. 섞으면 둘 다 품질이 떨어진다.

---

### `_REFERENTIAL_PATTERNS`
지시어 감지 정규식 패턴이다.

#### 포함 패턴
`그럼`, `거기`, `그건`, `그것`, `해당`, `방금`, `그거`, `그 `

#### 목적
이 패턴이 포함된 질문은 이전 맥락 없이 의미가 불분명할 가능성이 높다. 감지 시에만 Reformulation LLM 호출을 실행해 비용/지연을 줄인다.

---

## 반환 타입

### `Citation`
출처 정보를 표현하는 Pydantic BaseModel이다.

#### 필드
- `chunk_id`
- `header`
- `source_file`
- `distance`

---

### `RagResult`
최종 질의응답 결과를 표현하는 Pydantic BaseModel이다.

#### 필드
- `answer`
- `citations`
- `used_query` ← Phase 2 추가: 실제 검색에 사용된 query
- `reformulated_query` ← Phase 2 추가: 재작성이 발생한 경우만 값 존재
- `retrieved_count`
- `passed_threshold`
- `top_distance`
- `fallback`
- `fallback_reason`

#### Pydantic 전환 이유
FastAPI `response_model`로 선언 시 JSON 직렬화 자동화 + Swagger 자동 문서화.
중첩 모델(Citation)도 BaseModel이어야 Pydantic이 재귀 직렬화를 처리한다.

#### used_query vs reformulated_query 구분
- `used_query`: 실제 검색에 사용된 query (운영 관점)
- `reformulated_query`: 재작성이 일어났는지 여부 (디버깅 관점)
- 재작성이 없으면 `used_query = 원본`, `reformulated_query = None`

---

## 주요 함수

### `_trim_history(history, max_turns) -> list`
chat_history를 최근 max_turns 턴으로 슬라이싱한다.

#### 동작 방식
- 1턴 = user 메시지 + assistant 메시지 한 쌍 = 2개
- `history[-(max_turns * 2):]` 로 잘라낸다

#### 이유
사내 문서 QA 특성상 먼 과거 맥락의 가치가 낮고, 히스토리가 쌓이면 컨텍스트 창 초과 및 응답 지연이 발생한다. 최근 5턴 유지가 적정값으로 결정됨.

---

### `should_reformulate(query, chat_history) -> bool`
지시어가 포함되어 있고 이전 대화가 있을 때만 True를 반환한다.

#### 두 조건 동시 요구 이유
- chat_history가 없으면 재작성에 쓸 맥락이 없어 LLM 호출이 무의미하다
- 지시어가 없으면 독립 질문이므로 원본 그대로 검색해도 품질 차이가 없다
- 두 조건 중 하나라도 불충족이면 LLM 호출을 생략해 비용/지연을 아낀다

---

### `reformulate_query(query, chat_history, req_id) -> str`
이전 대화 맥락을 참고해 질문을 독립적인 형태로 재작성한다.

#### Fallback 처리
실패(LLM 오류, 빈 응답, 너무 짧은 결과) 시 원본 query를 그대로 반환한다. 재작성은 검색 품질을 높이는 보조 수단이므로 이 단계의 실패가 전체 답변 흐름을 차단해서는 안 된다.

---

### `retrieve(query, req_id, n_results=..., filters=None) -> list[dict]`
ChromaDB 검색 결과를 정규화된 청크 리스트로 변환하는 함수이다.

#### 역할
- `query_collection()` 호출
- Chroma raw 결과(`ids`, `documents`, `metadatas`, `distances`) 추출
- 각 결과를 평탄화하여 `list[dict]` 구조로 변환

---

### `filter_by_threshold(chunks, req_id, threshold=...) -> list[dict]`
검색 결과 중 distance 기준으로 관련 없는 청크를 제거하는 함수이다.

#### 동작 방식
- `distance`가 `threshold`(현재 0.55) 이하인 청크만 통과

---

### `build_context_block(chunks, req_id, max_chars=...) -> str`
통과 청크들을 LLM 입력용 문맥 문자열로 조합하는 함수이다.

#### 동작 방식
- 각 청크 앞에 `[문서 n]` 형식 부여
- 헤더와 source_file을 함께 포함
- 헤더의 Markdown 기호(`#`) 제거
- `max_context_chars`를 넘기면 하위 rank 청크부터 제외

---

### `build_prompt(query, context, chat_history=None) -> str`
시스템 프롬프트, (이전 대화), 참고 문서, 질문을 하나의 문자열로 조합하는 함수이다.

#### Phase 2에서의 변화
chat_history가 실질적으로 활성화됨. `_trim_history()`를 거친 history가 전달되어 이전 대화 맥락이 LLM에 포함된다.

---

### `generate_answer(prompt, req_id, max_retries=3) -> str`
Gemini를 호출하여 답변 텍스트를 생성하는 함수이다.

#### 특징
- temperature=0.0
- 빈 문자열 응답도 오류로 처리
- RPM 제한: 60초 단위 백오프 재시도 (최대 3회)
- RPD 제한: 즉시 중단

---

### `format_citations(chunks) -> list[Citation]`
통과 청크를 바탕으로 중복 제거된 출처 리스트를 생성하는 함수이다.

#### 중복 제거 기준
- `(header, source_file)` 조합

---

### `ask(query, chat_history=None, filters=None) -> RagResult`
이 파일의 핵심 메인 진입점이다.

#### 처리 단계
0. history 정리 (`_trim_history`)
1. Query Reformulation (조건부 — 지시어 감지 시만)
2. 검색 수행
3. threshold 필터링
4. 통과 청크 없으면 fallback 반환
5. context 조합
6. prompt 생성
7. LLM 호출
8. citation 정리
9. 결과 객체 반환

#### fallback 분기
- `retrieval_error`: ChromaDB 검색 실패
- `no_docs`: threshold 통과 청크 없음
- `llm_error`: Gemini 호출 실패

---

## 의존성

### 내부 의존성
- `llm_api.get_client`
- `llm_api.GeminiAPIError`
- `vector_db.query_collection`

### 외부 라이브러리
- `pydantic` (BaseModel — Phase 2 전환)
- `google.genai.types`
- `logging`, `os`, `re`, `time`, `uuid`, `pathlib`

---

## 설계 포인트

### 1. 오케스트레이션 레이어 분리
검색과 생성을 직접 구현하지 않고 조합만 담당하여 책임을 명확히 나눴다.

### 2. Reformulation 조건부 실행
모든 질문에 reformulation LLM을 호출하면 매 턴 Gemini 호출이 2회로 늘어난다. 지시어 감지 + chat_history 존재 시에만 실행해 비용/지연을 최소화한다.

### 3. threshold 기반 안전장치
검색 결과를 그대로 사용하지 않고 최소한의 관련성 기준(0.55)을 적용한다.

### 4. fallback 정책 명확화
문서 없음 / 검색 오류 / 생성 오류를 구분하여 운영 대응이 가능하게 했다.

### 5. 검색 통계 포함 반환
`retrieved_count`, `passed_threshold`, `top_distance`를 결과에 포함하여 문제 원인을 빠르게 진단할 수 있도록 했다.

### 6. Pydantic 전환
FastAPI 연동을 위해 `RagResult`와 `Citation`을 Pydantic BaseModel로 전환했다.

---

## 예외 및 주의사항

### 1. threshold 값 0.55는 실측 기반 결정
MiniLM 임베딩 모델이 한국 특허 문서의 `【】` 괄호 표기를 잘 처리하지 못해 발생한 문제를 분석하여 결정한 값이다. 문서 특성이나 임베딩 모델이 바뀌면 재조정이 필요하다.

### 2. context 길이 제한은 단순 cut-off 방식
현재는 `max_context_chars`를 넘기면 하위 rank 청크를 제외하는 방식이며, 보다 정교한 요약/압축 전략은 아직 없다.

### 3. reranking 없음
현재 검색 순서는 순수 벡터 유사도 결과에 의존한다.

### 4. Reformulation 실패 시 원본 query 사용
재작성 LLM 호출이 실패해도 원본 query로 자동 fallback되어 전체 흐름이 중단되지 않는다.

---

## CLI 및 테스트

### 단일 질문 실행
```bash
cd src && uv run python rag_chain.py "회사 주요 특허는 무엇인가요?"
```

### 배치 테스트 실행
```bash
cd src && uv run python rag_chain.py --test
```

---
최종 수정: 2026-03-18
관련 파일: 'src/rag_chain.py'
---
