# rag_chain.py
2026-03-19

Phase 2 완성 단계. 질문 전처리는 `query_processor.py`로 완전히 분리되었으며, 질문 유형(meta/existence/content)에 따른 라우팅 처리가 추가된 RAG 오케스트레이션 핵심 모듈.

## 개요
`rag_chain.py`는 `query_processor.py`가 생성한 `QueryContext`를 받아, 질문 유형에 따라 처리 경로를 분기하고 최종 답변을 생성하는 RAG 체인의 핵심 오케스트레이션 모듈이다.

현재 구현은 **Phase 2** 수준의 RAG 체인에 해당하며, 단발성 질의응답뿐만 아니라 대화 맥락을 반영한 연속 질문도 처리할 수 있다.

이 파일의 핵심 목적은 다음과 같다.

- 질문 유형(meta/existence/content)에 따라 처리 경로 분기
- ChromaDB 직접 조회 또는 벡터 검색 수행
- 검색 결과 중 관련성 낮은 청크 제거
- 관련 문서가 없을 경우 fallback 처리
- LLM에 전달할 문맥 구성 및 chat_history 반영
- 문서 기반 답변 생성
- 출처와 검색 통계 포함 결과 반환

즉, 이 모듈은 **검색 계층(vector_db)** 과 **생성 계층(LLM)** 사이를 연결하는 실행 레이어이며, 질문 전처리는 `query_processor.py`가 담당한다.

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
- 질문 전처리 전체를 `query_processor.py`로 분리 (`QueryContext` 수신)
- `handle_meta_query()` — ChromaDB 직접 조회, LLM 없이 파일 목록 반환
- `handle_existence_query()` — 벡터 검색으로 자료 존재 여부 판단
- `_handle_content()` — 기존 RAG 파이프라인, chat_history 실제 전달 수정
- `RagResult`에 `query_type`, `understood_query` 필드 추가
- rate limit 발생 시 60초 대기 제거 → 즉시 llm_error fallback 반환
- `chat_history`가 `build_prompt()`에 실제로 전달되지 않던 버그 수정

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
- `query_type` — "meta" / "existence" / "content"
- `used_query` — 실제 검색에 사용된 query
- `reformulated_query` — 지시어 재작성 발생 시 값 존재
- `understood_query` — 검색 최적화 변환 발생 시 값 존재
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

### `handle_meta_query(ctx: QueryContext) -> RagResult`
"파일 몇 개야?", "어떤 문서들 있어?" 같은 DB 현황 질문을 처리한다.

#### 동작 방식
- ChromaDB 메타데이터를 직접 조회
- LLM 호출 없이 파일 목록 반환
- 빠르고 정확하며 API 비용 없음

---

### `handle_existence_query(ctx: QueryContext, req_id: str) -> RagResult`
"에너지 관련 자료 있어?" 같은 존재 확인 질문을 처리한다.

#### 동작 방식
- 벡터 검색으로 관련 청크 조회
- threshold 통과 여부로 존재/부재 판단
- LLM 호출 없음

---

### `_handle_content(ctx, req_id, filters, chat_history) -> RagResult`
일반 문서 내용 질문을 처리하는 RAG 파이프라인이다.

#### 처리 단계
1. 벡터 검색
2. threshold 필터링
3. context 조합
4. prompt 생성 (chat_history 포함)
5. Gemini 호출
6. citation 정리

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
- rate limit(429) 발생 시 즉시 raise → llm_error fallback 반환 (대기 없음)

---

### `format_citations(chunks) -> list[Citation]`
통과 청크를 바탕으로 중복 제거된 출처 리스트를 생성하는 함수이다.

#### 중복 제거 기준
- `(header, source_file)` 조합

---

### `ask(query, chat_history=None, filters=None) -> RagResult`
이 파일의 핵심 메인 진입점이다.

#### 처리 단계
1. `process_query()` 호출 → `QueryContext` 생성
2. `query_type`에 따라 분기:
   - `meta` → `handle_meta_query()`
   - `existence` → `handle_existence_query()`
   - `content` → `_handle_content()`

#### fallback 분기
- `retrieval_error`: ChromaDB 검색 실패
- `no_docs`: threshold 통과 청크 없음
- `llm_error`: Gemini 호출 실패

---

## 의존성

### 내부 의존성
- `src.llm_api.get_client`
- `src.llm_api.GeminiAPIError`
- `src.query_processor.process_query`, `QueryContext`, `trim_history`
- `src.vector_db.get_or_create_collection`, `query_collection`

### 외부 라이브러리
- `pydantic` (BaseModel)
- `google.genai.types`
- `logging`, `os`, `re`, `uuid`, `pathlib`

---

## 설계 포인트

### 1. 질문 유형 라우팅
meta/existence/content 세 경로로 분기하여, 간단한 현황 질문은 LLM 호출 없이 빠르게 처리하고 비용을 절감한다.

### 2. 전처리 레이어 분리
질문 이해/재작성/라우팅은 `query_processor.py`가 담당하고, 이 파일은 실행만 담당한다. 책임이 명확히 나뉘어 디버깅과 테스트가 용이하다.

### 3. threshold 기반 안전장치
검색 결과를 그대로 사용하지 않고 최소한의 관련성 기준(0.55)을 적용한다.

### 4. fallback 정책 명확화
문서 없음 / 검색 오류 / 생성 오류를 구분하여 운영 대응이 가능하게 했다.

### 5. 검색 통계 포함 반환
`retrieved_count`, `passed_threshold`, `top_distance`를 결과에 포함하여 문제 원인을 빠르게 진단할 수 있도록 했다.

### 6. rate limit 즉시 반환
Gemini rate limit 발생 시 60초 대기 없이 즉시 llm_error를 반환해 사용자 응답 지연을 방지한다.

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
최종 수정: 2026-03-19
관련 파일: 'src/rag_chain.py'
---
