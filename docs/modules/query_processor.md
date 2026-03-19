# query_processor.py
2026-03-19

Phase 2에서 신규 추가된 쿼리 전처리 모듈. `rag_chain.py`에서 분리된 질문 이해/재작성/라우팅 전담 레이어.

## 개요
`query_processor.py`는 사용자 질문을 RAG 실행 전에 전처리하는 레이어이다.
원본 질문을 받아 검색에 최적화된 형태로 변환하고, 어떤 처리 경로를 거칠지 결정한 뒤 `QueryContext`를 반환한다.

`rag_chain.py`는 이 결과만 받아 실행에만 집중한다.

이 파일의 핵심 목적은 다음과 같다.

- 오래된 대화 기록 정리
- 지시어 포함 질문을 맥락 기반으로 재작성 (조건부)
- 구어체/짧은 질문을 벡터 검색에 최적화된 형태로 변환 (조건부)
- 질문 유형(meta/existence/content) 분류

즉, 이 모듈은 **사용자 자연어 질문과 RAG 실행 사이의 전처리 계층**이다.

---

## 역할
이 파일은 `rag_chain.ask()`가 호출되기 전에 질문을 정리하고 처리 방향을 결정한다.

구체적으로 다음 책임을 가진다.

- chat_history를 최근 N턴으로 슬라이싱
- 지시어 감지 시 LLM으로 질문 재작성 (Reformulation)
- 구어체/짧은 질문을 키워드 중심 문장으로 변환 (Understanding)
- 질문 유형을 패턴 기반으로 분류 (Routing)
- 결과를 `QueryContext` Pydantic 모델로 반환

---

## 이 파일이 필요한 이유
`rag_chain.py`가 전처리와 실행을 모두 담당하면 파일이 비대해지고 각 단계의 테스트/디버깅이 어렵다.

전처리를 분리하면 다음 이점이 생긴다.

- reformulation이 잘못됐는지, 검색이 잘못됐는지 원인을 분리해서 볼 수 있다
- 각 단계를 독립적으로 수정할 수 있다
- `rag_chain.py`는 실행에만 집중해 코드가 단순해진다

---

## 주요 구성 요소

### `PROCESSOR_CONFIG`
전처리 관련 설정을 한 곳에 모아둔 딕셔너리이다.

#### 포함 항목
- `gemini_model`: reformulation에 사용할 모델
- `understanding_model`: understanding에 사용할 모델
- `history_max_turns`: chat_history 보존 최대 턴 수 (기본 5턴)

---

### `QueryContext`
전처리 결과를 담는 Pydantic BaseModel이다.

#### 필드
- `original_query` — 사용자 원본 질문 (프롬프트 생성에 사용)
- `search_query` — 실제 벡터 검색에 사용할 쿼리
- `reformulated` — reformulation 발생 시 재작성 결과, 없으면 None
- `understood` — understanding 발생 시 변환 결과, 없으면 None
- `query_type` — "meta" / "existence" / "content"

---

### 패턴 상수

#### `_REFERENTIAL_PATTERN`
지시어 감지용 정규식이다.

포함 패턴: `그럼`, `거기`, `그건`, `그것`, `해당`, `방금`, `그거`, `그 `

이 패턴이 감지되고 chat_history가 있을 때만 Reformulation을 실행한다.

#### `_COLLOQUIAL_PATTERN`
구어체 어미 감지용 정규식이다.

포함 패턴: `알려줘`, `뭐야`, `있어`, `있나`, `해줘` 등

이 패턴이 감지되거나 질문이 15자 미만이면 Understanding을 실행한다.

#### `_META_PATTERN`
DB 전체 현황 질문 감지용 정규식이다.

"파일 몇 개야?", "파일 갯수 알려줘", "어떤 문서들 있어?" 등을 meta로 분류한다.

#### `_EXISTENCE_PATTERN`
특정 주제 자료 존재 여부 질문 감지용 정규식이다.

"에너지 관련 특허 있어?", "RAG 자료 있나요?" 등을 existence로 분류한다.

---

## 주요 함수

### `trim_history(history, max_turns) -> list`
chat_history를 최근 max_turns 턴으로 슬라이싱한다.

#### 동작 방식
- 1턴 = user + assistant 메시지 쌍 = 2개
- `history[-(max_turns * 2):]`

#### 이유
사내 문서 QA 특성상 오래된 맥락의 가치가 낮고, 히스토리가 쌓이면 컨텍스트 창 초과 및 응답 지연이 발생한다.

---

### `should_reformulate(query, history) -> bool`
지시어가 포함되어 있고 이전 대화가 있을 때만 True를 반환한다.

#### 두 조건 동시 요구 이유
- history가 없으면 재작성에 쓸 맥락이 없어 LLM 호출이 무의미하다
- 지시어가 없으면 독립 질문이므로 원본 그대로 검색해도 품질 차이가 없다

---

### `reformulate_query(query, history, req_id) -> str`
이전 대화 맥락을 참고해 지시어가 포함된 질문을 독립적인 형태로 재작성한다.

#### Fallback 처리
실패 시 원본 query 반환. 이 단계의 실패가 전체 흐름을 막아선 안 된다.

---

### `should_understand(query) -> bool`
구어체 어미가 감지되거나 15자 미만 짧은 질문일 때 True를 반환한다.

#### 15자 기준 이유
"에너지 사용량은?" (9자), "출원일 알려줘" (8자) 같은 짧은 질문은 벡터 검색 정밀도가 낮아 변환이 필요하다.

---

### `understand_query(query, req_id) -> str`
구어체 질문을 벡터 검색에 최적화된 키워드 중심 문장으로 변환한다.

#### Fallback 처리
실패 시 원본 query 반환.

---

### `classify_query(query) -> QueryType`
질문 유형을 패턴 기반으로 분류한다. LLM 호출 없이 처리한다.

#### 분류 결과
- `meta`: DB 전체 현황 질문 ("파일 몇 개야?")
- `existence`: 특정 주제 자료 존재 확인 ("에너지 관련 특허 있어?")
- `content`: 문서 내용 기반 질의응답 (그 외 모든 질문)

#### 패턴 기반 선택 이유
라우팅은 단순 분류라 LLM이 필요없다. LLM 호출 시 매 질문마다 추가 지연/비용이 발생한다.

---

### `process_query(query, chat_history, req_id) -> QueryContext`
이 파일의 메인 진입점이다.

#### 처리 순서
1. history trim
2. reformulation (조건부)
3. understanding (조건부)
4. routing

#### 반환 필드 의미
- `original_query`: 프롬프트 생성 시 사용 (자연어 원본 유지)
- `search_query`: ChromaDB 벡터 검색 시 사용 (최적화된 쿼리)
- `reformulated`: 재작성 발생 여부 (디버깅)
- `understood`: 검색 최적화 변환 발생 여부 (디버깅)
- `query_type`: 라우팅 결정 기준

---

## 처리 흐름

```
원본 질문
    ↓
history trim (항상)
    ↓
지시어 감지? + history 있음?
  Yes → reformulation → base_query 갱신
  No  → 원본 그대로
    ↓
구어체/짧은 질문?
  Yes → understanding → search_query 갱신
  No  → base_query 그대로
    ↓
패턴 매칭 → meta / existence / content 분류
    ↓
QueryContext 반환
```

---

## 의존성

### 내부 의존성
- `src.llm_api.get_client`

### 외부 라이브러리
- `pydantic` (BaseModel)
- `google.genai.types`
- `logging`, `os`, `re`

---

## 설계 포인트

### 1. 조건부 실행으로 비용 최소화
모든 질문에 reformulation과 understanding을 실행하면 매 턴 Gemini 호출이 최대 2회 추가된다. 필요한 경우에만 실행해 비용과 지연을 줄인다.

### 2. Fallback 필수
reformulation과 understanding은 보조 수단이므로, 실패 시 원본 query로 자동 복구되어 전체 흐름이 중단되지 않는다.

### 3. 라우팅은 패턴 기반
meta/existence 분류에 LLM이 필요 없으므로 정규식으로 처리한다. 속도와 비용 모두 절감된다.

### 4. original_query와 search_query 분리
LLM 프롬프트에는 자연스러운 원본 질문을, 벡터 검색에는 최적화된 쿼리를 각각 사용한다.

---

## 예외 및 주의사항

### 1. routing은 원본 query 기준
reformulation/understanding 이후가 아니라 원본 질문으로 routing한다. 변환 과정에서 질문 의도가 바뀌는 것을 방지하기 위함이다.

### 2. 패턴 미매칭은 content로 fallback
meta/existence 패턴에 해당하지 않는 질문은 모두 content로 분류된다.

---

## 사용 방법

`rag_chain.ask()` 내부에서 자동으로 호출된다. 직접 호출 예시:

```python
from src.query_processor import process_query

ctx = process_query(
    query="그럼 거기 출원일은?",
    chat_history=[...],
    req_id="test-001",
)
print(ctx.query_type)    # "content"
print(ctx.search_query)  # 재작성된 쿼리
```

---
최종 수정: 2026-03-19
관련 파일: 'src/query_processor.py'
---
