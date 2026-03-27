# query_processor.py

사용자 질문을 RAG 실행 전에 전처리하는 레이어. 히스토리 정리 → 지시어 재작성 → 구어체 변환 → 유형 분류 → 청구항 번호 추출까지 담당.

## 개요

`rag_chain.py`는 이 결과(`QueryContext`)만 받아 실행에만 집중한다.

---

## 설정값: `PROCESSOR_CONFIG`

| 키 | 기본값 | 의미 |
|----|--------|------|
| `gemini_model` | `"gemini-2.0-flash"` | reformulation 모델 |
| `understanding_model` | `"gemini-2.0-flash"` | understanding 모델 |
| `history_max_turns` | `5` | 보존할 최대 대화 턴 수 |

---

## `QueryContext`

전처리 결과를 담는 Pydantic BaseModel.

| 필드 | 타입 | 설명 |
|------|------|------|
| `original_query` | `str` | 사용자 원본 질문 (LLM 프롬프트에 사용) |
| `search_query` | `str` | 실제 벡터/BM25 검색에 사용할 쿼리 |
| `reformulated` | `str \| None` | 지시어 재작성 결과 (디버그) |
| `understood` | `str \| None` | 구어체 변환 결과 (디버그) |
| `query_type` | `str` | `"greeting"` \| `"meta"` \| `"existence"` \| `"content"` |
| `claim_numbers` | `list[int]` | 추출된 청구항 번호 (`[1, 3]` 등) |

---

## 패턴 상수

| 상수 | 역할 |
|------|------|
| `_REFERENTIAL_PATTERN` | `그럼`, `거기`, `그건` 등 지시어 감지 |
| `_COLLOQUIAL_PATTERN` | `알려줘`, `뭐야`, `있어` 등 구어체 어미 감지 |
| `_GREETING_PATTERN` | 인사·감정·날씨·일상 잡담 감지 |
| `_META_PATTERN` | "파일 몇 개야?", "어떤 문서들 있어?" 감지 |
| `_EXISTENCE_PATTERN` | "에너지 관련 특허 있어?" 등 존재 확인 감지 |
| `_TOPIC_LIST_PATTERN` | "에너지 관련 파일 뭐있어" 등 주제별 파일 목록 감지 |
| `_CLAIM_NUM_PATTERN` | "제N항", "N항" → "청구항 N" 정규화 |
| `_CLAIM_NUM_EXTRACT` | "청구항 N" → 정수 N 추출 |

---

## 주요 함수

### `trim_history(history, max_turns) -> list`
chat_history를 최근 `max_turns` 턴(= 메시지 2개)으로 슬라이싱한다.

---

### `should_reformulate(query, history) -> bool`
지시어 감지 **AND** chat_history 존재 시 True. 두 조건 모두 충족해야 reformulation 의미가 있다.

---

### `reformulate_query(query, history, req_id) -> str`
이전 대화 맥락 참고 → 지시어를 구체적 표현으로 재작성. 실패 시 원본 쿼리 반환.

---

### `should_understand(query) -> bool`
구어체 어미 감지 **OR** 15자 미만 짧은 질문 시 True.

---

### `understand_query(query, req_id) -> str`
구어체 질문 → 벡터 검색 최적화 키워드 문장으로 변환. LLM이 `[[CHITCHAT]]`을 반환하면 greeting으로 처리. 실패 시 원본 반환.

#### `[[CHITCHAT]]` 판정 기준
- 순수 사교적·개인적 표현(인사, 감정, 날씨, 개인 식사 계획 등)
- `사내/회사/대표/직원/CEO/복지/규정/연락처/전화` 단어가 하나라도 있으면 `[[CHITCHAT]]` 금지

---

### `_normalize_claim_terms(text) -> str`
"제N항", "N항" → "청구항 N" 정규화. understanding 전후 양쪽에 적용해 BM25 인덱스와 표기를 일치시킨다.

---

### `_extract_claim_numbers(text) -> list[int]`
"청구항 1과 청구항 3" → `[1, 3]` 추출. `QueryContext.claim_numbers`에 저장.

---

### `classify_query(query) -> QueryType`
패턴 기반 라우팅. LLM 호출 없음.

분류 우선순위:
1. `greeting` → `_GREETING_PATTERN` 매칭
2. `meta` → `_TOPIC_LIST_PATTERN` 또는 `_META_PATTERN` 매칭
3. `existence` → `_EXISTENCE_PATTERN` 매칭
4. `content` → 위 패턴 미매칭

---

### `process_query(query, chat_history, req_id) -> QueryContext`
메인 진입점.

```
history trim
  ↓
지시어 감지 + history 있음? → reformulation → base_query 갱신
  ↓
classify_query(base_query) → query_type 결정
  ↓
구어체/짧은 질문? → understand_query
  └── [[CHITCHAT]] 반환 → query_type = "greeting"
  ↓
_normalize_claim_terms → _extract_claim_numbers
  ↓
QueryContext 반환
```

---

## 설계 포인트

### 1. 조건부 LLM 호출
모든 질문에 reformulation + understanding을 실행하면 매 턴 Gemini 최대 2회 추가 호출. 필요한 경우에만 실행해 비용과 지연 최소화.

### 2. Greeting 판정 범위 제한
`[[CHITCHAT]]`은 순수 사교적 표현에만 적용. 회사·직원·복지 키워드가 있으면 content로 처리해 문서 검색을 시도한다. (회사 정보를 LLM이 지어내는 환각 방지)

### 3. 청구항 번호 추출
BM25 IDF 희석 문제를 우회하기 위해 `claim_numbers`를 별도 추출 → `rag_chain.retrieve()`에서 헤더 직접 매칭에 사용.

### 4. routing은 base_query 기준
reformulation 이전 기준으로 routing해 변환 과정에서 질문 의도가 바뀌는 것을 방지.

---

## 의존성

### 내부
- `llm_api.get_client`

### 외부
- `pydantic`, `google.genai`, `logging`, `os`, `re`

---
최종 수정: 2026-03-27
관련 파일: `src/query_processor.py`
---
