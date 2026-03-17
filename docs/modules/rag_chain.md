# rag_chain.py
2026-03-17;
phase 1 기준의 핵심 RAG 오케스트레이션 모듈이며, 검색과 생성 계층을 연결해 실제 문서 기반 질의응답 수행.
## 개요
`rag_chain.py`는 사용자 질문을 받아, 벡터 검색 → threshold 필터링 → 컨텍스트 조합 → 프롬프트 생성 → LLM 호출 → 출처 정리까지 수행하는 RAG 체인의 핵심 오케스트레이션 모듈이다.  

현재 구현은 **Phase 1 기준 코어 파이프라인**에 해당하며, 문서 검색과 답변 생성의 가장 기본적인 end-to-end 흐름을 구성한다.

이 파일의 핵심 목적은 다음과 같다.

- 사용자 질문을 기준으로 관련 문서 청크 검색
- 검색 결과 중 관련성 낮은 청크 제거
- 관련 문서가 없을 경우 fallback 처리
- LLM에 전달할 문맥 구성
- 문서 기반 답변 생성
- 출처와 검색 통계 포함 결과 반환

즉, 이 모듈은 **검색 계층(vector_db)** 과 **생성 계층(LLM)** 사이를 연결하여, 실제 문서 기반 질의응답을 동작시키는 중심 레이어이다.

---

## 역할
이 파일은 RAG 질의응답 흐름 전체를 조합하고 제어하는 역할을 한다.

구체적으로 다음 책임을 가진다.

- 사용자 질문 수신
- ChromaDB 검색 결과 정규화
- distance threshold 기반 필터링
- 문서 없음 fallback 분기
- LLM 입력용 context block 생성
- 시스템 프롬프트와 질문 결합
- Gemini 호출 및 재시도 처리
- citation 정리
- 최종 결과 구조화

즉, ChromaDB나 Gemini SDK를 직접 세부적으로 다루는 모듈이 아니라,  
각 하위 계층이 제공하는 기능을 연결하는 **순수 오케스트레이션 레이어**이다.

---

## 현재 구현 범위
현재 이 파일은 **Phase 1** 수준의 RAG 체인 코어를 구현한 상태이다.

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

### 아직 본격적으로 확장되지 않은 요소
- multi-turn 대화 맥락 활용
- metadata 사전 필터 자동 추출
- reranking
- answer post-processing 고도화
- 응답 품질 평가 자동화
- citation 세분화

즉, 현재 구조는 **핵심 뼈대는 갖추었지만, 후속 Phase에서 확장될 것을 전제로 한 기초 버전**이라고 볼 수 있다.

---

## 이 파일이 필요한 이유
문서 검색과 LLM 호출을 각각 따로 구현해도, 실제 질의응답 시스템으로 동작시키려면 다음과 같은 중간 계층이 필요하다.

- 검색 결과를 그대로 LLM에 던져도 되는지 판단해야 함
- 관련 없는 결과를 걸러야 함
- 통과 청크가 없으면 LLM 호출을 생략해야 함
- 여러 청크를 하나의 문맥으로 조립해야 함
- 출처와 답변을 함께 정리해야 함
- 검색 문제와 생성 문제를 구분할 수 있어야 함

`rag_chain.py`는 이 역할을 맡는 파일이다.  
즉, 단순 검색 함수나 단순 LLM 호출 함수가 아니라, **문서 기반 QA 파이프라인을 실제 동작 가능한 형태로 조합하는 계층**이다.

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
- `distance_threshold`: 관련 문서로 인정할 최대 distance
- `max_context_chars`: LLM에 넘길 컨텍스트 최대 길이

이 구조 덕분에 검색/생성 파라미터를 코드 여러 곳에서 수정하지 않고 한 곳에서 제어할 수 있다.

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

#### 의미
이 프롬프트는 단순 답변 생성이 아니라, **grounded generation**을 유도하기 위한 핵심 제어 장치이다.

---

## 반환 타입

### `Citation`
출처 정보를 표현하는 dataclass이다.

#### 필드
- `chunk_id`
- `header`
- `source_file`
- `distance`

#### 목적
사용자에게 보여줄 출처 정보를 구조적으로 관리하기 위함이다.

---

### `RagResult`
최종 질의응답 결과를 표현하는 dataclass이다.

#### 필드
- `answer`
- `citations`
- `retrieved_count`
- `passed_threshold`
- `top_distance`
- `fallback`
- `fallback_reason`

#### 의미
이 구조는 단순 답변 문자열만 반환하지 않고,  
검색 상태와 fallback 여부까지 함께 반환하여 디버깅과 운영성을 높인다.

---

## 주요 함수

### `retrieve(query, req_id, n_results=..., filters=None) -> list[dict]`
ChromaDB 검색 결과를 정규화된 청크 리스트로 변환하는 함수이다.

#### 역할
- `query_collection()` 호출
- Chroma raw 결과(`ids`, `documents`, `metadatas`, `distances`) 추출
- 각 결과를 평탄화하여 `list[dict]` 구조로 변환

#### 반환 청크 구조
각 청크는 다음 정보를 포함한다.

- `chunk_id`
- `text`
- `metadata`
- `distance`
- `header`
- `source_file`

#### 사용 목적
Chroma raw 결과는 중첩 리스트 구조이므로, 이후 단계에서 다루기 편하도록 정규화하기 위함이다.

---

### `filter_by_threshold(chunks, req_id, threshold=...) -> list[dict]`
검색 결과 중 distance 기준으로 관련 없는 청크를 제거하는 함수이다.

#### 동작 방식
- `distance`가 `threshold` 이하인 청크만 통과

#### 의미
ChromaDB는 관련 없어도 top-k를 반환할 수 있으므로,  
이 필터가 없으면 LLM이 엉뚱한 문서를 근거로 답변할 수 있다.

#### 목적
검색 결과 품질을 최소한으로 보정하고, hallucination 가능성을 낮추기 위함이다.

---

### `build_context_block(chunks, req_id, max_chars=...) -> str`
통과 청크들을 LLM 입력용 문맥 문자열로 조합하는 함수이다.

#### 동작 방식
- 각 청크 앞에 `[문서 n]` 형식 부여
- 헤더와 source_file을 함께 포함
- 헤더의 Markdown 기호(`#`) 제거
- `max_context_chars`를 넘기면 하위 rank 청크부터 제외

#### 목적
LLM이 어떤 문서의 어떤 섹션을 참고하고 있는지 인식할 수 있도록 하고,  
토큰 예산을 넘지 않도록 상위 검색 결과를 우선 보존하기 위함이다.

#### 중요한 포인트
헤더에서 `##` 같은 Markdown 기호를 제거하는 이유는,  
출처 표기를 사용자 친화적으로 만들고 LLM 인용 형식을 정리하기 위함이다.

---

### `build_prompt(query, context, chat_history=None) -> str`
시스템 프롬프트, 참고 문서, 질문을 하나의 문자열로 조합하는 함수이다.

#### 역할
- `SYSTEM_PROMPT` 삽입
- 필요 시 이전 대화 이력 포함
- `[참고 문서]` 블록 삽입
- `[질문]` 블록 삽입

#### 현재 상태
Phase 1에서는 `chat_history`가 실질적으로 비활성 상태이며,  
Phase 2 이후 multi-turn 확장을 고려한 형태만 미리 잡혀 있다.

---

### `generate_answer(prompt, req_id, max_retries=3) -> str`
Gemini를 호출하여 답변 텍스트를 생성하는 함수이다.

#### 역할
- `llm_api.get_client()`를 통해 Gemini 클라이언트 획득
- temperature 0.0으로 호출
- 빈 응답 방지
- rate limit 발생 시 재시도
- 일일 quota 초과 시 즉시 중단

#### 특징
- 빈 문자열 응답도 오류로 처리
- 429 / quota / resource exhausted 관련 에러를 감지
- RPM 제한은 60초 단위 백오프로 재시도
- RPD 제한은 재시도 없이 실패 처리

#### 사용 목적
문서 기반 답변 생성의 안정성을 높이고, LLM 호출 실패 상황을 상위 계층에서 일관되게 처리하기 위함이다.

---

### `format_citations(chunks) -> list[Citation]`
통과 청크를 바탕으로 중복 제거된 출처 리스트를 생성하는 함수이다.

#### 중복 제거 기준
- `(header, source_file)` 조합

#### 이유
`chunk_id`는 항상 고유하므로 출처 중복 제거 기준으로 적합하지 않다.  
같은 섹션에서 여러 청크가 통과했을 경우, 사용자에게 같은 출처가 반복 표시되는 것을 막기 위해 `(header, source_file)`를 사용한다.

#### 특징
- 헤더의 Markdown 기호 제거
- context block과 citation 목록의 헤더 표기 방식 일치

---

### `ask(query, chat_history=None, filters=None) -> RagResult`
이 파일의 핵심 메인 진입점이다.

#### 역할
사용자 질문을 받아 전체 RAG 흐름을 실행하고 최종 `RagResult`를 반환한다.

#### 처리 단계
1. 질문 수신
2. 검색 수행
3. threshold 필터링
4. 통과 청크 없으면 fallback 반환
5. context 조합
6. prompt 생성
7. LLM 호출
8. citation 정리
9. 결과 객체 반환

#### fallback 분기
다음 세 가지 fallback reason을 구분한다.

- `retrieval_error`
- `no_docs`
- `llm_error`

#### 의미
이 구분 덕분에 “문서가 없음”과 “시스템 오류”를 운영 레벨에서 다르게 다룰 수 있다.

---

## 처리 흐름

### 1. 질문 입력
사용자 질문이 `ask()`로 들어온다.

### 2. 벡터 검색
`retrieve()`가 `vector_db.query_collection()`을 호출하여 관련 청크를 가져온다.

### 3. threshold 필터링
`filter_by_threshold()`가 관련성 낮은 청크를 제거한다.

### 4. fallback 판단
통과 청크가 없으면 LLM 호출 없이 즉시 fallback 결과를 반환한다.

### 5. context 조합
`build_context_block()`이 통과 청크를 LLM 입력용 텍스트 블록으로 변환한다.

### 6. 프롬프트 생성
`build_prompt()`가 시스템 규칙, 참고 문서, 질문을 결합한다.

### 7. 답변 생성
`generate_answer()`가 Gemini를 호출해 답변을 생성한다.

### 8. citation 정리
`format_citations()`가 사용자에게 보여줄 출처 목록을 만든다.

### 9. 최종 결과 반환
`RagResult` 형태로 답변, 출처, 검색 통계, fallback 정보를 반환한다.

---

## 입력과 출력

### 입력
- 사용자 질문 문자열
- 선택적 대화 이력
- 선택적 metadata 필터

### 출력
- `RagResult`
  - 답변
  - 출처 리스트
  - 검색 개수
  - threshold 통과 개수
  - top distance
  - fallback 여부
  - fallback 원인

---

## 의존성

### 내부 의존성
- `llm_api.get_client`
- `llm_api.GeminiAPIError`
- `vector_db.query_collection`

### 외부 라이브러리
- `google.genai.types`
- `logging`
- `os`
- `re`
- `time`
- `uuid`
- `dataclasses`
- `pathlib`

### 의존성 설계 포인트
이 파일은 ChromaDB나 Gemini SDK를 직접 강하게 다루지 않고,  
하위 모듈이 제공하는 인터페이스만 호출하는 구조를 유지한다.

즉, 이 파일의 강점은 **오케스트레이션만 담당하고 세부 구현은 위임한다는 점**이다.

---

## 설계 포인트

### 1. 오케스트레이션 레이어 분리
검색과 생성을 직접 구현하지 않고 조합만 담당하여 책임을 명확히 나눴다.

### 2. 검색 결과 정규화
Chroma raw 결과를 초기에 평탄화하여 이후 단계의 복잡도를 줄였다.

### 3. threshold 기반 안전장치
검색 결과를 그대로 사용하지 않고, 최소한의 관련성 기준을 적용한다.

### 4. fallback 정책 명확화
문서 없음 / 검색 오류 / 생성 오류를 구분하여 운영 대응이 가능하게 했다.

### 5. 검색 통계 포함 반환
`retrieved_count`, `passed_threshold`, `top_distance`를 결과에 포함하여  
문제 원인을 빠르게 진단할 수 있도록 했다.

### 6. citation 표기 정리
헤더의 Markdown 기호를 제거하여 사용자에게 더 읽기 좋은 출처를 제공한다.

### 7. Phase 확장 고려
`chat_history`, `filters` 같은 인자를 미리 포함하여, 이후 다중 턴 대화나 metadata pre-filtering으로 확장할 수 있도록 설계했다.

---

## 예외 및 주의사항

### 1. 현재는 Phase 1 수준 구현
핵심 흐름은 완성되어 있지만, 아직 검색 품질 최적화나 답변 품질 고도화는 추가 여지가 많다.

### 2. threshold 값은 실험적으로 조정 필요
현재 기본값은 0.5지만, 실제 문서와 임베딩 모델에 따라 적절한 값은 달라질 수 있다.

### 3. context 길이 제한은 단순 cut-off 방식
현재는 `max_context_chars`를 넘기면 하위 rank 청크를 제외하는 방식이며, 보다 정교한 요약/압축 전략은 아직 없다.

### 4. reranking 없음
현재 검색 순서는 순수 벡터 유사도 결과에 의존한다.  
즉, 초기 검색 품질이 답변 품질에 직접적인 영향을 준다.

### 5. citation granularity는 섹션 단위
현재 출처는 `(header, source_file)` 기준으로 묶이므로, 더 세밀한 문장/행 단위 citation은 지원하지 않는다.

### 6. 테스트 케이스는 내장형
`_TEST_CASES`가 파일 내부에 포함되어 있어, 현재는 간단한 sanity check에는 좋지만 장기적으로는 별도 테스트 파일로 분리할 여지가 있다.

---

## CLI 및 테스트

### `_print_result(result)`
`RagResult`를 사람이 읽기 쉬운 형태로 출력하는 보조 함수이다.

### `_TEST_CASES`
현재 Phase 1 검증용 질문 세트가 파일 내부에 정의되어 있다.

#### 포함된 테스트 유형
- 문서에 있는 질문
- 문서에 없는 질문
- 수치/날짜 질문
- 모호한 질문
- 엉뚱한 질문

### 실행 방법

#### 단일 질문 실행
```bash
cd src && uv run python rag_chain.py "회사 주요 특허는 무엇인가요?"
```

---
최종 수정: 2026-03-17
관련 파일: 'src/rag_chain.py'
---