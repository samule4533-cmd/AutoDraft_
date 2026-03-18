# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

사내 챗봇을 위한 PDF 문서 파싱 → 청킹 → 벡터DB 적재 파이프라인. 한국어 기업 문서(특허, 인증서, 회사소개서 등) 처리에 특화된 RAG 인제스트 시스템.

## Architecture

### 데이터 흐름

```
data/raw/{company|sample_notices}/{subdir}/{file}.pdf
    → pdf_parser.py (Gemini File API로 Markdown 변환)
    → chunker.py (헤더 기반 청크 분리)
    → output_writer.py (data/processed/에 .md, .json, vector_chunks.json 저장)
    → vector_db.py (ChromaDB에 임베딩 적재)
    → data/vector_store/chroma/ (영구 저장)
```

### 모듈 역할

- **`pdf_parser.py`**: 파이프라인 진입점. PDF를 Gemini File API에 업로드 → Markdown 생성 → `normalize_markdown_headings()`로 볼드 제목을 헤더로 보정. `async_main()`이 실제 실행 함수.
- **`chunker.py`**: Markdown 헤더(`#`, `##`, `###`) 기준 1차 분리, 1500자 초과 섹션은 빈 줄 기준 2차 분리(`paragraph_group`). 청크 간 overlap 없음.
- **`vector_db.py`**: `paraphrase-multilingual-MiniLM-L12-v2` 로컬 임베딩, ChromaDB PersistentClient. `upsert_chunks_to_chroma()`는 배치 50개 단위로 적재. `query_collection()`이 검색 진입점.
- **`llm_api.py`**: Gemini 클라이언트 싱글턴, `safe_json_load()`는 JSON 파싱 실패 시 3단계 폴백(직접 파싱 → 펜스 코드블록 추출 → `{}` 복구).
- **`output_writer.py`**: 처리 결과를 `.md`, `.json`, `vector_chunks.json`, `parse_report.json`, `fields.json`으로 저장.
- **`field_extract.py`**: 조달/입찰 공고 문서 전용 금액 필드 추출. 현재 company 모드에서는 비활성화.
- **`image_parser.py`**: PyMuPDF로 이미지 추출 + Gemini 캡션 생성. `ENABLE_IMAGE_CAPTIONS=False`로 현재 비활성화.
- **`company_ingest.py`**: 미구현 빈 파일.

### 한글 파일명 처리

`pdf_parser.py`의 `_upload_and_wait()`는 한글 파일명 업로드 오류를 피하기 위해 임시 디렉토리에 영문명(`upload_input.pdf`)으로 복사 후 업로드한다.

### ChromaDB 컬렉션 구성

- 컬렉션명: `ninewatt_company_local` (company 문서)
- 저장 위치: `data/vector_store/chroma/`
- 거리 메트릭: cosine similarity
- 메타데이터 필터: `doc_type`, `document_id`, `header` 등으로 where 절 필터링 가능

## 현재 미구현 영역

- RAG 응답 생성 (검색 결과 → LLM 프롬프트 조립 → 응답): 완전 미구현
- `company_ingest.py`: 여러 PDF 일괄 처리 로직 없음
- `main.py`: 실질적 오케스트레이션 없음

## 내가 생각한 RAG 아키텍처 마일스톤
이 단계는 프로젝트의 핵심이라고 생각하고 있으며, 단순 예제 수준이 아니라 **실제 서비스 MVP에 바로 연결 가능한 구조**를 기준으로 설계/구현 방향을 정리해주길 바란다.

---

## 현재 목표

`rag_chain.py`에서 구현하려는 기본 기능은 아래와 같다.

1. `query_collection()`을 호출해 ChromaDB에서 유사 청크 n개 검색
2. distance threshold를 적용해 관련성이 낮은 청크 제외
3. 검색된 청크 본문과 메타데이터(`header`, `source_file`)를 조합해 프롬프트 구성
4. Gemini LLM 호출
5. 자연어 답변 반환
6. 답변에 출처 인용 포함 (`chunk_id`, `header`, `source_file`)

다만, 위 기능을 단순 나열형으로 구현하는 것이 아니라, **환각 방지**, **운영 안정성**, **추후 확장성**까지 고려한 구조로 설계하고 싶다.

---

## 2. 현재 생각 중인 마일스톤

### Phase 1: MVP 코어 구현
지금 당장 필요한 단계이며, 아래 항목은 필수라고 생각한다.

- **유사도 검색 및 필터링**
  - `query_collection()` 호출
  - distance threshold 적용
  - 노이즈 청크 제거

- **Fallback 강제**
  - threshold 적용 후 남은 청크가 0개이면 Gemini 호출을 생략하거나
  - 시스템 차원에서 `"사내 문서에 해당 내용이 없습니다"`처럼 답변하도록 강제
  - 목적: 환각 방지

- **프롬프트 조합 및 생성**
  - 검색된 본문 + 메타데이터를 조합해 Gemini에 전달
  - 답변 시 `chunk_id`, `header`, `source_file` 기반 출처 인용 강제

## Phase 2: 대화형 UX 통합

Phase 2는 서버 연동 시점에 맞춰, 단발성 질의응답을 넘어 **대화형 문서 QA**로 확장하는 것을 목표로 한다.

### 목표
- 이전 대화 맥락을 반영한 질의응답 지원
- 연속 질문에 대한 문맥 해석 강화
- 서버 연동이 가능한 형태로 RAG 체인 구조 정리
- API 입력/출력 구조 표준화

### 구현 항목
- `chat_history` 반영
- **history 길이 제한 전략 결정 (구현 전 반드시 결정)**
  - 대화가 쌓일수록 history가 그대로 프롬프트에 들어가면 컨텍스트 창 초과 및 응답 지연 발생
  - 사내 문서 QA 특성상 **최근 3~5턴만 유지**가 적합 (대화가 짧고 주제 전환 잦음)
  - 전략 선택지: 최근 N턴 슬라이싱 / 총 글자 수 기준 슬라이딩 / 오래된 history LLM 요약 압축
- **Query Reformulation — 조건부 실행**
  - 모든 질문에 reformulation LLM 호출 시 매 턴마다 Gemini 호출 2회 → 비용/지연 2배
  - `"그럼"`, `"거기"`, `"그건"`, `"그것"`, `"해당"`, `"방금"` 등 지시어 감지 시에만 실행
  - chat_history가 없으면 reformulation 생략 (독립 질문이므로)
  - reformulation 실패 시 원본 query를 그대로 사용하는 fallback 필수
- 연속 질문 대응
  예: `"그럼 거기는 얼마야?"` 같은 질문을 이전 맥락 기반으로 재해석
- `rag_chain.py`를 서버 연결 가능한 형태로 정리
  - `RagResult` dataclass → Pydantic BaseModel 교체 (FastAPI JSON 직렬화 자동화)
  - `reformulated_query: str | None` 필드 추가 (디버깅: 재작성 결과 추적용)
- 입력/출력 구조 고정

### 진행 순서
1. history 길이 제한 전략 결정 (N턴 기준 권고)
2. `chat_history` 반영 방식 정의 및 `rag_chain.py` 정리
3. Query Reformulation 구현 (조건부 실행)
4. FastAPI 기반 최소 API 레이어 구성
5. `POST /chat` 엔드포인트 구현 및 테스트
6. Postman 또는 Swagger를 통한 API 검증

### 완료 기준
- 이전 대화 맥락이 포함된 질문을 처리할 수 있다
- `/chat` API를 통해 질문과 응답을 주고받을 수 있다
- `chat_history`를 입력받아 답변 생성에 반영할 수 있다
- 응답 구조가 JSON 형태로 고정된다
- history가 길어져도 컨텍스트 창 초과가 발생하지 않는다
- reformulation 여부와 재작성된 query가 응답에 포함되어 디버깅 가능하다

### Phase 3: 검색 품질 고도화
E2E 테스트 후 도입 검토 예정.

- 하이브리드 검색
  - 벡터 검색 + 키워드 검색(BM25 등)

- Small-to-Big Retrieval
  - 검색은 작은 청크로
  - LLM에는 더 큰 문맥 단위 전달

- 메타데이터 사전 필터링
  - 질문에서 연도/부서 등을 추출해 ChromaDB `where` 조건으로 범위 축소

---

## 3. 내가 중요하게 보는 방향

- 지금은 **과한 고도화보다 안전한 MVP**가 중요하다.
- 모르면 모른다고 답하는 챗봇이 더 좋다.
- 출처 없는 답변은 최대한 막고 싶다.
- 구조는 단순하되, Phase 2/3로 자연스럽게 확장 가능해야 한다.
- `rag_chain.py`는 프로젝트 핵심 80%라고 생각한다.

---

## 4. 구현 시 반드시 고려해야 하는 포인트

### (1) 거리 metric 확인
- ChromaDB에서 사용하는 거리 지표가 L2인지 cosine distance인지 반드시 확인해야 한다.
- threshold 비교 부등호 방향이 metric에 따라 달라질 수 있다.
- 이 부분이 틀리면 retrieval 로직 전체가 반대로 동작할 수 있다.
- 따라서 metric과 threshold는 하드코딩하지 말고 config로 분리하는 것이 좋다.

### (2) retrieval과 generation 평가 분리
품질 테스트 시 아래를 분리해서 볼 수 있어야 한다.

1. 검색기가 맞는 문서를 찾았는가
2. 생성기가 찾은 문서 안에서만 답했는가

이렇게 나눠야 threshold 문제인지, 프롬프트 문제인지, chunking 문제인지 디버깅이 가능하다.

---

## 5. 네가 해줬으면 하는 일

아래 항목을 중심으로 종합적으로 정리해줘.

1. 현재 설계가 MVP 기준으로 충분한지 평가
2. 부족하다면 무엇을 추가해야 하는지
3. `rag_chain.py`를 어떤 함수 단위로 나누면 좋은지
4. threshold / fallback / citation을 어떤 방식으로 넣는 것이 좋은지
5. 프롬프트를 어떤 원칙으로 구성해야 하는지
6. 나중에 chat history / hybrid retrieval / metadata filtering / small-to-big을 붙이기 쉬운 구조인지 검토
7. 실제 운영 가능한 수준으로 만들기 위한 최소 방어 로직 제안
8. 가능하면 `rag_chain.py`의 pseudocode 또는 Python skeleton 제안

---

## 6. 답변할 때 꼭 반영해줬으면 하는 관점

아래 관점에서 다양하게 판단해줘.

- 아키텍처 관점
- 운영 안정성 관점
- 환각 방지 관점
- 디버깅/평가 관점
- 추후 확장성 관점
- 응답 품질 관점
- 구현 난이도 대비 효율 관점

## 개발 방향 및 작업 원칙

### 핵심 목표
이 프로젝트는 단순 PDF 텍스트 추출기가 아니라, 회사 문서를 신뢰 가능한 지식 자원으로 변환하기 위한 파이프라인이다.  
최종 목표는 PDF 문서 파싱 → 청킹 → 벡터DB 적재를 거쳐, 1차적으로 사내 챗봇이 근거 기반 질의응답을 수행할 수 있도록 만드는 것이다.

### 중요 기준
- 정확도 우선
- 회사자료/정형 문서 특화
- 청킹 및 검색 품질 중심
- RAG 연결을 고려한 구조 유지
- 실사용 가능한 결과 지향

### 작업하면서 중점적으로 본 점
- 한글/숫자/표가 최대한 깨지지 않도록 파싱 정확도 개선
- OCR, Markdown 변환, 후처리 보강 방식 비교 및 적용
- 청킹이 문맥을 해치지 않도록 구조 유지
- 벡터DB 적재 이후 검색 가능한 형태의 metadata 유지
- 겉보기 결과보다 실제 업무 활용 가능성을 더 중요하게 판단
- 항상 구체적인 내용과 어떤 작업을 하면 그 이유를 명확하게 드러내도록

### 유의사항
- 공고문/회사자료는 숫자, 조건, 표 정보 손실에 민감하므로 단순 텍스트 추출 품질만으로 판단하면 안 된다.
- preview 기준 확인만으로는 부족할 수 있어, 필요 시 full chunk 기준 검증이 필요하다.
- 파싱, 청킹, 검색은 분리된 단계가 아니라 최종 답변 품질에 함께 영향을 주는 하나의 흐름으로 본다.