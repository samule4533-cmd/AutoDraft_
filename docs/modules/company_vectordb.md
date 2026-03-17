# company_vectordb.py

## 개요
`company_vectordb.py`는 회사 문서 파싱 결과를 수집하여 ChromaDB에 적재하는 배치 스크립트이다.  
`company_ingest.py`를 통해 생성된 `vector_chunks.json` 파일들을 모아 하나의 컬렉션에 upsert하고, 적재가 정상적으로 수행되었는지 sanity check까지 수행한다.

이 파일의 핵심 목적은 다음과 같다.

- 파싱 결과 청크 파일 일괄 수집
- 전체 청크를 ChromaDB 컬렉션에 적재
- 필요 시 기존 컬렉션 초기화 후 재생성
- 적재 결과 요약 출력
- 실제 컬렉션 내 청크 수 확인

즉, 이 모듈은 **문서 파싱 결과를 검색 가능한 벡터 저장소 형태로 전환하는 배치 적재 단계**를 담당한다.

---

## 역할
이 파일은 `data/processed/parsing_result_company/` 하위의 모든 `vector_chunks.json`을 탐색하여, 각 청크를 ChromaDB에 upsert하는 오케스트레이션 레이어이다.

구체적으로 다음 책임을 가진다.

- 전체 파싱 결과 파일 수집
- 적재 대상 청크 통합
- 환경 변수 기반 적재 설정 로드
- 임베딩 모델 변경 시 컬렉션 초기화 처리
- `vector_db.py`의 적재 함수 호출
- 적재 결과와 컬렉션 상태 출력

즉, 개별 벡터 연산이나 컬렉션 내부 동작은 `vector_db.py`가 담당하고, 이 파일은 **회사 문서용 일괄 적재 실행 진입점** 역할을 한다.

---

## 이 파일이 필요한 이유
RAG 시스템에서 파싱된 문서는 그대로는 검색할 수 없다.  
문서 내용을 청크 단위로 분리한 뒤, 각 청크를 임베딩하고 벡터 DB에 저장해야 유사도 기반 검색이 가능해진다.

특히 회사 문서처럼 여러 PDF에서 생성된 청크를 하나의 검색 컬렉션으로 통합하려면 다음 작업이 필요하다.

- 여러 결과 폴더의 `vector_chunks.json`을 한 번에 모으기
- 동일한 컬렉션에 적재하기
- 임베딩 모델 변경 시 차원 불일치 문제 방지하기
- 적재가 정상적으로 되었는지 확인하기

`company_vectordb.py`는 이러한 요구를 처리하는 **벡터 저장소 적재용 배치 스크립트**이다.

---

## 주요 구성 요소

### 상수 및 설정
- `COMPANY_OUTPUT_ROOT`  
  회사 문서 파싱 결과가 저장된 루트 디렉터리

- `CHROMA_DIR`  
  ChromaDB persist 경로

- `EMBEDDING_PROVIDER`  
  사용할 임베딩 제공자 (`local`, `openai` 등)

- `COLLECTION_NAME`  
  적재 대상 ChromaDB 컬렉션명

- `CHROMA_RESET`  
  `true`일 경우 기존 컬렉션을 삭제 후 재생성

---

## 주요 함수

### `collect_all_chunks() -> List[Dict[str, Any]]`
`COMPANY_OUTPUT_ROOT` 하위의 모든 `vector_chunks.json` 파일을 탐색하고, 내부 청크를 하나의 리스트로 합쳐 반환하는 함수이다.

#### 동작 방식
1. 출력 루트 하위에서 `vector_chunks.json` 파일 검색
2. 각 파일을 JSON으로 로드
3. 전체 청크 리스트에 누적
4. 파일별 로드 개수를 로그로 기록

#### 목적
회사 문서 파싱 결과가 폴더별로 분산 저장되어 있어도, 적재 단계에서는 이를 하나의 통합 청크 리스트로 다루기 위함이다.

#### 반환값
- 전체 청크 리스트 (`List[Dict[str, Any]]`)

---

### `upsert_all() -> None`
이 파일의 메인 적재 로직이다.

#### 역할
- 전체 청크 수집
- 필요 시 컬렉션 초기화
- ChromaDB upsert 실행
- 적재 결과 요약 출력
- sanity check 수행

#### 처리 단계
1. 모든 `vector_chunks.json` 수집
2. 청크가 없으면 안내 메시지 출력 후 종료
3. `CHROMA_RESET` 여부에 따라 컬렉션 삭제/재생성
4. `upsert_chunks_to_chroma()` 호출
5. 적재 완료 정보 출력
6. ChromaDB 컬렉션 개수 확인

---

## 처리 흐름

### 1. 파싱 결과 파일 수집
`data/processed/parsing_result_company/` 하위의 모든 `vector_chunks.json` 파일을 탐색한다.

### 2. 전체 청크 통합
각 JSON 파일의 내용을 읽어 전체 청크 리스트로 합친다.

### 3. 적재 대상 확인
적재할 청크가 하나도 없으면, 먼저 `company_ingest.py`를 실행하라는 메시지를 출력하고 종료한다.

### 4. 컬렉션 리셋 여부 확인
환경 변수 `CHROMA_RESET=true`이면 기존 컬렉션을 삭제하고 다시 생성한다.

이 과정은 특히 임베딩 모델이 바뀌었을 때 중요하다.  
예를 들어, 기존 로컬 임베딩(384차원)으로 만든 컬렉션에 OpenAI 임베딩(1536차원) 데이터를 넣으려 하면 차원 불일치 오류가 발생할 수 있다.

### 5. ChromaDB upsert 실행
`vector_db.upsert_chunks_to_chroma()`를 호출하여 전체 청크를 컬렉션에 적재한다.

기본 `doc_type`은 `"company"`로 지정되어 있어, 이후 메타데이터 필터링 시 활용 가능하다.

### 6. 결과 요약 출력
다음 정보를 콘솔에 출력한다.

- 총 적재 청크 수
- 컬렉션명
- 임베딩 제공자
- ChromaDB 저장 경로

### 7. sanity check 수행
ChromaDB 컬렉션을 다시 열어 실제 count 값을 출력한다.  
이를 통해 적재 함수가 호출되었더라도 실제 저장소에 데이터가 들어갔는지 확인할 수 있다.

---

## 입력과 출력

### 입력
- `data/processed/parsing_result_company/` 하위의 `vector_chunks.json` 파일들
- `.env` 기반 설정값
  - `EMBEDDING_PROVIDER`
  - `CHROMA_COLLECTION_NAME`
  - `CHROMA_RESET`

### 출력
- ChromaDB 컬렉션에 저장된 청크 데이터
- 콘솔 로그 및 요약 출력
- sanity check 결과

---

## 의존성

### 내부 의존성
- `vector_db.get_chroma_dir`
- `vector_db.reset_collection`
- `vector_db.upsert_chunks_to_chroma`
- `vector_db.query_collection`
- `vector_db.print_query_summary`

이 중 실제 현재 실행 흐름에서 핵심적으로 사용하는 것은 다음과 같다.

- `get_chroma_dir`
- `reset_collection`
- `upsert_chunks_to_chroma`

`query_collection`, `print_query_summary`는 현재 코드에서 import되어 있지만 실제 메인 흐름에서는 사용되지 않는다.

### 외부 라이브러리
- `json`
- `logging`
- `os`
- `pathlib`
- `python-dotenv`
- `chromadb`

---

## 컬렉션 및 임베딩 관점에서의 의미

### 컬렉션
이 스크립트는 회사 문서용 ChromaDB 컬렉션을 구성한다.  
기본 컬렉션명은 환경 변수 기준 `ninewatt_company`이다.

### 임베딩 제공자
적재 시 사용하는 임베딩 방식은 `.env`의 `EMBEDDING_PROVIDER`를 따른다.  
따라서 같은 컬렉션을 유지할 때는 **동일한 임베딩 차원과 방식**을 유지해야 한다.

### 코사인 유사도
주석에 명시된 것처럼, 내부 컬렉션은 코사인 유사도 기준으로 동작한다.  
즉, 벡터의 절대 크기보다 방향(각도) 유사성을 중심으로 검색이 수행된다.

---

## 예외 및 주의사항

### 1. 파싱 결과가 없으면 적재할 수 없음
`vector_chunks.json` 파일이 하나도 없으면 적재를 수행하지 않고 종료한다.  
이 경우 먼저 `company_ingest.py`를 통해 문서 파싱을 완료해야 한다.

### 2. 임베딩 모델 전환 시 컬렉션 리셋 필요
기존 컬렉션이 다른 차원의 임베딩으로 생성되어 있다면, 새로운 차원의 벡터를 그대로 넣을 수 없다.  
이 경우 `CHROMA_RESET=true`로 컬렉션을 초기화한 후 다시 적재해야 한다.

### 3. import된 테스트 함수가 현재는 사용되지 않음
파일 상단에 `query_collection`, `print_query_summary`가 import되어 있지만 현재 실행 흐름에서는 사용되지 않는다.  
즉, “쿼리 테스트”라는 파일 설명과 달리 현재 코드는 **적재 + sanity check** 중심으로 동작한다.

### 4. JSON 형식 일관성 필요
각 `vector_chunks.json`의 내부 구조가 `upsert_chunks_to_chroma()`가 기대하는 형식과 일치해야 한다.  
즉, 청크 생성 단계(`chunker.py`, `pdf_parser.py`)의 출력 구조가 안정적으로 유지되어야 한다.

---

## 이 파일이 프로젝트에서 갖는 의미
`company_vectordb.py`는 회사 문서 기반 RAG 시스템에서 **파싱 결과를 실제 검색 가능한 저장소로 전환하는 단계**를 담당한다.

전체 흐름에서 보면 다음 위치에 해당한다.

1. 원본 PDF 수집
2. `company_ingest.py`로 PDF 파싱
3. `company_vectordb.py`로 ChromaDB 적재
4. `rag_chain.py`에서 검색 및 응답 생성

즉, 이 파일은 **문서 전처리 단계와 질의응답 단계 사이를 연결하는 벡터 적재 파이프라인**이다.

---

## 사용 방법

프로젝트 루트 기준:

```bash
cd src && uv run python company_vectordb.py
```

---
최종 수정: 2026-03-17
관련 파일: 'src/company_vectordb.py'
---