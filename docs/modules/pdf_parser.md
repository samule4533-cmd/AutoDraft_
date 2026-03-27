# pdf_parser.py
단일 PDF 기준의 핵심 파싱 모듈이며, 배치 처리에서는 'company_ingest.py'가 이 파일의 'parse_single_pdf()'를 호출.
## 개요
`pdf_parser.py`는 단일 PDF 문서를 Gemini File API를 이용해 Markdown으로 변환하고, 이를 청크 단위로 분리한 뒤 결과를 저장하는 파싱 모듈이다.  
이 파일은 배치 처리용 스크립트가 아니라 **PDF 1개 기준의 파싱 실행 로직**을 담당하며, `company_ingest.py`에서 import되어 배치 파이프라인의 실제 단일 문서 처리 함수로 사용된다.

이 파일의 핵심 목적은 다음과 같다.

- PDF 1개를 Gemini File API에 업로드
- 문서 구조를 보존한 Markdown 생성
- Gemini 출력 정규화
- Markdown을 RAG용 청크로 분리
- 결과를 파일로 저장
- 배치 파이프라인에서 재사용 가능한 단일 문서 파싱 함수 제공

즉, 이 모듈은 **원본 PDF를 RAG 파이프라인에 투입 가능한 구조화 결과로 변환하는 핵심 단일 문서 파서**이다.

---

## 역할
이 파일은 PDF 1개를 입력받아 파싱부터 저장까지 수행하는 오케스트레이션 계층이다.

구체적으로 다음 책임을 가진다.

- 실행 환경 및 경로 설정
- Gemini API 사용 가능 여부 사전 점검
- 출력 디렉터리 생성
- PDF 업로드 및 File API 처리 완료 대기
- Gemini 응답을 Markdown 본문으로 수신
- Markdown 헤더 형식 정규화
- 청킹 수행
- 문서 JSON 및 리포트 생성
- 결과 파일 저장
- 최종 청크 리스트 반환

즉, OCR/모델 호출/청킹/저장을 하나의 단일 PDF 기준 흐름으로 묶어주는 **문서 파싱 진입 모듈**이라고 볼 수 있다.

---

## 이 파일이 필요한 이유
회사 문서 기반 RAG 시스템에서는 PDF를 단순 텍스트로 추출하는 것만으로는 부족하다.  
다음과 같은 요구를 동시에 만족해야 한다.

- 제목, 소제목, 표, 번호 등 문서 구조를 최대한 보존해야 한다
- 표와 수치, 등록번호, 특허번호 같은 중요 정보가 훼손되면 안 된다
- 청킹 전 단계에서 Markdown 형태로 구조화할 필요가 있다
- 파싱 결과를 그대로 사람이 검토할 수 있어야 한다
- 후속 단계인 벡터 DB 적재를 위해 청크 단위 산출물이 필요하다

`pdf_parser.py`는 이러한 요구를 반영하여, 단일 PDF 기준의 **문서 변환 + 정규화 + 청킹 + 저장**을 하나의 흐름으로 묶은 파일이다.

---

## 주요 구성 요소

### 환경 및 경로 설정
이 파일은 단독 실행과 배치 실행을 모두 고려하여 경로를 설정한다.

#### 주요 변수
- `DOC_SOURCE_TYPE`
- `DEFAULT_PDF_SUBDIR`
- `DEFAULT_PDF_NAME`
- `INPUT_DIR`
- `OUTPUT_ROOT`
- `SOURCE_PDF`

#### 동작 방식
- `DOC_SOURCE_TYPE=company`이면 회사 문서 경로 사용
- `DOC_SOURCE_TYPE=notice`이면 공고문 경로 사용
- 기본값 기준 단일 PDF 테스트도 가능하도록 설계

즉, 배치에서는 외부에서 경로를 넘겨받고, 단독 실행 시에는 환경 변수 기반 기본 경로를 사용한다.

---

### Gemini 파싱 설정
#### 주요 변수
- `GEMINI_PDF_MODEL`
- `GEMINI_MAX_OUTPUT_TOKENS`
- `PDF_PARSE_PROMPT`

#### 특징
파싱 프롬프트는 일반 OCR 수준이 아니라, **RAG 최적화와 기업 문서 구조 보존**을 목표로 매우 구체적으로 설계되어 있다.

특히 다음 항목을 강조한다.

- 구조 보존
- 표 유지
- 숫자/번호/기관명 정확 보존
- 특허/인증/회사소개서 등 문서 유형 고려
- 요약 금지
- 페이지 번호 정규화
- Markdown 헤더 구조화

즉, 이 파일의 파싱 품질은 단순히 API 모델 성능뿐 아니라 **프롬프트 설계 품질**에도 크게 의존한다.

---

## 주요 함수

### `get_genai_client() -> genai.Client`
Gemini API 클라이언트를 생성하는 함수이다.

#### 역할
- 환경 변수 `GEMINI_API_KEY` 확인
- 키가 없으면 예외 발생
- 키가 있으면 `genai.Client` 반환

#### 사용 목적
실제 Gemini 호출 전에 인증 가능 여부를 점검하고, 단일 문서 파싱 흐름에서 클라이언트를 생성하기 위함이다.

---

### `build_output_dir(source_pdf: Path, input_dir: Path = None, output_root: Path = None) -> Path`
입력 PDF에 대응되는 출력 디렉터리를 생성하는 함수이다.

#### 역할
- 입력 루트 기준 상대 경로 유지
- 출력 루트 아래 동일한 폴더 구조 복원
- PDF 파일명(stem)을 최종 출력 폴더명으로 사용
- 실제 디렉터리 생성까지 수행

#### 사용 목적
원본 문서와 결과 디렉터리의 대응 관계를 명확히 유지하기 위함이다.

---

### `preflight_check(source_pdf: Path, output_dir: Path) -> Dict[str, Any]`
실행 전 필수 조건을 검사하는 함수이다.

#### 검사 항목
- PDF 파일 존재 여부
- `GEMINI_API_KEY` 존재 여부
- 출력 디렉터리 쓰기 가능 여부

#### 특징
검사 결과를 `{"ok": ..., "checks": [...]}` 구조로 반환하며,  
동시에 `preflight.json`으로 저장될 수 있도록 설계되어 있다.

#### 사용 목적
실제 API 호출 전에 실행 환경 문제를 조기에 감지하기 위함이다.

---

### `normalize_markdown_headings(markdown_text: str) -> str`
Gemini가 줄 전체를 `**제목**` 형태로 출력했을 때 이를 Markdown 헤더로 보정하는 함수이다.

#### 동작 방식
- 이미 `#`, `##` 같은 Markdown 헤더면 그대로 유지
- 줄 전체가 `**텍스트**` 형태면 `## 텍스트`로 변환
- 본문 내 인라인 강조는 그대로 유지

#### 사용 목적
Gemini 출력 형식을 청킹에 더 적합한 구조로 정규화하기 위함이다.

#### 의미
청킹 모듈은 헤더 구조를 기준으로 동작하므로, 이 보정은 검색 품질에도 영향을 줄 수 있는 중요한 후처리 단계이다.

---

### `_upload_and_wait(client: genai.Client, file_path: Path)`
Gemini File API에 PDF를 업로드하고 처리 완료까지 대기하는 내부 함수이다.

#### 역할
- 한글 파일명을 안전한 영문 파일명으로 복사
- Gemini File API 업로드 수행
- 처리 완료까지 폴링
- 실패 시 예외 발생

#### 중요한 특징
Gemini File API 업로드 시 한글 파일명에서 오류가 날 수 있어,  
임시 디렉터리에 `upload_input.pdf` 형태의 영문 파일명으로 복사한 뒤 업로드한다.

이 부분은 실제 운영에서 매우 실용적인 우회 처리이다.

---

### `parse_pdf_to_markdown(source_pdf: Path) -> str`
PDF를 Gemini File API에 업로드하고 Markdown 본문을 생성하는 함수이다.

#### 처리 흐름
1. Gemini 클라이언트 생성
2. `_upload_and_wait()` 호출
3. Gemini 모델에 파일과 프롬프트 전달
4. 응답 텍스트를 Markdown 문자열로 반환

#### 특징
- 비동기 함수
- `temperature=0.0`으로 재현성 확보
- `max_output_tokens` 환경 변수 기반 제어

---

### `parse_single_pdf(source_pdf: Path, input_dir: Path, output_root: Path) -> List[Dict[str, Any]]`
이 파일의 핵심 공개 함수이며, 배치 파이프라인에서 직접 사용하는 단일 PDF 파싱 함수이다.

#### 역할
- 출력 디렉터리 생성
- 사전 검사 수행
- Gemini를 통한 Markdown 변환
- Markdown 헤더 정규화
- 청킹 수행
- 문서 JSON 및 파싱 리포트 생성
- 파일 저장
- 최종 청크 리스트 반환

#### 입력
- `source_pdf`: 처리할 PDF 경로
- `input_dir`: 상대 경로 계산 기준이 되는 입력 루트
- `output_root`: 결과 저장 루트

#### 반환값
- `vector_chunks` 형식의 청크 리스트

#### 프로젝트 내 의미
이 함수가 바로 `company_ingest.py`에서 import되어 호출되는 핵심 함수이다.  
즉, 배치 처리 파이프라인도 결국 이 단일 문서 함수 위에서 동작한다.

---

### `async_main()` / `main()`
단독 실행용 진입점이다.

#### 역할
환경 변수 기준 기본 PDF를 하나 선택하여 `parse_single_pdf()`를 실행한다.

#### 목적
배치 처리 전에 단일 PDF 파싱 결과를 빠르게 검증하기 위한 테스트 진입점이다.

---

## 처리 흐름

### 1. 입력 경로 및 환경 설정
문서 종류와 기본 PDF 경로를 설정한다.

### 2. 출력 디렉터리 생성
원본 PDF에 대응되는 결과 폴더를 생성한다.

### 3. 사전 점검 수행
다음 항목을 검사한다.

- 입력 PDF 존재 여부
- Gemini API 키 존재 여부
- 출력 경로 쓰기 가능 여부

결과는 `preflight.json`으로 저장된다.

### 4. Gemini File API를 통한 Markdown 변환
PDF를 안전한 임시 파일명으로 업로드한 뒤, 처리 완료까지 대기하고 Markdown 응답을 생성한다.

### 5. Markdown 헤더 정규화
Gemini 출력 중 줄 전체 볼드 제목을 `##` 헤더로 보정한다.

### 6. 청킹 수행
`chunker.split_markdown_into_chunks()`를 호출하여 Markdown을 RAG용 청크로 분리한다.

### 7. 결과 구조화
다음 객체를 생성한다.

- `document_json`
- `parse_report`

### 8. 파일 저장
`output_writer.save_outputs()`를 호출하여 결과를 파일로 저장한다.

### 9. 청크 리스트 반환
최종적으로 `vector_chunks` 리스트를 반환하며, 이 값은 벡터 DB 적재 단계에서 활용된다.

---

## 저장 산출물
이 파일은 `output_writer.py`를 통해 다음 결과를 저장할 수 있다.

- 최종 Markdown (`.md`)
- 전체 문서 JSON (`.json`)
- `parse_report.json`
- `fields.json`
- `vector_chunks.json`
- `preflight.json`

즉, 파싱 과정과 결과를 모두 파일 단위로 남길 수 있어, 디버깅과 검증에 유리하다.

---

## 입력과 출력

### 입력
- PDF 파일 1개
- 입력 루트 경로
- 출력 루트 경로
- Gemini API 키
- 파싱 모델 및 토큰 설정

### 출력
- Markdown 본문
- 청크 리스트
- 문서 JSON
- 파싱 리포트
- 저장된 산출물 파일들

---

## 의존성

### 내부 의존성
- `chunker.split_markdown_into_chunks`
- `output_writer.build_document_json`
- `output_writer.save_outputs`

### 외부 라이브러리
- `asyncio`
- `json`
- `logging`
- `os`
- `re`
- `shutil`
- `tempfile`
- `time`
- `pathlib`
- `python-dotenv`
- `google.genai`
- `google.genai.types`

---

## 설계 포인트

### 1. 단일 PDF 처리에 집중
이 파일은 1개 PDF 파싱만 담당하고, 다수 문서 배치 처리는 `company_ingest.py`에 맡긴다.  
즉, 책임 분리가 명확하다.

### 2. Gemini File API 중심 구조
텍스트 추출이 아니라 **파일 자체를 모델에 업로드**하는 방식이라, 레이아웃과 구조 보존 측면에서 더 유리하다.

### 3. 운영을 고려한 사전 점검
실행 전 필수 조건을 검사하고 JSON으로 남겨, 환경 문제와 실행 문제를 명확히 분리할 수 있다.

### 4. 한글 파일명 업로드 우회 처리
실제 운영 중 발생할 수 있는 한글 파일명 문제를 임시 영문 파일 복사 방식으로 우회한다.

### 5. 청킹 및 저장과의 느슨한 결합
청킹과 저장은 별도 모듈에 위임하여, 이 파일은 파싱 중심 역할을 유지한다.

---

## 예외 및 주의사항

### 1. 단독 실행은 PDF 1개만 대상
이 파일은 기본적으로 단일 문서 테스트용이다.  
여러 PDF를 한 번에 처리하려면 `company_ingest.py`를 사용해야 한다.

### 2. Gemini 출력 품질에 의존
Markdown 구조 품질은 Gemini 응답 품질과 프롬프트 설계에 영향을 받는다.  
따라서 파싱 결과 검토가 필요할 수 있다.

### 3. `fields`는 현재 비활성화 상태
현재 회사 문서 흐름에서는 `fields`가 빈 딕셔너리로 유지된다.  
즉, 필드 추출은 구조상 고려되어 있지만 현재 활성화되어 있지 않다.

### 4. 이미지 청크는 현재 사용하지 않음
`image_count=0`, `image_chunk_count=0`으로 고정되어 있어, 현재 파이프라인은 텍스트 중심 구조이다.

### 5. 대용량 PDF는 출력 토큰 수 영향 가능
`GEMINI_MAX_OUTPUT_TOKENS` 값에 따라 긴 문서의 응답이 잘릴 가능성이 있으므로, 필요 시 설정 조정이 필요하다.

---

## 이 파일이 프로젝트에서 갖는 의미
`pdf_parser.py`는 프로젝트 전체 문서 처리 흐름의 핵심 시작점 중 하나이다.  
원본 PDF를 구조화된 Markdown과 RAG용 청크로 변환하여, 이후 벡터 적재와 질의응답이 가능하도록 만든다.

전체 파이프라인에서 보면 다음 위치에 해당한다.

1. 원본 PDF 준비
2. `pdf_parser.py`로 단일 문서 파싱
3. `chunker.py`로 청크 분리
4. `output_writer.py`로 결과 저장
5. `company_vectordb.py`로 ChromaDB 적재
6. `rag_chain.py`로 검색 및 응답 생성

즉, 이 파일은 **원천 PDF를 검색 가능한 지식 단위로 바꾸는 핵심 변환 모듈**이다.

---

## 사용 방법

### 단독 실행
프로젝트 루트 기준:

```bash
cd src && uv run python pdf_parser.py
```

---
최종 수정: 2026-03-27
관련 파일: `src/pdf_parser.py`
---