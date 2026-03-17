# AutoDraft(Chatbot)

> 기업 문서를 대상으로 **PDF 파싱 → 구조적 청킹 → 벡터 적재 → 출처 기반 질의응답**까지 연결한 사내 문서 RAG 파이프라인

## 프로젝트 개요

AutoDraft(Chatbot)는 특허, 인증서, 회사소개서, 기술자료 등 한국어 기업 문서를 구조적으로 파싱하고,  
검색 가능한 청크 단위로 변환한 뒤, 벡터 DB에 적재하여 문서 기반 질의응답을 수행하기 위한 인제스트 중심 RAG 파이프라인입니다.

이 프로젝트는 단순 텍스트 추출이 아니라, 문서의 **제목 구조, 표, 수치, 번호, 식별 정보**를 최대한 보존하면서  
후속 검색과 답변 생성에 적합한 형태로 문서를 정제하는 것을 목표로 합니다.

---

## 목표

- 한국어 기업 문서를 구조적으로 Markdown으로 변환
- 문서 계층, 표, 숫자, 식별자 정보를 최대한 보존
- RAG에 적합한 청크 단위로 분할
- ChromaDB 기반 내부 문서 지식 베이스 구축
- 검색된 문서만 근거로 자연어 질의응답 수행
- threshold 및 fallback 정책으로 환각 가능성 감소

---

## 핵심 기능

### 구조 보존 PDF 파싱
- Gemini File API를 사용해 PDF를 Markdown으로 변환
- 제목, 소제목, 표, 목록, 번호, 수치 정보 최대한 보존
- 특허, 인증서, 회사소개서, 브로슈어 등 기업 문서 유형에 맞춘 파싱 프롬프트 적용

### 검색 친화적 청킹
- Markdown 헤더 기준 1차 분리
- 긴 섹션은 문단 또는 표 단위로 2차 분리
- 표 청크는 헤더 행을 보존하여 독립적으로 해석 가능하게 구성
- `chunk_type`, `has_table`, `chunk_position` 메타데이터 포함

### 결과 산출물 분리 저장
- Markdown 원문 저장
- 문서 전체 JSON 저장
- parse report 저장
- field 결과 저장
- 벡터 적재용 `vector_chunks.json` 저장

### 벡터 저장소 적재 및 검색
- ChromaDB 기반 벡터 저장소 구성
- 로컬 임베딩 / OpenAI 임베딩 전환 가능 구조
- `chunk_id` 기준 upsert 지원
- 코사인 유사도 기반 검색

### Phase 1 RAG 체인
- 벡터 검색 → threshold 필터 → fallback → Gemini 답변 생성
- 출처(헤더, 파일명) 및 검색 통계(retrieved / passed / top_distance) 반환
- 관련 문서 없을 시 LLM 호출 없이 즉시 응답

### 배치 인제스트 파이프라인
- 회사 문서 PDF 배치 스캔 및 파싱
- 재시도 및 실패 로그 기록
- 파싱 결과 일괄 ChromaDB 적재

---

## 프로젝트 구조

```
AutoDraft_clean/
│
├── src/                          # 핵심 소스 코드
│   ├── pdf_parser.py             # PDF → Markdown 변환 (Gemini File API)
│   ├── chunker.py                # Markdown → RAG 청크 분리
│   ├── output_writer.py          # 파싱 결과 파일 저장 (.md / .json / vector_chunks.json)
│   ├── vector_db.py              # ChromaDB 임베딩 적재 및 유사도 검색
│   ├── llm_api.py                # Gemini API 공통 래퍼 (싱글턴, JSON 복구)
│   ├── rag_chain.py              # RAG 체인 코어 (검색 → 필터 → 생성 → 출처 반환)
│   ├── company_ingest.py         # 회사 문서 배치 파싱 파이프라인
│   ├── company_vectordb.py       # 파싱 결과 → ChromaDB 일괄 적재
│   └── image_parser.py           # 이미지 추출 + Gemini 캡션 생성 (현재 비활성화)
│
├── data/
│   ├── raw/                      # 원본 PDF 입력
│   │   └── company/              # 회사 문서 (폴더별 분류)
│   ├── processed/                # 파싱 결과 출력
│   │   └── parsing_result_company/   # .md / .json / vector_chunks.json
│   └── vector_store/
│       └── chroma/               # ChromaDB 영구 저장소
│
├── docs/                         # 문서
│   ├── architecture.md           # 전체 시스템 아키텍처 설계
│   ├── pipeline.md               # 데이터 흐름 및 실행 가이드
│   └── modules/                  # 모듈별 상세 설명
│       ├── pdf_parser.md
│       ├── chunker.md
│       ├── output_writer.md
│       ├── vector_db.md
│       ├── llm_api.md
│       ├── rag_chain.md
│       ├── company_ingest.md
│       └── company_vectordb.md
│
├── tests/                        # 테스트
├── pyproject.toml                # 의존성 관리 (uv)
└── .env                          # 환경변수 (API 키 등, git 제외)
```

---

## 데이터 흐름

```
data/raw/company/{subdir}/{file}.pdf
    │
    ▼
pdf_parser.py       Gemini File API → Markdown 변환
    │
    ▼
chunker.py          헤더 기준 1차 분리 / 표·문단 단위 2차 분리
    │
    ▼
output_writer.py    .md / .json / vector_chunks.json 저장
    │
    ▼
vector_db.py        paraphrase-multilingual-MiniLM-L12-v2 임베딩 → ChromaDB 적재
    │
    ▼
rag_chain.py        질문 → 검색 → threshold 필터 → Gemini 답변 → 출처 반환
```

---

## 실행 방법

```bash
# 의존성 설치
uv sync

# 1단계: 회사 문서 전체 파싱
cd src && uv run python company_ingest.py

# 2단계: ChromaDB 적재
cd src && uv run python company_vectordb.py

# 3단계: RAG 질의 테스트(단일 질문)
cd src && uv run python rag_chain.py "회사 주요 특허는 무엇인가요?"

# 전체 테스트 케이스 실행
cd src && uv run python rag_chain.py --test
```

---

## 환경변수 (.env)

```
GEMINI_API_KEY=...              # Gemini File API 키 (필수)
EMBEDDING_PROVIDER=local        # local | openai
CHROMA_COLLECTION_NAME=ninewatt_company
GEMINI_RAG_MODEL=gemini-2.0-flash

# 단일 PDF 파싱 시 (pdf_parser.py 직접 실행 시)
DOC_SOURCE_TYPE=company
DEFAULT_PDF_SUBDIR=certification_list_1
DEFAULT_PDF_NAME=sample.pdf
```

---

## 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `google-genai` | Gemini File API (PDF 파싱) |
| `chromadb` | 벡터 저장소 |
| `sentence-transformers` | 로컬 임베딩 모델 |
| `openai` | OpenAI 임베딩 (선택적) |
| `pymupdf` | PDF 이미지 추출 |
| `python-dotenv` | 환경변수 로드 |

---

## 문서

- [아키텍처 설계](docs/architecture.md)
- [파이프라인 흐름](docs/pipeline.md)
- [모듈별 상세 설명](docs/modules/)
