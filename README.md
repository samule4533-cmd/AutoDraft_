# AutoDraft(Chatbot)

> 기업 문서를 대상으로 **PDF 파싱 → 구조적 청킹 → 벡터 적재 → 출처 기반 질의응답 → HTTP API → 챗봇 UI**까지 연결한 사내 문서 RAG 시스템

## 프로젝트 개요

AutoDraft(Chatbot)는 특허, 인증서, 회사소개서, 기술자료 등 한국어 기업 문서를 구조적으로 파싱하고,
검색 가능한 청크 단위로 변환한 뒤, 벡터 DB에 적재하여 문서 기반 질의응답을 수행하는 RAG 시스템입니다.

이 프로젝트는 단순 텍스트 추출이 아니라, 문서의 **제목 구조, 표, 수치, 번호, 식별 정보**를 최대한 보존하면서
후속 검색과 답변 생성에 적합한 형태로 문서를 정제하고, 실제 사용자가 챗봇 UI를 통해 질의응답을 수행할 수 있는 완성된 서비스 구조를 목표로 합니다.

---

## 목표

- 한국어 기업 문서를 구조적으로 Markdown으로 변환
- 문서 계층, 표, 숫자, 식별자 정보를 최대한 보존
- RAG에 적합한 청크 단위로 분할
- ChromaDB 기반 내부 문서 지식 베이스 구축
- 검색된 문서만 근거로 자연어 질의응답 수행
- threshold 및 fallback 정책으로 환각 가능성 감소
- FastAPI + React 기반 챗봇 UI로 실제 사용 가능한 서비스 구성

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

### 벡터 저장소 적재 및 검색
- ChromaDB 기반 벡터 저장소 구성
- 로컬 임베딩 / OpenAI 임베딩 전환 가능 구조
- `chunk_id` 기준 upsert 지원
- 코사인 유사도 기반 검색

### Phase 2 RAG 체인
- Query Reformulation: 지시어(`그건`, `거기` 등) 감지 시 조건부 LLM 재작성
- chat_history 반영: 최근 5턴 슬라이딩 윈도우로 대화 맥락 유지
- 벡터 검색 → threshold 필터(0.55) → fallback → Gemini 답변 생성
- 출처(헤더, 파일명) 및 검색 통계(retrieved / passed / top_distance) 반환
- 관련 문서 없을 시 LLM 호출 없이 즉시 응답

### FastAPI HTTP 서버
- `POST /chat`: 질문 수신 → RAG 처리 → JSON 응답
- `GET /health`: 서버 상태 및 ChromaDB 청크 수 확인
- Pydantic 스키마 자동 검증 + Swagger 자동 문서화

### 챗봇 UI (React + TypeScript)
- 대화 기록 localStorage 저장/복원/삭제
- 사이드바 대화 목록 (토글, 스크롤, 검색, 날짜 표시)
- 채팅방 제목 인라인 편집
- 빈 상태: 입력창 중앙 배치 → 첫 질문 후 하단 이동
- 마크다운 렌더링 (볼드, 목록 등)
- 출처 태그 표시

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
│   ├── api.py                    # FastAPI HTTP 서버 (/chat, /health)
│   ├── rag_chain.py              # RAG 체인 코어 (검색 → 필터 → 생성 → 출처 반환)
│   ├── pdf_parser.py             # PDF → Markdown 변환 (Gemini File API)
│   ├── chunker.py                # Markdown → RAG 청크 분리
│   ├── vector_db.py              # ChromaDB 임베딩 적재 및 유사도 검색
│   ├── llm_api.py                # Gemini API 공통 래퍼 (싱글턴, JSON 복구)
│   ├── output_writer.py          # 파싱 결과 파일 저장 (.md / .json / vector_chunks.json)
│   ├── company_ingest.py         # 회사 문서 배치 파싱 파이프라인
│   ├── company_vectordb.py       # 파싱 결과 → ChromaDB 일괄 적재
│   └── image_parser.py           # 이미지 추출 + Gemini 캡션 생성 (현재 비활성화)
│
├── frontend/                     # React 챗봇 UI
│   ├── src/
│   │   ├── App.tsx               # 챗봇 UI 컴포넌트 (대화, 사이드바, 검색)
│   │   ├── App.css               # 챗봇 UI 스타일
│   │   ├── index.css             # 전역 CSS 변수 및 기반 스타일
│   │   └── main.tsx              # React 앱 진입점
│   ├── package.json
│   └── vite.config.ts
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
│   ├── modules/                  # 모듈별 상세 설명
│   │   ├── api.md
│   │   ├── rag_chain.md
│   │   ├── pdf_parser.md
│   │   ├── chunker.md
│   │   ├── vector_db.md
│   │   ├── llm_api.md
│   │   ├── output_writer.md
│   │   ├── company_ingest.md
│   │   └── company_vectordb.md
│   └── worklog/                  # 날짜별 작업 일지
│       ├── 2026_03_17.md
│       └── 2026_03_18.md
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
rag_chain.py        (Query Reformulation →) 검색 → threshold 필터 → Gemini 답변 → 출처 반환
    │
    ▼
api.py              POST /chat → RagResult JSON 응답
    │
    ▼
frontend/App.tsx    React 챗봇 UI → 사용자 질의응답
```

---

## 실행 방법

### 백엔드

```bash
# 의존성 설치
uv sync

# 1단계: 회사 문서 전체 파싱
cd src && uv run python company_ingest.py

# 2단계: ChromaDB 적재
cd src && uv run python company_vectordb.py

# 3단계: API 서버 실행
cd src && uv run python api.py
# → http://localhost:8000 에서 서버 실행
# → http://localhost:8000/docs 에서 Swagger UI 확인
```

### 프론트엔드

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173 에서 챗봇 UI 실행
```

### RAG 직접 테스트 (CLI)

```bash
# 단일 질문
cd src && uv run python rag_chain.py "회사 주요 특허는 무엇인가요?"

# 전체 테스트 케이스 실행
cd src && uv run python rag_chain.py --test
```

---

## 환경변수 (.env)

```
GEMINI_API_KEY=...              # Gemini API 키 (필수)
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
| `google-genai` | Gemini File API (PDF 파싱, RAG 답변 생성) |
| `fastapi` | HTTP API 서버 |
| `uvicorn` | ASGI 서버 |
| `chromadb` | 벡터 저장소 |
| `sentence-transformers` | 로컬 임베딩 모델 |
| `pydantic` | 입출력 스키마 검증 |
| `openai` | OpenAI 임베딩 (선택적) |
| `pymupdf` | PDF 이미지 추출 |
| `python-dotenv` | 환경변수 로드 |

---

## 개발 로드맵

### Phase 1 — MVP 코어 ✅ 완료
- PDF 파싱, 청킹, 벡터 적재, RAG 체인 코어

### Phase 2 — 대화형 UX ✅ 완료
- Chat History, Query Reformulation, FastAPI 서버, React 챗봇 UI

### Phase 3 — 검색 품질 고도화 🔲 예정
- 하이브리드 검색 (벡터 + BM25)
- Small-to-Big Retrieval
- 메타데이터 사전 필터링
- PDF 업로드 API

---

## 문서

- [아키텍처 설계](docs/architecture.md)
- [파이프라인 흐름](docs/pipeline.md)
- [모듈별 상세 설명](docs/modules/)
- [작업 일지](docs/worklog/)
