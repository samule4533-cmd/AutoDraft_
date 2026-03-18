# Architecture Overview

사내 챗봇을 위한 **한국어 문서 기반 RAG 시스템**의 전체 아키텍처 문서.
PDF 문서를 신뢰 가능한 지식 자원으로 변환하고, 근거 기반 질의응답을 수행하는 것을 목표로 한다.

---

## 목차

- [시스템 개요](#시스템-개요)
- [모듈 구조](#모듈-구조)
- [모듈별 역할](#모듈별-역할)
- [RAG 체인 상세](#rag-체인-상세)
- [임베딩 전략](#임베딩-전략)
- [벡터 DB 설계](#벡터-db-설계)
- [핵심 설계 결정](#핵심-설계-결정)
- [환경 변수 참조](#환경-변수-참조)
- [개발 로드맵](#개발-로드맵)

---

## 시스템 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                         AutoDraft RAG                           │
│                                                                 │
│  [PDF 문서]                                                     │
│      │                                                          │
│      ▼  Ingestion Pipeline                                      │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌───────────┐  │
│  │pdf_parser│──▶│ chunker  │──▶│output_    │──▶│ vector_db │  │
│  │(Gemini)  │   │(헤더기반)│   │writer     │   │(ChromaDB) │  │
│  └──────────┘   └──────────┘   └───────────┘   └───────────┘  │
│                                                       │         │
│                                                       ▼         │
│  [사용자 질문] ──────────────────────────▶ rag_chain.py        │
│                                              │                  │
│                                              ▼                  │
│                                        [RagResult]             │
│                                    answer / citations           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 모듈 구조

```
AutoDraft_clean/
├── src/
│   ├── api.py               # FastAPI HTTP 서버 (Phase 2 — /chat, /health 엔드포인트)
│   ├── pdf_parser.py        # PDF → Markdown 변환 (Gemini File API)
│   ├── chunker.py           # Markdown → 청크 분리 (헤더 기반)
│   ├── vector_db.py         # ChromaDB 임베딩 적재 및 검색
│   ├── rag_chain.py         # RAG 오케스트레이션 (핵심 80%)
│   ├── llm_api.py           # Gemini 클라이언트 싱글턴
│   ├── output_writer.py     # 처리 결과 파일 저장
│   ├── company_ingest.py    # 일괄 PDF 파싱 파이프라인
│   ├── company_vectordb.py  # 일괄 ChromaDB 적재
│   └── image_parser.py      # 이미지 추출 + 캡션 (현재 비활성화)
│
├── data/
│   ├── raw/company/         # 입력 PDF 원본
│   ├── processed/           # 파싱 결과물 (.md, .json, vector_chunks.json)
│   └── vector_store/chroma/ # ChromaDB 영구 저장소
│
├── docs/                    # 문서
│   ├── architecture.md      # 이 파일 - 전체 아키텍처
│   └── pipeline.md          # 데이터 흐름 상세
│
├── pyproject.toml           # uv 의존성 관리
└── CLAUDE.md                # 프로젝트 명세 및 개발 방향
```

---

## 모듈별 역할

### `pdf_parser.py` — PDF → Markdown 변환

파이프라인의 진입점. PDF를 Gemini File API에 업로드해 Markdown으로 변환한다.

**핵심 동작:**
- 한글 파일명 업로드 오류 방지 → 임시 디렉토리에 `upload_input.pdf`로 복사 후 업로드
- `normalize_markdown_headings()`: Gemini가 `**제목**` 형태로 출력하는 볼드 헤더를 `##` 헤더로 보정
- 파싱 프롬프트: 표, 숫자, 날짜, 인증번호, 특허 청구항 등 정보 손실 방지에 집중

**출력:** 청크 리스트 (`list[dict]`) — ChromaDB 적재 형식

---

### `chunker.py` — 청크 분리

Markdown 텍스트를 검색 가능한 단위로 분리한다.

**2단계 분리 전략:**

```
Stage 1: 헤더 기반 분리
  # / ## / ### 헤더 기준으로 섹션 분리

Stage 2: 크기 기반 2차 분리 (섹션 > 1500자)
  ├─ 표(table): 헤더 행 + 구분자 보존하며 데이터 행 분리
  └─ 텍스트: 한국어 문장 경계(다./임./음.) → 줄바꿈 → 1300자 제한
```

**청크 메타데이터:**

| 필드 | 값 | 설명 |
|------|-----|------|
| `chunk_type` | `section` / `table` / `paragraph_group` | 청크 유형 |
| `chunk_position` | `only` / `first` / `middle` / `last` | 섹션 내 위치 |
| `has_table` | `bool` | 표 포함 여부 |
| `header` | `string` | 소속 헤더 텍스트 |
| `document_id` | `string` | 원본 PDF 식별자 |
| `source_file` | `string` | 원본 파일 경로 |

---

### `vector_db.py` — 임베딩 및 검색

ChromaDB 기반 벡터 저장소 관리.

**임베딩 옵션:**

| Provider | 모델 | 차원 | 비용 | 특징 |
|----------|------|------|------|------|
| `local` | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 무료 | 오프라인, 한국어 지원 |
| `openai` | `text-embedding-3-small` | 1536 | 유료 | 더 높은 한국어 정밀도 |

> `EMBEDDING_PROVIDER`를 변경할 때는 `CHROMA_RESET=true`로 컬렉션을 재생성해야 한다 (차원 불일치 방지).

**주요 함수:**
- `upsert_chunks_to_chroma()`: 배치 50개 단위 적재
- `query_collection()`: cosine 유사도 기반 검색, `where` 조건 필터 지원
- `_get_or_init_client()`: `PersistentClient` 캐싱 (챗봇 반복 요청 시 초기화 비용 절감)

---

### `api.py` — FastAPI HTTP 서버 (Phase 2)

RAG 파이프라인을 HTTP API로 노출하는 얇은 레이어.

- `POST /chat`: 사용자 질문 수신 → `rag_chain.ask()` 호출 → `RagResult` 반환
- `GET /health`: ChromaDB 접근 가능 여부 및 청크 수 확인
- 서버 시작 시 ChromaDB + 임베딩 모델 warm-up

> 상세 내용은 `docs/modules/api.md` 참조.

---

### `rag_chain.py` — RAG 오케스트레이션 ⭐

시스템의 핵심 모듈. (Query Reformulation →) 검색 → 필터링 → 프롬프트 조립 → 생성 → 인용 전 과정을 담당한다.

> 상세 내용은 [RAG 체인 상세](#rag-체인-상세) 섹션 참조.

---

### `llm_api.py` — Gemini 클라이언트

싱글턴 패턴으로 Gemini API 클라이언트를 관리한다.

**JSON 파싱 폴백 (`safe_json_load`):**
1. 직접 `json.loads()` 시도
2. 펜스 코드블록 ` ```json ... ``` ` 추출 후 파싱
3. 첫 `{`부터 마지막 `}` 범위 추출 후 파싱
4. 전부 실패 시 `{"ocr_text": raw_text}` 반환

---

### `company_ingest.py` — 일괄 파싱

`data/raw/company/` 하위 모든 PDF를 순차 파싱한다.

- `vector_chunks.json` 존재 시 스킵 (중복 파싱 방지)
- 실패 시 지수 백오프 재시도 (최대 3회, 5s → 10s → 20s)
- 실패 목록을 `failed_parse.log`에 기록

---

### `company_vectordb.py` — 일괄 적재

파싱 결과물(`vector_chunks.json`)을 ChromaDB에 일괄 업서트한다.

---

## RAG 체인 상세

```
사용자 질문
    │
    ▼
┌─────────────────────────────────────────────────┐
│  ask(query, chat_history, filters)              │
│                                                 │
│  1. retrieve()                                  │
│     ChromaDB 유사도 검색 (top-N)               │
│                                                 │
│  2. filter_by_threshold()                       │
│     distance ≤ 0.55 인 청크만 통과             │
│                                                 │
│  3. [분기]                                      │
│     ├─ 통과 청크 = 0 → fallback 반환           │
│     └─ 통과 청크 > 0 → 계속                    │
│                                                 │
│  4. build_context_block()                       │
│     청크 본문 + 헤더 + 출처 조합               │
│     최대 6000자 초과 시 하위 랭크 청크 제거    │
│                                                 │
│  5. build_prompt()                              │
│     시스템 프롬프트 + [이전 대화] + 참고 문서  │
│     + 질문 조립                                 │
│                                                 │
│  6. generate_answer()                           │
│     Gemini LLM 호출 (temperature=0.0)          │
│     429 → 지수 백오프 재시도 (최대 3회)        │
│                                                 │
│  7. format_citations()                          │
│     (header, source_file) 기준 중복 제거       │
│                                                 │
│  ▼ RagResult 반환                               │
└─────────────────────────────────────────────────┘
```

**RagResult 구조 (Pydantic BaseModel — Phase 2):**

```python
class RagResult(BaseModel):
    answer: str                  # 최종 답변 또는 fallback 메시지
    citations: list[Citation]    # 인용 출처 목록
    used_query: str              # 실제 검색에 사용된 query (reformulation 결과 또는 원본)
    reformulated_query: str | None  # 재작성이 발생한 경우만 값 존재
    retrieved_count: int         # ChromaDB 검색 결과 수
    passed_threshold: int        # 임계값 통과 청크 수
    top_distance: float | None   # 가장 유사한 청크의 거리
    fallback: bool               # True = 정상 답변이 아님
    fallback_reason: str | None  # "no_docs" | "retrieval_error" | "llm_error"
```

**Fallback 3단계:**

| 원인 | 메시지 | `fallback_reason` |
|------|--------|-------------------|
| 관련 청크 없음 | "사내 문서에서 해당 내용을 찾을 수 없습니다." | `no_docs` |
| 검색 오류 | "문서 검색 중 오류가 발생했습니다." | `retrieval_error` |
| 생성 오류 | "답변 생성 중 오류가 발생했습니다." | `llm_error` |

**시스템 프롬프트 원칙:**
```
1. 반드시 [참고 문서]에 있는 내용에만 근거하여 답하라.
2. 문서에 없는 내용은 "해당 내용이 문서에 없습니다"라고 답하라.
3. 답변 내 사실마다 [출처: {헤더명}] 형태로 표기하라.
4. 표, 수치, 날짜는 원문 그대로 인용하라.
5. 한국어로 답하라.
```

---

## 임베딩 전략

### 로컬 임베딩 (`local`)

```
paraphrase-multilingual-MiniLM-L12-v2
  - 차원: 384
  - 특징: 50개 언어 지원, 오프라인, API 비용 없음
  - 적합: 프로토타입, 비용 민감 환경
```

### OpenAI 임베딩 (`openai`)

```
text-embedding-3-small
  - 차원: 1536
  - 특징: 높은 한국어 정밀도, 입력 2500자 제한 (토큰 버짓 내)
  - 적합: 프로덕션, 검색 품질 우선 환경
```

> 임베딩 모델 전환 시 반드시 `CHROMA_RESET=true` → 컬렉션 재생성 필요.

---

## 벡터 DB 설계

### ChromaDB 컬렉션 설정

```python
{
    "name": "ninewatt_company",           # CHROMA_COLLECTION_NAME
    "hnsw:space": "cosine",               # 거리 메트릭
    "persist_dir": "data/vector_store/chroma"
}
```

### Cosine Distance 해석

```
0.0  ─────────────────────── 1.0 ─────────── 2.0
│                             │
│ 완전 동일                   │ 직교 (무관)
│                             │
│ ← distance ≤ threshold 통과 → 차단 →
```

- threshold 현재값: `0.55`
- 낮을수록 엄격한 필터 (false negative 위험)
- 높을수록 느슨한 필터 (환각 위험)
- 도메인/임베딩 모델에 따라 튜닝 필요
- 0.55 결정 근거: MiniLM이 `【청구항 N】` 등 특수 괄호 표기를 0.50~0.55 구간에 분포시키며, 완전 무관한 질문은 0.65 이상으로 안전 마진 확보됨

### 메타데이터 필터링 (`where` 조건)

```python
# 특정 문서만 검색
query_collection(query, where={"document_id": "patent_001"})

# 표 포함 청크만 검색
query_collection(query, where={"has_table": True})

# doc_type 필터
query_collection(query, where={"doc_type": "certification"})
```

---

## 핵심 설계 결정

### 1. 헤더 기반 청킹

문서의 의미 경계(섹션)를 자연스럽게 보존. 단순 고정 길이 분리 대비 문맥 손실 최소화.
표는 헤더 행을 각 청크에 복사해 context 유지.

### 2. temperature = 0.0

팩트 기반 답변에 창의성 불필요. 결정론적 출력으로 디버깅 용이. 원문 재서술 방지.

### 3. 검색/생성 평가 분리

`RagResult.retrieved_count`(검색기 성능) vs `RagResult.passed_threshold`(필터 성능)를
분리 노출해 버그 원인(임계값? 프롬프트? 청킹?) 진단 가능.

### 4. req_id 트레이싱

각 `ask()` 호출에 UUID 기반 `req_id` 부여. 로그에서 요청별 전 과정 추적 가능.

### 5. Fallback 강제

threshold 통과 청크 = 0일 때 LLM 호출을 완전히 생략. 빈 컨텍스트로 LLM 호출 시
발생하는 환각을 시스템 레벨에서 차단.

---

## 환경 변수 참조

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GEMINI_API_KEY` | (필수) | Gemini API 인증 |
| `GEMINI_PDF_MODEL` | `gemini-2.0-flash` | PDF 파싱 모델 |
| `GEMINI_RAG_MODEL` | `gemini-2.5-flash` | RAG 답변 생성 모델 |
| `GEMINI_MAX_OUTPUT_TOKENS` | `40000` | PDF 파싱 최대 출력 토큰 |
| `DOC_SOURCE_TYPE` | `company` | `company` 또는 `notice` |
| `DEFAULT_PDF_SUBDIR` | (없음) | `data/raw/{type}/` 하위 서브디렉토리 |
| `DEFAULT_PDF_NAME` | `sample_company.pdf` | 단독 실행 시 대상 PDF |
| `EMBEDDING_PROVIDER` | `local` | `local` 또는 `openai` |
| `CHROMA_COLLECTION_NAME` | `ninewatt_company` | ChromaDB 컬렉션명 |
| `OPENAI_API_KEY` | (openai 시 필수) | OpenAI API 인증 |
| `CHROMA_RESET` | `false` | `true` 시 컬렉션 초기화 후 재생성 |

---

## 개발 로드맵

### Phase 1 — MVP 코어 ✅ 완료

- [x] ChromaDB 유사도 검색
- [x] Distance threshold 필터링
- [x] 관련 청크 없음 fallback
- [x] 프롬프트 조립 (본문 + 메타데이터)
- [x] Gemini LLM 호출 (temperature=0)
- [x] 출처 인용 (`chunk_id`, `header`, `source_file`)
- [x] Rate limit 지수 백오프 재시도

### Phase 2 — 대화형 UX ✅ 완료

- [x] Chat History 반영 — 최근 5턴 슬라이딩 윈도우 (`_trim_history`)
- [x] Query Reformulation — 지시어 감지 시 조건부 LLM 재작성 (`should_reformulate`, `reformulate_query`)
- [x] FastAPI HTTP 서버 — `POST /chat`, `GET /health` (`api.py`)
- [x] Pydantic BaseModel 전환 — FastAPI JSON 직렬화 자동화
- [x] `used_query` / `reformulated_query` 디버깅 필드 추가
- [x] React 챗봇 프론트엔드 — 대화 기록 저장/검색, 사이드바 UI (`frontend/`)

### Phase 3 — 검색 품질 고도화 🔲 검토 예정

- [ ] 하이브리드 검색 (벡터 + BM25 키워드)
- [ ] Small-to-Big Retrieval (작은 청크 검색 → 큰 문맥 전달)
- [ ] 메타데이터 사전 필터링 (질문에서 연도/부서 추출 → `where` 조건)
- [ ] 이미지 캡션 RAG 통합 (`image_parser.py`)
