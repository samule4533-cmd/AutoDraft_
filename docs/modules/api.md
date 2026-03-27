# api.py

FastAPI 기반 HTTP 서버 진입점. RAG 파이프라인과 문서 인제스트를 HTTP API로 노출하는 얇은 레이어.

## 개요

`api.py`는 RAG 파이프라인을 HTTP API로 노출하는 FastAPI 애플리케이션이다. 비즈니스 로직은 `rag_chain.py`와 각 서비스 모듈에서 처리하고, 이 파일은 입출력 변환과 서버 설정만 담당한다.

---

## 역할

- HTTP 요청 수신 및 Pydantic 스키마 검증
- `rag_chain.ask()` 호출 및 결과 반환
- Google Drive 파일 업로드 → 파싱 → ChromaDB 적재 처리
- 서버 시작 시 ChromaDB, BM25 인덱스, parent_store warm-up
- CORS 설정 (개발 단계 전체 허용)
- 서버 상태 확인 엔드포인트 제공

---

## 엔드포인트

### `GET /health`
서버 상태 및 ChromaDB 접근 가능 여부를 확인한다.

#### 반환 예시
```json
{
  "status": "ok",
  "collection": "ninewatt_company",
  "chunk_count": 412
}
```

---

### `POST /chat`
사용자 질문을 받아 RAG 답변을 반환한다.

#### 요청 스키마 (`ChatRequest`)

| 필드 | 타입 | 설명 |
|------|------|------|
| `query` | `str` (1~1000자) | 사용자 질문 |
| `chat_history` | `list[ChatMessage] \| None` | 이전 대화 맥락 |
| `filters` | `dict \| None` | ChromaDB where 조건 (메타데이터 필터) |

#### `ChatMessage` 스키마

| 필드 | 타입 | 제약 |
|------|------|------|
| `role` | `"user" \| "assistant"` | 그 외 값은 422 반환 |
| `content` | `str` | min_length=1 |

#### 응답 스키마
`rag_chain.RagResult` Pydantic 모델 그대로 반환. 상세 필드는 `rag_chain.md` 참조.

#### `fallback_reason` 값

| 값 | 원인 |
|----|------|
| `"no_docs"` | threshold 통과 문서 없음 |
| `"retrieval_error"` | ChromaDB 검색 실패 |
| `"llm_error"` | Gemini 호출 실패 |

---

### `POST /ingest`
Google Drive에서 받은 파일을 파싱하고 ChromaDB에 적재한다.

#### 요청 (multipart/form-data)

| 필드 | 타입 | 설명 |
|------|------|------|
| `file` | `UploadFile` | PDF 파일 |
| `file_id` | `str` | Drive 파일 ID (idempotent key) |
| `file_name` | `str` | 파일명 (메타데이터에 저장) |

#### 처리 흐름
1. 같은 `file_id` 기존 청크 먼저 삭제 (idempotent)
2. PDF → Markdown 파싱 (`pdf_parser`)
3. 청킹 (`chunker`)
4. ChromaDB upsert + parent_index 병합
5. BM25 인덱스 재빌드

#### 반환 예시
```json
{
  "chunk_count": 42,
  "file_id": "abc123",
  "file_name": "특허문서.pdf"
}
```

---

### `PATCH /ingest/{file_id}`
적재된 파일의 파일명 메타데이터를 변경한다.

#### 요청 바디 (`RenameRequest`)

| 필드 | 타입 | 설명 |
|------|------|------|
| `file_name` | `str` | 변경할 파일명 |

#### 반환 예시
```json
{
  "updated_chunks": 42,
  "file_id": "abc123",
  "file_name": "새이름.pdf"
}
```

변경 후 BM25 인덱스 자동 재빌드.

---

### `DELETE /ingest/{file_id}`
적재된 파일의 청크와 parent_index 항목을 삭제한다.

#### 반환 예시
```json
{
  "deleted": true,
  "file_id": "abc123"
}
```

삭제 후 BM25 인덱스 자동 재빌드.

---

### `GET /summaries`
ChromaDB에 적재된 문서의 AI 요약 목록을 반환한다.

#### 반환 예시
```json
[
  {"index": 1, "filename": "특허_A.pdf", "summary": "..."},
  {"index": 2, "filename": "회사소개서.pdf", "summary": "..."}
]
```

캐시(`summaries.json`) 기반. 신규 파일 있을 때만 Gemini 호출.

---

## Lifespan (서버 시작/종료)

```python
@asynccontextmanager
async def lifespan(app):
    # 시작: ChromaDB, BM25 인덱스, parent_store, (reranker) warm-up
    collection = get_or_create_collection(...)
    bm25_retriever.build_index(collection)
    parent_store.load()
    # reranker.load_model()  # GPU 서버 준비 시 활성화
    yield
    # 종료: 로그 기록
```

warm-up 실패 시에도 서버는 계속 기동된다 (데이터 미적재 환경 대응).

---

## 설계 포인트

### 1. 얇은 레이어 원칙
비즈니스 로직을 두지 않는다. 입력 검증은 Pydantic에, 처리 로직은 각 서비스 모듈에 위임한다.

### 2. Idempotent ingest
같은 `file_id`로 재업로드 시 기존 청크를 먼저 삭제 후 재적재한다. Drive 파일이 수정되어도 안전하게 반영된다.

### 3. BM25 자동 갱신
POST/PATCH/DELETE /ingest 후 `bm25_retriever.rebuild_index()`를 항상 호출해 인메모리 인덱스와 ChromaDB 상태를 동기화한다.

### 4. CORS 설정
개발 단계에서는 `allow_origins=["*"]`로 전체 허용. 운영 배포 시 실제 프론트엔드 도메인만 지정해야 한다.

### 5. 입력 방어
- `query` max_length=1000: 긴 입력이 LLM 호출을 복수로 유발하는 것을 방어
- `role` Literal 제약: 잘못된 role 값을 FastAPI가 422로 자동 차단

---

## 실행 방법

```bash
# 개발 서버 (코드 변경 시 자동 재시작)
cd src && uv run python api.py

# 특정 파일 변경 감지 제외 (test_suite.py 수정 시 서버 재시작 방지)
cd src && uv run uvicorn api:app --reload --reload-exclude "test_suite.py"
```

---

## 의존성

### 내부 의존성
- `rag_chain.ask`, `rag_chain.RAG_CONFIG`, `rag_chain.RagResult`
- `vector_db.get_or_create_collection`
- `bm25_retriever.build_index`, `bm25_retriever.rebuild_index`
- `parent_store.load`, `parent_store.merge_parents`, `parent_store.remove_by_document`
- `summary_service.get_summaries`
- `pdf_parser.parse_single_pdf`
- `chunker.split_markdown_into_chunks`

### 외부 라이브러리
- `fastapi`, `uvicorn`, `pydantic`

---
최종 수정: 2026-03-27
관련 파일: `src/api.py`
---
