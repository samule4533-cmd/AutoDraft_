# api.py
2026-03-18;
FastAPI 기반 HTTP 서버 진입점. `rag_chain.ask()`를 외부에 노출하는 얇은 레이어.

## 개요
`api.py`는 RAG 파이프라인을 HTTP API로 노출하는 FastAPI 애플리케이션이다. 비즈니스 로직은 `rag_chain.py`에서 처리하고, 이 파일은 입출력 변환과 서버 설정만 담당한다.

---

## 역할

- HTTP 요청 수신 및 Pydantic 스키마 검증
- `rag_chain.ask()` 호출 및 결과 반환
- 서버 시작 시 ChromaDB + 임베딩 모델 warm-up
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

#### 목적
배포 후 서버가 실제로 쿼리를 처리할 준비가 됐는지 즉시 확인할 수 있다. ChromaDB 접근 불가 시 503 반환.

---

### `POST /chat`
사용자 질문을 받아 RAG 답변을 반환한다.

#### 요청 스키마 (`ChatRequest`)

| 필드 | 타입 | 설명 |
|------|------|------|
| `query` | `str` (1~1000자) | 사용자 질문 |
| `chat_history` | `list[ChatMessage] \| None` | 이전 대화 맥락 (없으면 단발성 처리) |
| `filters` | `dict \| None` | ChromaDB where 조건 (Phase 3 메타데이터 필터링용) |

#### `ChatMessage` 스키마

| 필드 | 타입 | 제약 |
|------|------|------|
| `role` | `"user" \| "assistant"` | 그 외 값은 422 반환 |
| `content` | `str` | min_length=1 |

#### 응답 스키마 (`RagResult`)
`rag_chain.RagResult` Pydantic 모델 그대로 반환. 상세 필드는 `rag_chain.md` 참조.

#### `fallback_reason` 의미
| 값 | 원인 |
|----|------|
| `"no_docs"` | threshold 통과 문서 없음 |
| `"retrieval_error"` | ChromaDB 검색 실패 |
| `"llm_error"` | Gemini 호출 실패 |

---

## Lifespan (서버 시작/종료)

```python
@asynccontextmanager
async def lifespan(app):
    # 시작: ChromaDB 클라이언트 + 임베딩 모델 warm-up
    get_or_create_collection(...)
    yield
    # 종료: 로그 기록
```

#### warm-up 이유
SentenceTransformer 모델 로딩이 수 초 걸린다. 첫 요청에서 지연이 발생하지 않도록 서버 시작 시점에 미리 초기화한다. warm-up 실패 시에도 서버는 계속 기동됨 (데이터 미적재 환경 대응).

---

## 설계 포인트

### 1. 얇은 레이어 원칙
비즈니스 로직을 두지 않는다. 입력 검증은 Pydantic에, 처리 로직은 `rag_chain.ask()`에 위임한다.

### 2. 입력 방어
- `query` max_length=1000: 지나치게 긴 입력이 Reformulation + Retrieve 두 번의 LLM 호출을 유발하는 것을 방어
- `role` Literal 제약: 잘못된 role 값을 FastAPI가 422로 자동 차단

### 3. CORS 설정
개발 단계에서는 `allow_origins=["*"]`로 전체 허용. 운영 배포 시 실제 프론트엔드 도메인만 지정해야 한다.

### 4. 예외 처리
`ask()` 내부에서 이미 fallback 처리를 하므로 대부분의 오류는 `RagResult.fallback=True`로 반환된다. 예상치 못한 예외만 500으로 반환한다.

---

## 실행 방법

```bash
# 개발 서버 (코드 변경 시 자동 재시작)
cd src && uv run python api.py

# 또는 uvicorn 직접 실행
cd src && uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 의존성

### 내부 의존성
- `rag_chain.ask`, `rag_chain.RAG_CONFIG`, `rag_chain.RagResult`
- `vector_db.get_or_create_collection`

### 외부 라이브러리
- `fastapi`
- `uvicorn`
- `pydantic`

---
최종 수정: 2026-03-18
관련 파일: 'src/api.py'
---
