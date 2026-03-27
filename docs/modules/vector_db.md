# vector_db.py

ChromaDB 클라이언트 관리, 임베딩 함수 선택, 청크 upsert, 유사도 검색을 담당하는 벡터 저장소 레이어.

## 개요

`vector_db.py`는 임베딩-저장-검색 전 과정을 안정적으로 다루는 공통 벡터 인프라 계층이다. 상위 모듈(`rag_chain`, `company_vectordb`, `api`)은 이 파일을 통해서만 ChromaDB에 접근한다.

---

## 주요 함수

### `get_chroma_dir() -> Path`
`PROJECT_ROOT/data/vector_store/chroma` 반환.

---

### `get_embedding_function(provider, model) -> Any`
임베딩 제공자 라우터.

| `provider` | 반환 |
|-----------|------|
| `"local"` | `SentenceTransformerEmbeddingFunction` (MiniLM 다국어 모델) |
| `"openai"` | `_OpenAIEmbeddingFunction` (text-embedding-3-small) |

---

### `_get_or_init_client(persist_dir) -> Any`
ChromaDB 클라이언트 싱글턴 반환.

- `CHROMA_HOST` 환경변수 있으면 → `HttpClient` (Docker ChromaDB)
- 없으면 → `PersistentClient` (로컬 파일)
- 인스턴스를 `_chroma_client_cache`에 캐시 → 같은 경로로 중복 생성 방지 (파일 lock 충돌 예방)

---

### `get_or_create_collection(persist_dir, collection_name, embedding_provider, embedding_model) -> Any`
컬렉션이 없으면 생성, 있으면 기존 반환. 코사인 거리(`hnsw:space=cosine`) 설정.

---

### `reset_collection(collection_name, persist_dir, embedding_provider, embedding_model) -> Any`
컬렉션 삭제 후 재생성. 임베딩 모델 전환 시 (차원 불일치 방지) 필요.

> `CHROMA_RESET=true` 환경변수 또는 `force_reset=True` 인자로 트리거.

---

### `prepare_chroma_items(chunks, default_doc_type) -> dict`
청크 리스트 → ChromaDB upsert 입력 형식 변환.

- 빈 텍스트 청크 제거
- `doc_type` 기본값 보정
- `clean_metadata_for_chroma()`: `str/int/float/bool` 유지, 복합 타입 문자열 변환
- 반환: `{"ids": [...], "documents": [...], "metadatas": [...]}`

---

### `upsert_chunks_to_chroma(chunks, collection_name, persist_dir, batch_size, embedding_provider, embedding_model, default_doc_type)`
배치 upsert. `batch_size=50` 기본값. OpenAI 임베딩 실패 시 3회 재시도 (10s/20s 백오프).

---

### `query_collection(query_text, collection_name, persist_dir, embedding_provider, embedding_model, n_results, where) -> dict`
유사도 검색. `where` 파라미터로 ChromaDB 메타데이터 필터 지원.

---

### `_OpenAIEmbeddingFunction._embed(input)`
OpenAI API 호출 내부 메서드.

- 2500자 초과 청크 자동 truncate (token limit 대응)
- 실패 시 최대 3회 재시도 (10s, 20s 백오프)
- 모든 재시도 실패 시 raise

---

## 설계 포인트

### 1. 클라이언트 캐싱
같은 `persist_dir`에 대해 클라이언트를 한 번만 생성. 여러 모듈이 `get_or_create_collection()`을 호출해도 중복 파일 lock 없음.

### 2. HTTP vs 로컬 자동 전환
`CHROMA_HOST` 환경변수 유무로 Docker 배포 / 로컬 개발 환경을 자동 구분.

### 3. 코사인 거리
HNSW `space=cosine` 설정. 청크 길이가 다양해도 정규화된 유사도 비교 가능.

### 4. 임베딩 모델 전환
`local(384차원)` → `openai(1536차원)` 등 차원이 달라지면 컬렉션 삭제 후 재생성 필요 → `reset_collection()` 또는 `CHROMA_RESET=true`.

---

## 의존성

### 외부 라이브러리
- `chromadb`
- `sentence_transformers`
- `openai`
- `python-dotenv`, `logging`, `pathlib`, `os`

---
최종 수정: 2026-03-27
관련 파일: `src/vector_db.py`
---
