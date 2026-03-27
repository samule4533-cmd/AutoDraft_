# company_vectordb.py

파싱된 회사 문서 전체를 ChromaDB와 parent_index.json에 일괄 적재하는 배치 로딩 스크립트.

## 개요

`company_vectordb.py`는 `data/processed/parsing_result_company/` 하위의 모든 파싱 결과를 수집해 한 번에 ChromaDB와 parent_index.json에 적재한다. `company_ingest.py`가 개별 파일 단위 스트리밍 적재라면, 이 스크립트는 전체 일괄 배치 적재다.

> **주요 사용 시점**: 임베딩 모델 전환, ChromaDB 초기화, 전체 재적재가 필요할 때.

---

## 사용 방법

```bash
# 기본 실행 (기존 컬렉션에 upsert)
cd src && uv run python company_vectordb.py

# 컬렉션 초기화 후 전체 재적재 (임베딩 모델 전환 시)
CHROMA_RESET=true uv run python company_vectordb.py
```

---

## 설정 (`.env`)

| 변수 | 기본값 | 의미 |
|------|--------|------|
| `EMBEDDING_PROVIDER` | `"local"` | `"local"` \| `"openai"` |
| `CHROMA_COLLECTION_NAME` | `"ninewatt_company"` | ChromaDB 컬렉션명 |
| `CHROMA_RESET` | `"false"` | `"true"` 시 컬렉션 삭제 후 재생성 |

---

## 주요 함수

### `collect_all_chunks() -> list[dict]`
`parse_report.json` + `*.md` 파일 쌍을 수집해 청킹 후 전체 청크 반환.

```
COMPANY_OUTPUT_ROOT/**/parse_report.json 탐색
  ↓
각 report에서 document_id, model, source_file 추출
  ↓
같은 디렉터리의 *.md 읽기
  ↓
split_markdown_into_chunks() 실행
  ↓
전체 청크 합산 반환
```

> `vector_chunks.json`은 더 이상 사용하지 않는다. `*.md` 존재 여부가 파싱 완료의 유일한 기준.

---

### `upsert_all(force_reset=False) -> None`
메인 적재 함수.

```
collect_all_chunks()
  ↓
청크 타입별 라우팅:
  ├── section / child → ChromaDB upsert
  └── parent         → parent_store.save_index()
  ↓
(CHROMA_RESET=true 또는 force_reset=True) → reset_collection()
  ↓
upsert_chunks_to_chroma()
  ↓
sanity check: 컬렉션 총 청크 수 출력
```

---

## 설계 포인트

### 1. 항상 재청킹
`*.md`에서 매번 청킹을 재실행한다. `vector_chunks.json`을 그대로 사용하지 않아 청킹 로직 변경이 즉시 반영된다.

### 2. 원자적 parent_index 교체
`save_index()`로 `parent_index.json` 전체를 교체. 배치 적재 시 일관성 보장.

### 3. 컬렉션 리셋 옵션
임베딩 모델 전환(차원 변경) 시 기존 컬렉션과 차원이 맞지 않으면 에러. `CHROMA_RESET=true`로 컬렉션 삭제 후 재생성.

---

## company_ingest.py와의 차이

| 항목 | `company_ingest.py` | `company_vectordb.py` |
|------|--------------------|-----------------------|
| 목적 | 신규 파일 파싱 + 적재 | 기존 파싱 결과 전체 일괄 적재 |
| Gemini 호출 | 있음 (PDF 파싱) | 없음 |
| 스킵 로직 | 있음 (*.md 존재 시) | 없음 (전체 재적재) |
| parent_index | merge (기존 보존) | save (전체 교체) |
| 사용 시점 | 일상적 신규 문서 추가 | 초기 구축, 전체 재적재 |

---

## 의존성

### 내부
- `chunker.split_markdown_into_chunks`
- `vector_db.upsert_chunks_to_chroma`, `vector_db.reset_collection`, `vector_db.get_chroma_dir`
- `parent_store.save_index`

### 외부
- `chromadb`, `json`, `logging`, `pathlib`, `python-dotenv`

---
최종 수정: 2026-03-27
관련 파일: `src/company_vectordb.py`
---
