# parent_store.md

Parent-Child 청킹 구조의 parent 청크 인덱스를 관리하는 모듈. child 청크 검색 결과를 parent 도입부 + 인접 형제로 확장하는 컨텍스트 확장 레이어를 제공한다.

## 개요

`chunker.py`는 긴 섹션을 parent(도입부 요약) + child(세부 내용) 구조로 분리한다. ChromaDB에는 child 청크만 적재되어 검색된다. `parent_store.py`는 검색된 child 청크를 parent 도입부 + 인접 형제로 확장해 더 풍부한 컨텍스트를 LLM에 제공한다.

- `data/processed/parent_index.json` 에 영구 저장 (원자적 쓰기)
- 서버 시작 시 인메모리 로드, 이후 O(1) 조회
- 배치 인제스트(`save_index`)와 단건 인제스트(`merge_parents`) 분리

---

## 저장 구조

### `parent_index.json`

```json
{
  "doc_id_p0": {
    "parent_chunk_id": "doc_id_p0",
    "document_id": "doc_id",
    "header": "## 청구항 1",
    "source_file": "특허.pdf",
    "intro_text": "본 발명은 건물 에너지 모델링 자동화 시스템에 관한 것으로...",
    "children": [
      {
        "chunk_id": "doc_id_p0_ch0",
        "text": "...",
        "position": 0
      },
      ...
    ]
  }
}
```

### 인메모리 인덱스 (3개 병렬 딕셔너리)

| 딕셔너리 | 키 → 값 | 용도 |
|---------|---------|------|
| `_parent_map` | `parent_chunk_id → entry` | parent 조회 |
| `_child_to_parent` | `child_id → parent_chunk_id` | child → parent 역조회 |
| `_child_order` | `child_id → 위치 인덱스` | 인접 형제 계산 |

---

## 청크 ID 체계

| 타입 | 형식 | 예시 |
|------|------|------|
| Parent | `{document_id}_p{N}` | `patent_001_p2` |
| Child | `{document_id}_p{N}_ch{i}` | `patent_001_p2_ch0` |

---

## 주요 함수

### `load(path=None) -> None`
`parent_index.json` 로드 → `_parent_map`, `_child_to_parent`, `_child_order` 빌드. 서버 시작 시 `api.py` lifespan에서 호출.

---

### `get_parent(parent_chunk_id) -> dict | None`
parent_chunk_id로 parent entry 반환. 없으면 None.

---

### `get_adjacent_child_ids(child_id, window=1) -> list[str]`
child 기준 ±window 인접 형제 ID 반환. window=1 이면 `[이전, 현재, 다음]` 최대 3개.

`rag_chain._expand_context()`에서 컨텍스트 확장에 사용:
```
child_hit → get_adjacent_child_ids() → [prev, hit, next]
         → parent intro_text + 전후 child text 합산
```

---

### `get_child_text(child_id) -> str | None`
child 청크 텍스트를 parent_store에서 직접 조회.

---

### `save_index(parent_chunks, child_map, path=None) -> None`
전체 parent_index.json 교체 (배치 인제스트용). `company_vectordb.upsert_all()`에서 호출.

---

### `merge_parents(parent_chunks, child_map, path=None) -> None`
기존 parent_index.json에 신규 parent 병합 (단건 인제스트용). `api.py POST /ingest`에서 호출.

---

### `remove_by_document(document_id, path=None) -> None`
`document_id` prefix를 가진 parent entry 전체 삭제. `api.py DELETE /ingest/{file_id}`에서 호출.

---

## 설계 포인트

### 1. 원자적 파일 쓰기
임시 파일 생성 후 rename. 쓰기 도중 서버 크래시가 나도 기존 파일이 보존된다.

### 2. 스레드 세이프 읽기
lock으로 레퍼런스만 복사 후 즉시 해제 → 이후 읽기는 스냅샷으로 진행 (성능 유지).

### 3. Graceful Degradation
parent_store가 비어있거나 child_id를 찾지 못하면 `_expand_context()`가 원본 child 청크를 그대로 반환. 하드 실패 없음.

### 4. 배치 vs 단건 분리
- `save_index`: 전체 교체 (배치 처리, 회사 전체 재인제스트 시)
- `merge_parents`: 병합 (단건 Drive 업로드 시, 기존 데이터 유지)

---

## 의존성

### 외부 라이브러리
- `json`, `pathlib`, `threading`, `logging`

---
최종 수정: 2026-03-27
관련 파일: `src/parent_store.py`
---
