# summary_service.py
2026-03-20
사내 문서 전체 AI 요약 생성 및 캐시 관리 모듈.

## 개요
`summary_service.py`는 ChromaDB에 적재된 사내 문서를 Gemini가 자동 요약하고, 결과를 JSON 캐시로 관리한다. `/summaries` 엔드포인트의 비즈니스 로직을 담당하며, `api.py`에서 호출된다.

---

## 역할

- ChromaDB에서 파일별 청크 텍스트 수집
- Gemini 1회 호출로 전체 파일 요약 생성 (1M 토큰 컨텍스트 활용)
- 결과를 `data/processed/summaries.json`에 캐시
- 재호출 시 새 파일만 추가 생성, 삭제된 파일은 캐시에서 제거

---

## 캐시 전략

| 상황 | 동작 |
|------|------|
| 변화 없음 | Gemini 호출 0회 — 캐시 즉시 반환 |
| 새 파일 추가 | 신규 파일만 Gemini 호출 → 기존 캐시에 병합 |
| 파일 삭제 | 캐시에서 해당 파일 항목 제거 |
| 파일 내용 변경 | **미지원** — 파일명이 같으면 기존 캐시 그대로 사용 |

버튼을 반복 클릭해도 Gemini를 재호출하지 않는다. 파일이 추가되거나 삭제될 때만 Gemini를 호출한다.

> **개선 예정**: 캐시에 청크 수를 함께 저장해 청크 수가 달라지면 재요약을 트리거하는 방식으로 내용 변경 감지를 보완할 예정.

---

## 주요 함수

### `get_summaries() → list[dict]`
메인 진입점. 전체 문서 요약 목록을 반환한다.

**반환 형식**
```python
[
    {"index": 1, "filename": "특허_A.pdf", "summary": "요약 내용..."},
    {"index": 2, "filename": "회사소개서.pdf", "summary": "요약 내용..."},
]
```

**내부 흐름**
1. `_get_file_chunks()` — ChromaDB에서 현재 파일 목록 수집
2. `_load_cache()` — 기존 캐시 로드
3. 신규/삭제 파일 비교
4. 신규 파일 있으면 `_generate_summaries()` 호출
5. 캐시 업데이트 후 목록 반환

---

### `_get_file_chunks() → dict[str, list[str]]`
ChromaDB에서 전체 청크를 수집해 `{source_file: [청크 텍스트, ...]}` 형태로 반환한다.

---

### `_generate_summaries(file_chunks, target_files) → dict[str, str]`
Gemini를 1회 호출해 `target_files` 전체를 요약하고 `{filename: summary}` dict로 반환한다.

- 파일당 최대 `_MAX_CHUNKS_PER_FILE = 30`개 청크만 사용 (토큰 한도 안전 마진)
- `temperature=0.0` — 출력 일관성 확보
- JSON 파싱 실패 시 해당 파일은 `"요약 생성에 실패했습니다."` 처리

**프롬프트 구조**
```
[파일: A.pdf]
청크1
청크2
...

---

[파일: B.pdf]
청크1
...
```

---

### `_load_cache() → dict[str, str]`
`data/processed/summaries.json`에서 기존 요약 캐시를 로드한다. 파일이 없거나 파싱 실패 시 빈 dict 반환.

### `_save_cache(summaries) → None`
요약 캐시를 JSON 파일로 저장한다. 디렉토리가 없으면 자동 생성.

---

## 캐시 파일 구조

**경로**: `data/processed/summaries.json`

```json
{
  "summaries": {
    "특허_A.pdf": "이 문서는 건물 에너지 모델링 자동화 시스템에 관한 특허로...",
    "회사소개서.pdf": "9watt는 에너지 분야 전문 기업으로..."
  }
}
```

---

## 설계 포인트

### 1. Gemini 1회 호출 전략
문서가 8개일 때 개별 호출(8회) 대신 1회 일괄 호출로 비용과 지연을 줄인다. Gemini 2.0 Flash의 1M 토큰 컨텍스트를 활용한다.

### 2. 파일명 기준 캐시 키
캐시 키는 ChromaDB `source_file` 메타데이터 값을 그대로 사용한다. Gemini 프롬프트에도 동일한 파일명을 전달하고 `파일 이름은 그대로 사용하라`고 지시한다.

> **주의**: Gemini가 드물게 파일명을 변형(언더스코어→공백 등)할 경우 캐시 키 불일치가 발생할 수 있다. `temperature=0.0`과 명시적 지시로 방어하고 있으나, 실제로 증상이 발생하면 응답 파일명 정규화 로직을 추가한다.

### 3. 코드펜스 파싱 방어
Gemini 응답에 ` ```json ` 코드펜스가 포함될 경우 자동으로 제거 후 JSON 파싱한다.

---

## 의존성

### 내부 의존성
- `src.llm_api.get_client` — Gemini 클라이언트 싱글턴
- `src.rag_chain.RAG_CONFIG` — ChromaDB 경로/컬렉션 설정
- `src.vector_db.get_or_create_collection` — ChromaDB 접근

### 외부 라이브러리
- `google.genai`
- `pathlib`, `json`, `logging` (표준 라이브러리)

---

## 호출 경로

```
GET /summaries (api.py)
    └── get_summaries()
            ├── _get_file_chunks()       # ChromaDB 조회
            ├── _load_cache()            # 캐시 로드
            ├── _generate_summaries()    # Gemini 호출 (신규 파일 있을 때만)
            └── _save_cache()            # 캐시 저장 (변화 있을 때만)
```

---

최종 수정: 2026-03-20
관련 파일: 'src/summary_service.py', 'src/api.py'
---
