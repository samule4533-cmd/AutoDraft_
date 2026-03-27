# summary_service.py

ChromaDB에 적재된 문서의 AI 요약을 생성하고 캐시로 관리하는 모듈.

## 개요

`GET /summaries` 엔드포인트의 비즈니스 로직을 담당. ChromaDB 파일 목록을 수집하고, 신규/변경 파일만 Gemini에 요약을 요청해 `summaries.json`에 캐시한다.

---

## 캐시 전략

| 상황 | 동작 |
|------|------|
| 변화 없음 | Gemini 호출 0회 — 캐시 즉시 반환 |
| 새 파일 추가 | 신규 파일만 Gemini 호출 → 캐시 병합 |
| 파일 내용 변경 | 청크 수 변화 감지 → 해당 파일 재요약 |
| 파일 삭제 | 캐시에서 해당 파일 항목 제거 |

청크 수(`chunk_count`) 추적으로 파일 내용 변경을 감지한다.

---

## 캐시 파일 구조

**경로**: `data/processed/summaries.json`

```json
{
  "summaries": {
    "특허_A.pdf": {
      "summary": "이 문서는 건물 에너지 모델링...",
      "chunk_count": 42
    }
  }
}
```

구버전(문자열 형식) 캐시를 자동으로 새 형식으로 마이그레이션한다.

---

## 주요 함수

### `get_summaries() -> list[dict]`
메인 진입점.

```
_get_file_chunks()        # ChromaDB에서 현재 파일 목록 + 청크 수 수집
  ↓
_load_cache()             # 기존 캐시 로드 + 구버전 형식 마이그레이션
  ↓
신규/변경/삭제 파일 감지
  (청크 수 불일치 → 변경으로 판단)
  ↓
신규·변경 파일 있으면 _generate_summaries() 호출
  ↓
_save_cache()             # 캐시 업데이트 (원자적 저장)
  ↓
[{"index": i, "filename": f, "summary": s}, ...] 반환
```

---

### `_get_file_chunks() -> dict[str, list[str]]`
ChromaDB 전체 청크를 `{source_file: [청크 텍스트, ...]}` 형태로 수집.

---

### `_generate_summaries(file_chunks, target_files) -> dict[str, dict]`
단일 Gemini 호출로 `target_files` 전체 요약 생성.

- 파일당 최대 `_MAX_CHUNKS_PER_FILE = 30`개 청크 사용 (토큰 한도 안전 마진)
- `temperature=0.0` — 출력 일관성
- JSON 파싱 실패 시 해당 파일은 `"요약 생성에 실패했습니다."` 처리
- 반환: `{filename: {"summary": str, "chunk_count": int}}`

---

## 설계 포인트

### 1. Gemini 1회 호출 전략
문서 N개를 개별 호출(N회) 대신 한 번에 배치 처리. Gemini 2.0 Flash의 1M 토큰 컨텍스트를 활용해 비용/지연 절감.

### 2. 청크 수 기반 변경 감지
파일명이 같아도 청크 수가 달라지면 재요약을 트리거. 문서 내용이 변경되거나 재파싱된 경우를 감지.

### 3. 파일명 정규화
Gemini가 드물게 파일명에 언더스코어/공백 변형을 넣는 경우에 대비한 정규화 처리.

### 4. 코드펜스 파싱 방어
Gemini 응답에 ` ```json ` 코드펜스가 포함될 경우 자동 제거 후 JSON 파싱.

---

## 의존성

### 내부
- `llm_api.get_client`
- `rag_chain.RAG_CONFIG`
- `vector_db.get_or_create_collection`

### 외부
- `google.genai`, `pathlib`, `json`, `logging`

---
최종 수정: 2026-03-27
관련 파일: `src/summary_service.py`
---
