# company_ingest.py

회사 문서 PDF 배치 파싱 파이프라인. `data/raw/company/` 스캔 → 신규 파일 파싱 → ChromaDB + parent_index 적재까지 한 번에 실행. `--rechunk-only` 모드로 재파싱 없이 재청킹도 지원.

## 개요

`company_ingest.py`는 PDF 수집부터 ChromaDB 적재까지 전 과정을 명령어 하나로 완료하는 배치 인제스트 진입점이다. 개별 파싱 로직은 `pdf_parser.py`, 적재 로직은 `vector_db.py`와 `parent_store.py`가 담당하며, 이 파일은 전체 파이프라인을 오케스트레이션한다.

---

## 실행 모드

```bash
# 기본 모드: 신규 PDF만 파싱 + 적재
cd src && uv run python company_ingest.py

# 재청킹 모드: API 호출 없이 기존 .md 재청킹 + 전체 재적재
cd src && uv run python company_ingest.py --rechunk-only
```

---

## 설정 상수

| 상수 | 값 |
|------|-----|
| `COMPANY_INPUT_DIR` | `data/raw/company/` |
| `COMPANY_OUTPUT_ROOT` | `data/processed/parsing_result_company/` |
| `FAILED_LOG_PATH` | `COMPANY_OUTPUT_ROOT/failed_parse.log` |
| `MAX_RETRIES` | `3` |
| `RETRY_BASE_DELAY` | `5`초 (지수 백오프 시작값) |

---

## 주요 함수

### `parse_all() -> None`
기본 파싱 루프.

```
data/raw/company/**/*.pdf 스캔
  ↓
각 PDF → parse_report.json + *.md 존재 확인
  ├── 이미 있음 → SKIP
  └── 없음 → parse_with_retry() 호출
                  ↓
              파싱 성공 → 청킹 → _route_and_ingest()
              파싱 실패 → failed_parse.log 기록
  ↓
전체 결과 요약 출력 (파싱/스킵/실패 수)
```

스킵 판단 기준: `parse_report.json` + `*.md` 존재 여부.

---

### `rechunk_all() -> None`
재청킹 모드. Gemini API 호출 없이 기존 파싱 결과를 재청킹해 전체 재적재.

```
COMPANY_OUTPUT_ROOT/**/parse_report.json + *.md 수집
  ↓
split_markdown_into_chunks() 재실행
  ↓
ChromaDB 컬렉션 reset (force_reset=True)
  ↓
전체 청크 일괄 upsert + parent_index.json 교체
```

임베딩 모델 변경, 청킹 로직 수정 후 전체 재적재 시 사용.

---

### `_route_and_ingest(chunks, force_reset) -> None`
청크 타입별 라우팅 + 적재.

| 청크 타입 | 처리 |
|----------|------|
| `section`, `child` | ChromaDB upsert |
| `parent` | `parent_store.merge_parents()` |

적재 후 `bm25_retriever.rebuild_index()` 호출해 인메모리 인덱스 갱신.

---

### `parse_with_retry(pdf_path) -> list`
단일 PDF 파싱 + 재시도.

- 실패 시 지수 백오프: 5s → 10s → 20s
- 모든 시도 실패 시 예외 raise → `failed_parse.log` 기록

---

## 처리 흐름

```
data/raw/company/
    │
    ├── 특허리스트_1/특허A/특허A.pdf
    └── 특허리스트_2/특허B/특허B.pdf

      ↓ company_ingest.py

data/processed/parsing_result_company/
    ├── 특허리스트_1/특허A/
    │   ├── parse_report.json
    │   └── 특허A.md
    └── 특허리스트_2/특허B/
        ├── parse_report.json
        └── 특허B.md

      ↓ _route_and_ingest()

ChromaDB (section + child 청크)
parent_index.json (parent 청크)
BM25 인덱스 (인메모리, 서버 시작 시 재빌드)
```

---

## 설계 포인트

### 1. 스킵 로직 (증분 처리)
`parse_report.json` + `*.md` 두 파일 모두 존재하면 스킵. API 호출 비용 없이 새로 추가된 PDF만 처리.

### 2. Per-file 적재
파일 하나 파싱 완료 시 즉시 적재. 전체 파싱 후 일괄 적재가 아니라 중간 실패 시 이미 처리된 파일은 보존.

### 3. failed_parse.log 누적
실패 파일 정보(시각, 경로, 원인)를 append 방식으로 기록. 재처리 대상 확인과 에러 유형 분석에 사용.

### 4. 재청킹 모드
`--rechunk-only` 플래그로 Gemini API 호출 없이 청킹 로직만 변경해 전체 재적재 가능. 청킹 파라미터 튜닝 후 빠르게 반영할 때 사용.

---

## 의존성

### 내부
- `pdf_parser.parse_single_pdf`
- `chunker.split_markdown_into_chunks`
- `vector_db.upsert_chunks_to_chroma`, `vector_db.reset_collection`
- `parent_store.merge_parents`, `parent_store.save_index`
- `bm25_retriever.rebuild_index`

### 외부
- `asyncio`, `argparse`, `logging`, `python-dotenv`, `pathlib`

---
최종 수정: 2026-03-27
관련 파일: `src/company_ingest.py`
---
