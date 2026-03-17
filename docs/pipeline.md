# Pipeline Reference

PDF 원본 문서에서 RAG 질의응답까지의 **전체 데이터 흐름** 상세 문서.

---

## 목차

- [전체 흐름 한눈에 보기](#전체-흐름-한눈에-보기)
- [Step 1 — PDF 파싱](#step-1--pdf-파싱)
- [Step 2 — 청킹](#step-2--청킹)
- [Step 3 — 결과 저장](#step-3--결과-저장)
- [Step 4 — 벡터 DB 적재](#step-4--벡터-db-적재)
- [Step 5 — RAG 질의응답](#step-5--rag-질의응답)
- [일괄 처리 파이프라인](#일괄-처리-파이프라인)
- [실행 방법](#실행-방법)
- [디버깅 가이드](#디버깅-가이드)

---

## 전체 흐름 한눈에 보기

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Ingestion Pipeline                           │
│                                                                     │
│  data/raw/company/{subdir}/{file}.pdf                               │
│          │                                                          │
│          ▼ Step 1: PDF 파싱                                         │
│  pdf_parser.py                                                      │
│    ├─ Gemini File API 업로드 (한글명 → upload_input.pdf 복사)       │
│    ├─ PDF → Markdown 변환 (temperature=0, max_tokens=40000)        │
│    └─ normalize_markdown_headings() (**제목** → ## 헤더 보정)       │
│          │                                                          │
│          ▼ Step 2: 청킹                                             │
│  chunker.py                                                         │
│    ├─ Stage 1: 헤더(#/##/###) 기준 섹션 분리                        │
│    └─ Stage 2: 1500자 초과 섹션 → 표/문장 경계 기준 2차 분리        │
│          │                                                          │
│          ▼ Step 3: 결과 저장                                        │
│  output_writer.py                                                   │
│    data/processed/parsing_result_company/{subdir}/{file}/           │
│    ├─ {file}.md                                                     │
│    ├─ {file}.json                                                   │
│    ├─ parse_report.json                                             │
│    ├─ fields.json                                                   │
│    └─ vector_chunks.json  ◀── ChromaDB 적재 입력                   │
│          │                                                          │
│          ▼ Step 4: 벡터 DB 적재                                     │
│  vector_db.py / company_vectordb.py                                 │
│    ├─ 임베딩 생성 (local: MiniLM-L12 | openai: text-embedding-3)   │
│    └─ ChromaDB upsert (배치 50개)                                   │
│          │                                                          │
│  data/vector_store/chroma/  ◀── 영구 저장                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         RAG Query Pipeline                          │
│                                                                     │
│  사용자 질문                                                         │
│          │                                                          │
│          ▼ Step 5: RAG 질의응답                                     │
│  rag_chain.py :: ask()                                              │
│    ├─ retrieve()           ChromaDB 유사도 검색 (top-N)             │
│    ├─ filter_by_threshold() distance ≤ 0.5 필터                    │
│    ├─ [fallback]           0개 통과 시 → "문서에 없습니다" 반환     │
│    ├─ build_context_block() 청크 본문 + 출처 조합                   │
│    ├─ build_prompt()        시스템 프롬프트 + 참고 문서 + 질문      │
│    ├─ generate_answer()     Gemini LLM 호출 (temperature=0)        │
│    └─ format_citations()    (header, source_file) 기준 중복 제거    │
│          │                                                          │
│  RagResult { answer, citations, retrieved_count, ... }             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1 — PDF 파싱

**담당 모듈:** `src/pdf_parser.py`

### 입력

```
data/raw/company/{subdir}/{filename}.pdf
```

### 처리 과정

```
1. _upload_and_wait()
   ├─ 한글 파일명 → tmpdir/upload_input.pdf 복사 (Gemini API 오류 방지)
   ├─ genai.Client.files.upload() 호출
   └─ ACTIVE 상태 될 때까지 폴링 (최대 60초, 2초 간격)

2. parse_pdf_to_markdown()
   ├─ Gemini 파싱 프롬프트 전달
   │   - 표, 숫자, 날짜, 인증번호, 특허 청구항 보존 강조
   │   - 섹션 계층 그대로 유지
   │   - 요약/생략 금지
   └─ Markdown 텍스트 반환

3. normalize_markdown_headings()
   ├─ 정규식: 줄 전체가 **텍스트** 형태인 경우 → ## 텍스트 변환
   └─ Gemini가 종종 헤더를 볼드로 출력하는 문제 보정
```

### 파싱 프롬프트 핵심 지침

| 항목 | 지침 |
|------|------|
| 레이아웃 | 원본 구조와 계층 그대로 유지 |
| 표 | Markdown 테이블 형식으로 변환, 셀 병합 주석 처리 |
| 숫자/날짜 | 원문 그대로 (변환/반올림 금지) |
| 특허 | 발명 명칭, 기술 분야, 배경, 해결책, 효과, 청구항, 도면 설명 포함 |
| 인증서 | 인증번호, 발급 기관, 유효기간, 인증 범위 보존 |

### 출력

```python
list[dict]  # 청크 목록, 각 항목은 vector_chunks.json 형식과 동일
```

---

## Step 2 — 청킹

**담당 모듈:** `src/chunker.py`

### 청킹 전략

#### Stage 1: 헤더 기반 1차 분리

```
# 제목          →  섹션 1
## 소제목 1     →  섹션 2
### 세부 항목   →  섹션 3
## 소제목 2     →  섹션 4
```

- `#`, `##`, `###` 헤더를 경계로 섹션 분리
- 헤더만 있고 본문 없는 섹션은 필터링 (환각 방지)

#### Stage 2: 크기 기반 2차 분리 (> 1500자)

```
표(table) 섹션
  └─ 헤더 행 + 구분자(---) 유지
     데이터 행을 그룹 단위로 분리
     각 청크마다 헤더 행 복사 (표 context 유지)

텍스트 섹션
  └─ 1차: 한국어 문장 경계 분리
         "다.", "임.", "음.", "등." 기준
     2차: 빈 줄(단락 경계) 분리
     3차: 1300자 제한으로 강제 분리
```

### 청크 구조 예시

```json
{
  "chunk_id": "patent_001_003",
  "text": "## 발명의 효과\n본 발명에 따르면 건물 에너지 모델링을 자동화하여...",
  "metadata": {
    "document_id": "patent_001",
    "source_file": "data/raw/company/patent/sample.pdf",
    "header": "## 발명의 효과",
    "section_order": 3,
    "chunk_type": "section",
    "chunk_position": "only",
    "has_table": false,
    "doc_type": "company",
    "model": "gemini-2.0-flash",
    "mode": "fast_gemini_file_api"
  }
}
```

### 청크 크기 설정

| 설정 | 값 | 설명 |
|------|-----|------|
| `section_max_len` | 1500자 | 이 이상이면 2차 분리 트리거 |
| `group_max_len` | 1300자 | 개별 텍스트 청크 최대 크기 |

---

## Step 3 — 결과 저장

**담당 모듈:** `src/output_writer.py`

### 출력 경로

```
data/processed/parsing_result_company/{subdir}/{pdf_stem}/
├── {pdf_stem}.md               ← 전체 Markdown 원문
├── {pdf_stem}.json             ← 문서 구조 + 청크 메타데이터 전체
├── parse_report.json           ← 파싱 메타데이터 (모델, 소요 시간, 청크 수)
├── fields.json                 ← 구조화 필드 추출 결과 (company 모드: 비어 있음)
└── vector_chunks.json          ← ChromaDB 적재 전용 형식
```

### `vector_chunks.json` 형식

```json
[
  {
    "chunk_id": "sample_001",
    "text": "섹션 본문 텍스트...",
    "metadata": {
      "document_id": "sample",
      "source_file": "...",
      "header": "## 제목",
      "doc_type": "company",
      "has_table": false,
      "chunk_type": "section",
      "chunk_position": "only",
      "section_order": 1
    }
  }
]
```

### `parse_report.json` 형식

```json
{
  "source_file": "data/raw/company/sample.pdf",
  "model": "gemini-2.0-flash",
  "elapsed_sec": 45.23,
  "chunk_count": 42,
  "text_len": 15000,
  "korean_ratio": 0.8234,
  "warnings": []
}
```

---

## Step 4 — 벡터 DB 적재

**담당 모듈:** `src/vector_db.py`, `src/company_vectordb.py`

### 적재 과정

```
1. collect_all_chunks()
   └─ data/processed/ 하위 모든 vector_chunks.json 수집

2. (선택) reset_collection()
   └─ CHROMA_RESET=true 시 컬렉션 삭제 후 재생성
      임베딩 모델 전환 시 필수 (차원 불일치 방지)

3. prepare_chroma_items()
   ├─ 빈 텍스트 청크 필터링
   ├─ 메타데이터 타입 검증 (ChromaDB는 str/int/float/bool만 허용)
   └─ id, document, metadata 분리

4. upsert_chunks_to_chroma()
   ├─ 배치 크기: 50
   └─ upsert (중복 chunk_id는 덮어쓰기)
```

### ChromaDB 컬렉션 설정

```python
collection = client.get_or_create_collection(
    name="ninewatt_company",
    metadata={"hnsw:space": "cosine"}
)
```

### 임베딩 모델 비교

```
[local: paraphrase-multilingual-MiniLM-L12-v2]
  차원: 384  |  속도: 빠름  |  비용: 무료  |  오프라인: O
  적합: 개발/테스트, 비용 민감 환경

[openai: text-embedding-3-small]
  차원: 1536  |  속도: API 호출  |  비용: 유료  |  오프라인: X
  적합: 프로덕션, 한국어 검색 품질 우선
```

---

## Step 5 — RAG 질의응답

**담당 모듈:** `src/rag_chain.py`

### `ask()` 함수 흐름

#### 5-1. retrieve()

```python
results = query_collection(
    query_text=query,
    n_results=10,        # RAG_CONFIG["n_results"]
    where=filters        # 선택적 메타데이터 필터
)
# 반환: [{chunk_id, text, metadata, distance, header, source_file}]
```

ChromaDB 반환 중첩 구조를 평탄화해 리스트로 변환.

#### 5-2. filter_by_threshold()

```python
# cosine distance: 낮을수록 유사
passed = [c for c in chunks if c["distance"] <= threshold]

# 예시 로그:
# [req_id] threshold=0.5: 10개 검색 → 6개 통과, 4개 제거
```

| 거리 범위 | 의미 |
|-----------|------|
| 0.0 ~ 0.3 | 매우 유사 |
| 0.3 ~ 0.5 | 관련 있음 (기본 threshold) |
| 0.5 ~ 0.8 | 약한 관련 |
| 0.8 이상  | 무관, 차단 |

#### 5-3. build_context_block()

```
[문서 1]
출처: 발명의 효과 | data/raw/company/patent/sample.pdf

본 발명에 따르면 건물 에너지 모델링을 자동화하여 설계 시간을
80% 단축할 수 있다...

---
[문서 2]
출처: 특허 청구항 | ...
...
```

- 최대 6000자 (`max_context_chars`) 초과 시 낮은 랭크 청크부터 제거
- 헤더에서 `#` 기호 제거 (가독성)

#### 5-4. build_prompt()

```
[시스템]
당신은 회사 내부 문서를 기반으로 질문에 답하는 어시스턴트입니다.
[규칙] 참고 문서에만 근거, 없으면 "없습니다" 답변, [출처: 헤더명] 표기...

[이전 대화]  ← Phase 2: chat_history 있을 때만 포함
사용자: ...
어시스턴트: ...

[참고 문서]
[문서 1] ...
---
[문서 2] ...

[질문]
{query}
```

#### 5-5. generate_answer()

```python
response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={"temperature": 0.0}
)
```

**Rate Limit 처리:**

```
429 응답 수신
  ├─ RPM (per-minute) → 60s / 120s / 180s 대기 후 재시도 (최대 3회)
  └─ RPD (per-day)    → 즉시 중단, llm_error fallback 반환
```

#### 5-6. format_citations()

```python
# 중복 제거: (header, source_file) 동일하면 하나만 유지
citations = [
    Citation(
        chunk_id="patent_001_003",
        header="발명의 효과",
        source_file="data/raw/company/patent/sample.pdf",
        distance=0.2847
    )
]
```

### 최종 반환 예시

```python
RagResult(
    answer="회사의 주요 특허는 건물 에너지 모델링 자동화 시스템입니다. "
           "[출처: 발명의 효과]\n설계 시간을 80% 단축할 수 있으며...",
    citations=[
        Citation(chunk_id="patent_001_003", header="발명의 효과",
                 source_file="sample.pdf", distance=0.2847)
    ],
    retrieved_count=10,
    passed_threshold=4,
    top_distance=0.2847,
    fallback=False,
    fallback_reason=None
)
```

---

## 일괄 처리 파이프라인

여러 PDF를 한 번에 처리할 때의 흐름.

```
company_ingest.py                   company_vectordb.py
─────────────────                   ───────────────────
1. PDF 목록 스캔                    1. vector_chunks.json 전체 수집
   data/raw/company/**/*.pdf           data/processed/***/vector_chunks.json

2. 각 PDF에 대해:                   2. (선택) 컬렉션 초기화
   ├─ 출력 있으면 스킵                  CHROMA_RESET=true 시
   └─ 없으면 parse_single_pdf()
                                    3. 일괄 upsert
3. 실패 시 재시도                      배치 50개 단위
   지수 백오프 (5s → 10s → 20s)
   최대 3회                         4. 완료 확인
                                       컬렉션 총 청크 수 출력
4. failed_parse.log 기록
```

---

## 실행 방법

### 단일 PDF 파싱

```bash
cd src
uv run python pdf_parser.py
# .env의 DEFAULT_PDF_SUBDIR + DEFAULT_PDF_NAME 대상
```

### 일괄 파싱

```bash
cd src
uv run python company_ingest.py
# data/raw/company/ 하위 모든 PDF 처리
```

### 벡터 DB 적재

```bash
cd src
uv run python company_vectordb.py
# data/processed/ 하위 모든 vector_chunks.json → ChromaDB
```

### RAG 질의

```bash
# 단일 질문
cd src
uv run python rag_chain.py "회사 주요 특허는 무엇인가요?"

# 테스트 케이스 전체 실행
cd src
uv run python rag_chain.py --test
```

### 의존성 설치

```bash
uv sync
```

---

## 디버깅 가이드

### 증상별 원인 분석

| 증상 | 가능한 원인 | 진단 방법 | 해결 |
|------|------------|-----------|------|
| "사내 문서에 없습니다" (예상 외) | threshold가 너무 낮음 | `RagResult.top_distance` 확인 | `distance_threshold` 상향 (0.5 → 0.6) |
| 관련 없는 청크 포함 | threshold가 너무 높음 | 반환된 `citations` 확인 | `distance_threshold` 하향 |
| 답변이 없거나 빈 문자열 | Gemini 안전 필터 트리거 | 로그 "빈 응답" 확인 | 질문 표현 변경, 프롬프트 재검토 |
| 429 오류 반복 | API 쿼터 초과 | 로그 RPD/RPM 구분 | RPD: 내일까지 대기, RPM: 잠시 대기 |
| 임베딩 차원 오류 | 모델 전환 후 미초기화 | ChromaDB 오류 메시지 | `CHROMA_RESET=true` 후 재적재 |
| 한글 깨짐 | PDF 이미지 기반 (OCR 불가) | 파싱 결과 `.md` 파일 확인 | Gemini 파싱 품질 한계, 이미지 캡션 활성화 검토 |

### 검색 품질 진단 흐름

```
1. retrieved_count 확인
   └─ 0이면? → 컬렉션 비어있거나 임베딩 오류

2. passed_threshold 확인
   └─ 0이면? → threshold 너무 낮음 또는 청크 품질 문제

3. top_distance 확인
   └─ 0.7 이상? → 관련 문서가 없거나 청킹이 문맥 파괴

4. citations 내용 확인
   └─ 엉뚱한 문서? → 임베딩 모델 품질 또는 청크 텍스트 문제

5. 답변 내 [출처] 표기 확인
   └─ 없음? → 시스템 프롬프트 미준수 (temperature 확인)
```

### 주요 로그 키워드

```bash
# req_id로 특정 요청 전 과정 추적
grep "req_id=abc123" app.log

# 검색 결과 확인
grep "threshold=" app.log

# Rate limit 확인
grep "429\|rate_limit\|RPD\|RPM" app.log

# Fallback 발생 확인
grep "fallback_reason" app.log
```
