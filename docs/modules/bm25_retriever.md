# bm25_retriever.py

BM25 키워드 검색 인덱스 관리 모듈. Kiwi 한국어 형태소 분석기로 토크나이즈하고, 인메모리 BM25 인덱스로 키워드 검색을 제공한다.

## 개요

벡터 검색은 의미 유사도에 강하지만, 특허번호·날짜·고유명사 등 정확한 키워드 매칭에는 약하다. `bm25_retriever.py`는 이 약점을 보완하는 BM25 키워드 검색 레이어다.

- BM25 인덱스는 디스크가 아닌 메모리에만 존재 (휘발성)
- 서버 시작 시 ChromaDB에서 전체 청크를 읽어 인메모리 인덱스 빌드
- 새 문서 ingest 후 `rebuild_index()`로 갱신

---

## 전역 변수

| 변수 | 타입 | 설명 |
|------|------|------|
| `_bm25` | `BM25Okapi \| None` | BM25 인덱스 싱글턴 |
| `_chunks` | `list[dict]` | `_bm25`와 인덱스가 1:1 대응되는 청크 목록 |
| `_index_lock` | `threading.Lock` | build/search 간 동시성 제어 |
| `_kiwi` | `Kiwi \| None` | Kiwi 형태소 분석기 싱글턴 |
| `_kiwi_lock` | `threading.Lock` | Kiwi 초기화 더블-체크 락 |

---

## 토크나이저

### 보존 품사 (`_KEEP_POS`)

| 태그 | 의미 |
|------|------|
| `NNG`, `NNP`, `NNB` | 일반/고유/의존 명사 |
| `NR`, `NP` | 수사, 대명사 |
| `VV` | 동사 어간 (형용사 제외) |
| `XR` | 어근 |
| `SL`, `SH`, `SN` | 외국어, 한자, 숫자 |
| `W_SERIAL` | 일련번호·날짜 (특허번호 보존) |
| `MM`, `MAG` | 관형사, 일반부사 |

조사(JK*, JX, JC), 어미(E*), 접속부사(MAJ)는 제거 → "회사는"/"회사가" → 모두 "회사" 토큰.

### 도메인 불용어 (`_DOMAIN_STOPWORDS`)

특허 문서 전체에 고빈도로 등장해 IDF가 낮고 검색 신호가 없는 단어:

- **동사**: `관하`, `따르`, `위하`, `통하`, `의하`, `이루`, `포함하`, `구성하`, `수행하`, `제공하`
- **의존명사**: `것`, `수`, `년`, `호`, `일`, `등`
- **지시어**: `상기`, `해당`, `본`

### 사용자 사전 (`_PATENT_USER_WORDS`)

Kiwi가 분리하는 복합어를 단일 명사로 등록:

| 단어 | 이유 |
|------|------|
| `청구항` | 청구 + 항으로 분리 방지 |
| `출원인`, `발명자`, `특허청` | 복합어 단일화 |
| `등록번호`, `출원번호`, `출원일`, `등록일` | 특허 메타 복합어 |
| `기준면`, `풋프린트` | 도메인 특수 복합어 |

---

## 동시성 설계

### 문제
`_bm25`와 `_chunks`는 전역 변수. build와 search가 동시에 실행되면:
1. `_bm25 = new` 와 `_chunks = new` 사이에 search가 끼어들면 인덱스-청크 불일치
2. `_chunks = []` 초기화 직후 search 실행 시 빈 결과

### 해결
- **build/rebuild**: 로컬에서 완전히 빌드 → `_index_lock` 안에서 한 번에 swap
- **search**: `_index_lock` 안에서 레퍼런스 스냅샷만 복사 → 즉시 lock 해제 → 이후 읽기는 스냅샷으로 진행

---

## 주요 함수

### `build_index(collection) -> None`
ChromaDB 컬렉션 전체 청크 로드 → Kiwi 토크나이즈 → `BM25Okapi(corpus)` 생성 → lock 안에서 swap.

---

### `search(query, top_k=10) -> list[dict]`
BM25 검색. lock으로 스냅샷 획득 후 즉시 해제 → 스냅샷으로 검색. 결과에 `bm25_score` 필드 추가. score ≤ 0인 청크는 제외.

---

### `fetch_by_claim_numbers(numbers) -> list[dict]`
청구항 번호에 해당하는 청크를 헤더 패턴 매칭으로 직접 반환.

- `"청구항 N"` 이 header에 포함된 청크 반환
- `bm25_score = float("inf")` 설정 → RRF에서 rank 0 취급 (최상위 점수)
- BM25 점수 계산 우회 → IDF 희석 문제 완전히 회피

**사용 이유**: Kiwi가 조사를 제거하면서 "청구항"이 모든 특허 청크에 등장 → IDF가 낮아져 청구항 헤더 청크 순위 하락. 헤더 직접 매칭으로 우회.

---

### `rebuild_index(collection) -> None`
새 문서 ingest 후 BM25 인덱스 갱신. `build_index()` 래퍼.

---

### `debug_tokenize(text) -> None`
Kiwi 원시 형태소 결과와 최종 BM25 토큰을 나란히 출력하는 디버그 유틸리티.

---

## 의존성

### 외부 라이브러리
- `rank_bm25.BM25Okapi`
- `kiwipiepy.Kiwi`
- `threading`, `re`, `logging`

---
최종 수정: 2026-03-27
관련 파일: `src/bm25_retriever.py`
---
