# vector_db.py

## 개요
`vector_db.py`는 문서 청크를 벡터 형태로 저장하고 검색하는 기능을 담당하는 모듈이다.  
프로젝트에서 ChromaDB를 직접 다루는 핵심 계층으로, 청크 전처리, 임베딩 함수 선택, 컬렉션 생성 및 관리, 벡터 upsert, 유사도 검색, CLI 기반 sanity check까지 포함한다.

이 파일의 핵심 목적은 다음과 같다.

- 청크 데이터를 ChromaDB에 적재 가능한 형태로 정리
- 로컬 임베딩과 OpenAI 임베딩을 선택적으로 지원
- ChromaDB PersistentClient와 컬렉션을 일관된 방식으로 관리
- 청크를 batch 단위로 upsert
- 자연어 질의 기반 유사 청크 검색 제공
- 검색 결과를 사람이 읽기 쉬운 형태로 출력

즉, 이 모듈은 **문서 청크와 벡터 저장소 사이를 연결하는 벡터 검색 인프라 계층**이다.

---

## 역할
이 파일은 파싱 및 청킹이 끝난 결과를 벡터 DB에 저장하고, 이후 질의에 대해 유사한 청크를 검색하는 역할을 한다.

구체적으로 다음 책임을 가진다.

- JSON 청크 파일 로드
- ChromaDB에 맞는 metadata 정리
- 적재 대상 청크 전처리
- 임베딩 함수 선택 및 생성
- PersistentClient 캐싱
- 컬렉션 생성, 접근, 초기화
- 청크 batch upsert
- 자연어 쿼리 검색
- 검색 결과 요약 출력
- 단독 실행 시 컬렉션 상태 확인

즉, 이 파일은 **벡터 DB 관련 공통 로직을 한 곳에 모아둔 저장/검색 레이어**라고 볼 수 있다.

---

## 이 파일이 필요한 이유
RAG 시스템에서 문서를 검색하려면 단순 텍스트 파일만으로는 부족하다.  
문서를 청크 단위로 나누고, 각 청크를 임베딩하여 벡터 저장소에 넣어야 유사도 검색이 가능해진다.

이 과정에서 다음과 같은 문제가 발생할 수 있다.

- metadata 구조가 ChromaDB와 맞지 않을 수 있음
- 임베딩 제공자를 바꿀 수 있어야 함
- 컬렉션 차원 불일치 문제가 생길 수 있음
- 쿼리마다 클라이언트를 새로 만들면 느려짐
- 검색 결과를 디버깅하기 어렵다

`vector_db.py`는 이러한 문제를 해결하기 위해, **임베딩-저장-검색 전 과정을 안정적으로 다루는 공통 벡터 계층**으로 설계되었다.

---

## 주요 구성 요소

### 환경 및 경로
- `.env`를 로드하여 임베딩 설정을 읽음
- 프로젝트 루트 기준 ChromaDB 저장 경로 계산
- 단독 PDF/단일 컬렉션 기준 테스트에 사용할 기본값 제공

---

## 주요 함수

### `load_chunks_from_json(chunks_path: Path) -> List[Dict[str, Any]]`
JSON 파일에서 청크 리스트를 읽어오는 함수이다.

#### 역할
- `vector_chunks.json` 같은 청크 파일을 로드
- 이후 적재 로직의 입력으로 사용

#### 사용 목적
파일 기반 파싱 결과를 벡터 적재 단계로 연결하기 위함이다.

---

### `clean_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]`
ChromaDB에 안전하게 저장할 수 있도록 metadata를 정리하는 함수이다.

#### 동작 방식
- `None` 값 제거
- `str`, `int`, `float`, `bool` 타입은 그대로 유지
- 단순 타입 리스트는 유지
- `dict`나 복합 객체는 문자열로 변환

#### 사용 목적
ChromaDB metadata 필드는 제한된 타입만 안정적으로 다룰 수 있으므로,  
적재 전에 형식을 정리하여 오류를 줄이기 위함이다.

#### 의미
이 함수는 벡터 검색 품질보다 **적재 안정성**을 위한 방어 계층에 가깝다.

---

### `prepare_chroma_items(chunks: List[Dict[str, Any]], default_doc_type: Optional[str] = None) -> Dict[str, List[Any]]`
청크 리스트를 ChromaDB upsert에 바로 넣을 수 있는 형태로 변환하는 함수이다.

#### 역할
- 빈 텍스트 청크 제거
- metadata 정리
- `doc_type` 기본값 보정
- `ids`, `documents`, `metadatas` 구조로 변환

#### 반환 구조
```python
{
    "ids": [...],
    "documents": [...],
    "metadatas": [...]
}
```

---

### `_OpenAIEmbeddingFunction._embed(input)`
OpenAI 임베딩 API를 호출하는 내부 메서드이다.

#### 특징
- 2500자 초과 청크 자동 truncate
- 실패 시 최대 3회 재시도 (10초 → 20초 백오프)
- 모든 재시도 실패 시 에러 로그 후 raise

#### 목적
수백 개 문서 배치 처리 중 일시적인 API 오류나 rate limit으로 전체 인제스트가 중단되는 것을 방지한다.

---

## 변경 이력

### 2026-03-19
- 파일 최상단 불필요한 주석 제거
- `load_and_upsert_chunks()` 미사용 함수 제거
- OpenAI 임베딩 재시도 로직 추가 (3회, 10s/20s 백오프)

---
최종 수정: 2026-03-19
관련 파일: 'src/vector_db.py'
---