# chunker.py

Markdown → RAG 청크 분리 모듈. 단순 길이 분할이 아닌 헤더 구조, 문단 경계, 표 형식을 고려해 section/parent/child 계층 청크를 생성한다.

## 개요

PDF 파싱으로 얻은 Markdown을 벡터 DB 적재에 적합한 청크로 변환한다. 짧은 섹션은 단일 section 청크로, 긴 섹션은 parent(도입부) + child(세부 내용) 계층 구조로 분리한다.

---

## 설정값: `CHUNKER_CONFIG`

| 키 | 기본값 | 의미 |
|----|--------|------|
| `section_max_len` | `1500` | 이 이하 섹션은 단일 청크 유지 |
| `max_paragraphs` | `3` | 이 초과 문단 수는 parent-child로 분리 |
| `enum_min_hits` | `3` | 열거 패턴 감지 기준 횟수 |
| `group_max_len` | `1300` | child 청크 최대 길이 |
| `min_child_len` | `200` | child 최소 길이 (미만이면 이전 child에 병합) |
| `min_chunk_content_len` | `80` | 콘텐츠 최소 길이 (미만이면 청크 생성 제외) |
| `parent_intro_target` | `600` | parent 도입부 목표 길이 |
| `parent_intro_min` | `350` | parent 도입부 최소 길이 |
| `parent_intro_max` | `800` | parent 도입부 최대 길이 (강제 cut) |

---

## 청크 타입

| `chunk_type` | 설명 | ChromaDB 적재 | parent_index |
|-------------|------|--------------|--------------|
| `section` | 짧은 섹션 단일 청크 | ✅ | ✗ |
| `parent` | 긴 섹션 도입부 요약 | ✗ | ✅ |
| `child` | 긴 섹션 세부 내용 | ✅ | parent_index에 텍스트 보관 |

parent 청크는 ChromaDB에 적재하지 않는다 (벡터 검색 대상은 child만). `parent_store.py`가 도입부와 child 목록을 관리하며, 검색 후 컨텍스트 확장에 사용된다.

---

## 청크 ID 체계

| 타입 | 형식 |
|------|------|
| section | `{document_id}_c{N}` |
| parent | `{document_id}_p{sec_idx}` |
| child | `{document_id}_p{sec_idx}_ch{i}` |

---

## 청크 메타데이터

```python
{
    "chunk_id":      "...",
    "header":        "## 청구항 1",
    "text":          "...",
    "metadata": {
        "document_id":    "...",
        "source_file":    "특허.pdf",
        "section_order":  0,
        "header":         "## 청구항 1",
        "source":         "gemini_file_api_markdown",
        "model":          "gemini-2.0-flash",
        "chunk_type":     "section" | "parent" | "child",
        "has_table":      True | False,
        "chunk_position": "only" | "first" | "middle" | "last",
        "parent_chunk_id": "...",   # child 청크에만 존재
    }
}
```

---

## 주요 함수

### `split_markdown_into_chunks(markdown_text, document_id, source_pdf, model_name, source_type) -> list[dict]`
메인 진입점.

```
Markdown 헤더 파싱 → 섹션 분리
  ↓
섹션 길이/구조 분석 → _needs_parent_child() 판단
  ├── False → section 청크 1개
  └── True  → _extract_parent_intro() + _build_child_chunks()
                    ↓                           ↓
              parent 청크 1개         child 청크 N개
```

---

### `_needs_parent_child(section_text, non_empty_paras) -> bool`
다음 중 하나라도 해당하면 parent-child 분리:
- 섹션 길이 > `section_max_len(1500)`
- 문단 수 > `max_paragraphs(3)`
- 열거 패턴(`_ENUM_PATTERN`) 3회 이상 감지

---

### `_extract_parent_intro(header, body_paras) -> str`
적응형 도입부 추출.

- 문단을 순서대로 누적하면서 target(600자)에 도달하면 종료
- 문단 경계를 넘어 target에 도달하지 못하면 min(350자)까지 계속
- max(800자) 도달 시 문장 경계에서 강제 cut
- 표 문단은 도입부에서 제외 (구조 파악에 부적합)

---

### `_build_child_chunks(sec, base_metadata, document_id, sec_idx) -> list[dict]`
child 청크 생성.

1. 문단을 `_preprocess_paragraph()`로 전처리 (표/텍스트 분기)
2. `group_max_len(1300)` 기준으로 조각 누적
3. 마지막 child가 `min_child_len(200)` 미만이면 이전 child에 병합
4. `chunk_position`(first/middle/last/only) 부여

---

### `_split_table_paragraph(text, max_len) -> list[str]`
표를 행 단위로 분리. 각 청크에 헤더 행 반복 포함 → 독립 해석 가능.

---

### `_split_text_paragraph(text, max_len) -> list[str]`
1. 한국어 문장 경계(`다.`, `임.`, `.`, `!`, `?`) 우선 분리
2. 여전히 길면 줄바꿈 기준 추가 분리

---

## 설계 포인트

### 1. Parent-Child 구조의 이유
- child 청크는 세부 내용에 특화 → 검색 정밀도 향상
- parent 도입부는 섹션 전체 맥락 제공 → 검색 후 컨텍스트 확장
- 두 역할을 분리해 검색과 답변 생성 모두 최적화

### 2. 적응형 도입부 추출
고정 길이 cut이 아니라 목표/최소/최대 범위 안에서 문단 경계를 지킨다. 문장이 중간에 끊기지 않는다.

### 3. 표 헤더 반복
표가 여러 청크로 나뉘어도 각 청크가 독립적으로 해석 가능하도록 헤더 행을 반복 삽입한다.

### 4. 최소 길이 필터
80자 미만 청크는 생성하지 않는다. 헤더만 있고 내용이 없는 섹션, 빈 섹션이 노이즈로 적재되는 것을 방지한다.

---

## 의존성

### 외부 라이브러리
- `re`, `pathlib`, `logging`, `typing`

---
최종 수정: 2026-03-27
관련 파일: `src/chunker.py`
---
