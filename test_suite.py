#!/usr/bin/env python3
"""
RAG 전체 케이스 자동 테스트 스크립트
실행: .venv/bin/python3 test_suite.py
"""
import json, time, textwrap
import requests

BASE = "http://localhost:8001"
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

# =============================================================================
# 케이스 정의
# =============================================================================
CASES = [
    {
        "id": 1, "name": "서지정보 Exact Match",
        "queries": [
            {"q": "건물 에너지 모델링 자동화 시스템 특허번호 알려줘",  "expect_type": "content", "expect_kw": ["10-2708831"]},
            {"q": "건물 에너지 모델링 특허 언제 출원했어?",            "expect_type": "content", "expect_kw": ["2023"]},
            {"q": "건물 에너지 모델링 특허 등록일은 언제야?",          "expect_type": "content", "expect_kw": []},
            {"q": "건물 에너지 모델링 자동화 특허 발명자가 누구야?",   "expect_type": "content", "expect_kw": ["김영록"]},
            {"q": "건물 에너지 모델링 특허 출원인이 누구야?",          "expect_type": "content", "expect_kw": ["나인와트"]},
        ],
    },
    {
        "id": 2, "name": "의미 검색 / 설명형",
        "queries": [
            {"q": "건물 에너지 모델링할 때 창문 정보는 어떻게 만들어?", "expect_type": "content", "expect_kw": []},
            {"q": "창호 생성 방식 설명해줘",                           "expect_type": "content", "expect_kw": []},
            {"q": "건물 외형 정보는 어떤 방식으로 생성해?",            "expect_type": "content", "expect_kw": []},
            {"q": "기준면은 어떻게 확인해?",                           "expect_type": "content", "expect_kw": []},
            {"q": "메인 창문은 어디에 배치돼?",                        "expect_type": "content", "expect_kw": []},
        ],
    },
    {
        "id": 3, "name": "청구항 / 실시예",
        "queries": [
            {"q": "청구항 1 설명해줘",                                 "expect_type": "content", "expect_kw": [], "expect_claim": [1]},
            {"q": "청구항 1에서 창문은 어떻게 처리해?",               "expect_type": "content", "expect_kw": [], "expect_claim": [1]},
            {"q": "청구항 1에서 기준면은 어떻게 쓰여?",               "expect_type": "content", "expect_kw": [], "expect_claim": [1]},
            {"q": "실시예에서는 창문을 어떻게 생성해?",               "expect_type": "content", "expect_kw": []},
            {"q": "과제의 해결 수단에서 창문 관련 내용만 설명해줘",   "expect_type": "content", "expect_kw": []},
        ],
    },
    {
        "id": 4, "name": "문서명 / 특허명 매칭",
        "queries": [
            {"q": "전력 시계열 데이터 적응형 청킹 방법 및 장치 특허 요약해줘", "expect_type": "content", "expect_kw": []},
            {"q": "전력 시계열 청킹 특허 요약해줘",                           "expect_type": "content", "expect_kw": []},
            {"q": "에너지 모델링 자동화 시스템 특허 요약해줘",               "expect_type": "content", "expect_kw": []},
            {"q": "RAG LLM 기반 리트로핏 특허 내용 알려줘",                  "expect_type": "content", "expect_kw": []},
            {"q": "에너지 분석 플랫폼 특허가 있어?",                         "expect_type": "existence", "expect_kw": []},
        ],
    },
    {
        "id": 5, "name": "메타 / 시스템",
        "queries": [
            {"q": "등록된 문서가 몇 개야?",         "expect_type": "meta", "expect_kw": []},
            {"q": "어떤 문서들이 등록되어 있어?",   "expect_type": "meta", "expect_kw": []},
            {"q": "지금 특허 문서만 몇 개야?",      "expect_type": "meta", "expect_kw": []},
            {"q": "에너지 관련 문서가 몇 개 있어?", "expect_type": "meta", "expect_kw": []},
            {"q": "현재 인덱싱된 총 청크 수가 몇 개야?", "expect_type": "content", "expect_no_docs": True},
        ],
    },
    {
        "id": 6, "name": "존재 확인",
        "queries": [
            {"q": "RAG 관련 특허 있어?",              "expect_type": "existence", "expect_kw": []},
            {"q": "건물 에너지 모델링 관련 특허 있어?","expect_type": "existence", "expect_kw": []},
            {"q": "창문 생성 관련 특허 있어?",         "expect_type": "existence", "expect_kw": []},
            {"q": "탄소 배출 관련 특허 있어?",         "expect_type": "existence", "expect_kw": [], "expect_no_docs": True},
            {"q": "리트로핏 관련 문서 있어?",          "expect_type": "existence", "expect_kw": []},
        ],
    },
    {
        "id": 7, "name": "No-Docs / 환각 방지",
        "queries": [
            {"q": "우리 회사 직원 수가 몇 명이야?",    "expect_type": "content", "expect_no_docs": True},
            {"q": "회사 점심 메뉴 규정 있어?",         "expect_type": "content", "expect_no_docs": True},
            {"q": "대표 전화번호가 뭐야?",             "expect_type": "content", "expect_no_docs": True},
            {"q": "사내 복지 포인트 지급 기준이 뭐야?","expect_type": "content", "expect_no_docs": True},
            {"q": "CEO 생일이 언제야?",                "expect_type": "content", "expect_no_docs": True},
        ],
    },
    {
        "id": 9, "name": "형식 변환 / 요약",
        "queries": [
            {"q": "에너지 모델링 특허 핵심 내용을 발표용으로 3줄 요약해줘", "expect_type": "content", "expect_kw": []},
            {"q": "이 특허 내용을 신입사원도 이해하게 쉽게 설명해줘",      "expect_type": "content", "expect_kw": []},
            {"q": "에너지 모델링 특허 핵심 키워드 5개만 뽑아줘",          "expect_type": "content", "expect_kw": []},
            {"q": "에너지 모델링 특허 항목별로 표처럼 정리해줘",          "expect_type": "content", "expect_kw": []},
            {"q": "에너지 모델링 특허 차별점만 bullet로 정리해줘",        "expect_type": "content", "expect_kw": []},
        ],
    },
]

# Case 8: 연속 대화 (별도 처리)
CONVO_CASES = [
    {
        "id": "8-1", "name": "연속 대화: 청구항 → 창문",
        "turns": [
            {"q": "건물 에너지 모델링 특허에서 청구항 1 설명해줘", "expect_reformulated": False},
            {"q": "거기서 창문은 어떻게 처리해?",                 "expect_reformulated": True},
        ],
    },
    {
        "id": "8-2", "name": "연속 대화: 특허 요약 → 핵심 아이디어",
        "turns": [
            {"q": "전력 시계열 청킹 특허 요약해줘",  "expect_reformulated": False},
            {"q": "거기서 핵심 아이디어만 말해줘",   "expect_reformulated": True},
        ],
    },
]

# =============================================================================
# 헬퍼
# =============================================================================
def chat(query: str, history: list | None = None) -> dict:
    payload = {"query": query}
    if history:
        payload["chat_history"] = history
    r = requests.post(f"{BASE}/chat", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


def check(r: dict, q_meta: dict) -> list[str]:
    """간단한 자동 판정. 이슈 문자열 리스트 반환 (빈 리스트 = 이상 없음)."""
    issues = []
    answer = r.get("answer", "")
    qtype  = r.get("query_type", "")

    # query_type 체크
    expect_type = q_meta.get("expect_type")
    if expect_type and qtype != expect_type:
        issues.append(f"type={qtype} (기대={expect_type})")

    # 키워드 존재 체크
    for kw in q_meta.get("expect_kw", []):
        if kw not in answer:
            issues.append(f"kw '{kw}' 없음")

    # llm_error 체크
    if r.get("fallback_reason") == "llm_error":
        issues.append("llm_error (LLM 호출 실패)")

    # no-docs 기대 시
    if q_meta.get("expect_no_docs"):
        no_docs_signals = ["찾을 수 없", "없습니다", "문서에 없", "없어", "등록되어 있지 않"]
        if not any(s in answer for s in no_docs_signals):
            issues.append("no-docs 기대했는데 답변 생성됨")

    # claim_numbers 체크
    if q_meta.get("expect_claim"):
        # claim_numbers는 API 응답에 없으므로 로그로만 확인 가능
        # 대신 top_distance가 낮은지 간접 확인
        td = r.get("top_distance")
        if td is not None and td > 0.60:
            issues.append(f"top_dist={td:.3f} 높음 (청구항 chunk 못 찾았을 수 있음)")

    return issues


def fmt_answer(text: str, width: int = 80) -> str:
    lines = text.strip().splitlines()
    short = " ".join(lines)[:160]
    return textwrap.fill(short + ("…" if len(" ".join(lines)) > 160 else ""), width)


def fmt_citations(citations: list) -> str:
    if not citations:
        return "(없음)"
    return " / ".join(
        f"{c.get('source_file','?')[:25]} > {c.get('header','')[:20]}"
        for c in citations[:3]
    )


# =============================================================================
# 단건 케이스 실행
# =============================================================================
def run_cases():
    total = passed = 0

    for case in CASES:
        print(f"\n{'━'*72}")
        print(f"  CASE {case['id']}: {case['name']}")
        print('━'*72)

        for i, qm in enumerate(case["queries"], 1):
            q = qm["q"]
            try:
                r = chat(q)
            except Exception as e:
                print(f"  [{case['id']}-{i}] ❌ ERROR: {e}")
                total += 1
                continue

            issues = check(r, qm)
            status = PASS if not issues else FAIL
            total += 1
            if not issues:
                passed += 1

            td    = r.get("top_distance")
            rrf   = r.get("top_rrf_score")
            qtype = r.get("query_type", "?")
            fb    = r.get("fallback_reason") or ""

            print(f"\n  [{case['id']}-{i}] {status} {q}")
            print(f"    type={qtype}  dist={f'{td:.3f}' if td else 'N/A'}  rrf={f'{rrf:.4f}' if rrf else 'N/A'}  {fb}")
            print(f"    citations: {fmt_citations(r.get('citations', []))}")
            print(f"    answer: {fmt_answer(r.get('answer',''))}")
            if issues:
                for iss in issues:
                    print(f"    {WARN}{iss}")

            time.sleep(2.0)

    return total, passed


# =============================================================================
# 연속 대화 케이스
# =============================================================================
def run_convo_cases():
    print(f"\n{'━'*72}")
    print("  CASE 8: 연속 대화 / reformulation")
    print('━'*72)

    for cv in CONVO_CASES:
        print(f"\n  [{cv['id']}] {cv['name']}")
        history: list[dict] = []

        for i, turn in enumerate(cv["turns"], 1):
            q = turn["q"]
            try:
                r = chat(q, history if history else None)
            except Exception as e:
                print(f"    턴{i} ❌ ERROR: {e}")
                break

            reformulated = r.get("reformulated_query")
            expect_ref   = turn.get("expect_reformulated", False)

            if expect_ref and not reformulated:
                ref_status = f"{FAIL} reformulation 없음 (기대했음)"
            elif not expect_ref and reformulated:
                ref_status = f"{WARN}reformulation 발생 (불필요할 수 있음): {reformulated}"
            else:
                ref_status = f"{PASS} reformulation={'있음: ' + reformulated if reformulated else '없음'}"

            td = r.get("top_distance")
            print(f"    턴{i}: {q}")
            print(f"      {ref_status}")
            print(f"      dist={f'{td:.3f}' if td else 'N/A'}  answer: {fmt_answer(r.get('answer',''), 60)}")

            history.append({"role": "user",      "content": q})
            history.append({"role": "assistant",  "content": r.get("answer", "")})
            time.sleep(2.0)


# =============================================================================
# Case 10: 토크나이저 직접 검증
# =============================================================================
def run_tokenizer_cases():
    print(f"\n{'━'*72}")
    print("  CASE 10: 토크나이저 직접 검증")
    print('━'*72)

    import sys
    sys.path.insert(0, ".")
    try:
        from src.bm25_retriever import _tokenize
    except Exception as e:
        print(f"  토크나이저 로드 실패: {e}")
        return

    checks = [
        ("출원일이 언제야?",           ["출원일"]),
        ("등록번호 알려줘",            ["등록번호"]),
        ("발명자가 누구야?",           ["발명자"]),
        ("청구항 1 설명해줘",          ["청구항", "1"]),
        ("창문 생성 방식 알려줘",      ["창문", "생성"]),
        ("기준면은 어떻게 확인해?",    ["기준면"]),
        ("RAG 관련 특허 있어?",        ["rag"]),
        ("2023.11.30 출원 문서 있어?", ["2023.11.30"]),
        ("10-2708831 번호 문서 알려줘",["10-2708831"]),
    ]

    for text, must_have in checks:
        tokens = _tokenize(text)
        missing = [m for m in must_have if m.lower() not in [t.lower() for t in tokens]]
        status  = PASS if not missing else FAIL
        print(f"  {status} '{text}'")
        print(f"       tokens={tokens}")
        if missing:
            print(f"       {WARN}누락: {missing}")


# =============================================================================
# main
# =============================================================================
def main():
    print(f"\n{'='*72}")
    print("  AutoDraft RAG 테스트 스위트")
    print(f"{'='*72}")

    try:
        h = requests.get(f"{BASE}/health", timeout=5).json()
        print(f"\n  서버 정상 | collection={h.get('collection')} | chunks={h.get('chunk_count')}\n")
    except Exception as e:
        print(f"\n  서버 연결 실패: {e}\n")
        return

    total, passed = run_cases()
    run_convo_cases()
    run_tokenizer_cases()

    print(f"\n{'='*72}")
    print(f"  결과: {passed}/{total} 통과  (자동 판정 기준 — 수동 확인 필요 항목 별도)")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
