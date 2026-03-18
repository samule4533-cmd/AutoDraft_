"""
threshold_experiment.py
=======================
ChromaDB에 다양한 쿼리를 던져 raw cosine distance 분포를 측정하고
최적 threshold를 추천한다. LLM 호출 없이 retrieval 레이어만 테스트.

실행:
    cd tests && uv run python threshold_experiment.py
"""

import io
import os
import statistics
import sys
from pathlib import Path

# tests/ 에서 실행해도 src/ 모듈을 찾을 수 있도록 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vector_db import query_collection, get_chroma_dir

# Windows 터미널 한글/특수문자 출력 보장
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# 실험 쿼리 목록
# 레이블: "relevant" = 문서에 있어야 할 질문
#         "irrelevant" = 문서에 없는 질문 (fallback 유도)
#         "ambiguous"  = 키워드만 있는 모호한 질문
# =============================================================================
TEST_QUERIES = [
    # ── relevant: 특허 관련 구체적 질문 ──────────────────────────────────────
    ("relevant", "건물 에너지 모델링 자동화 시스템의 구성요소는 무엇인가요?"),
    ("relevant", "에너지 효율화 대상 건물을 선정하는 방법의 단계를 설명해줘"),
    ("relevant", "건물 에너지 모델링 자동화 시스템의 출원번호는?"),
    ("relevant", "건물 에너지 모델링 자동화 시스템 특허의 출원일은 언제야?"),
    ("relevant", "인공지능 모델 기반 건물의 에너지 사용량 추론 특허 등록번호"),
    ("relevant", "RAG LLM 기반 리트로핏 대상물의 에너지 운영 시뮬레이션 장치 특허"),
    ("relevant", "전력 시계열 데이터 적응형 청킹 방법이란?"),
    ("relevant", "전력 시스템 상황 기반 계층적 LLM 에이전트 라우팅 방법"),
    ("relevant", "건물 에너지 분석 플랫폼 RAG 질의응답 서비스 특허 출원인"),
    ("relevant", "주소지의 에너지 사용량 평가 정보를 결정하는 방법"),
    ("relevant", "에너지 효율화 건물 선정 서버 특허 청구항"),
    ("relevant", "건물 에너지 모델링 특허의 발명자 이름"),
    ("relevant", "에너지 사용량 추론 솔루션의 핵심 알고리즘"),
    ("relevant", "전력 시계열 적응형 청킹 장치의 구조"),
    ("relevant", "리트로핏 에너지 시뮬레이션 장치 특허의 청구항 수"),

    # ── relevant: 약간 다른 표현으로 같은 내용 ───────────────────────────────
    ("relevant", "건물 에너지 절약 방법 특허"),
    ("relevant", "에너지 효율 건물 선정 알고리즘"),
    ("relevant", "자연어 질의응답 에너지 플랫폼"),
    ("relevant", "LLM 에이전트 라우팅 전력 시스템"),
    ("relevant", "에너지 사용량 평가 서버"),

    # ── ambiguous: 키워드만, 어느 특허인지 불명확 ────────────────────────────
    ("ambiguous", "에너지 사용량"),
    ("ambiguous", "건물 에너지"),
    ("ambiguous", "특허 청구항"),
    ("ambiguous", "인공지능 모델"),
    ("ambiguous", "LLM"),
    ("ambiguous", "청킹"),
    ("ambiguous", "임베딩"),
    ("ambiguous", "RAG"),

    # ── irrelevant: 문서에 없는 내용 ────────────────────────────────────────
    ("irrelevant", "서울 날씨 알려줘"),
    ("irrelevant", "직원 복지 제도는 어떻게 되나요?"),
    ("irrelevant", "인사평가 기준을 알고 싶어"),
    ("irrelevant", "회사 식당 메뉴가 뭐야?"),
    ("irrelevant", "파이썬으로 피보나치 수열 구현하는 법"),
    ("irrelevant", "삼성전자 주가 알려줘"),
    ("irrelevant", "오늘 점심 뭐 먹을까"),
    ("irrelevant", "복리후생 정책"),
    ("irrelevant", "연차 신청 방법"),
    ("irrelevant", "영업팀 연락처"),
    ("irrelevant", "회사 주소가 어디야?"),
    ("irrelevant", "채용 공고 어디서 봐?"),
]

THRESHOLDS_TO_TEST = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
N_RESULTS = 10


def run_experiment():
    chroma_dir = str(get_chroma_dir())
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "ninewatt_company")

    print(f"\n{'='*70}")
    print(f"Threshold 실험: collection={collection_name}")
    print(f"쿼리 수: {len(TEST_QUERIES)}개 | n_results={N_RESULTS}")
    print(f"{'='*70}\n")

    # 각 쿼리별 top-1 distance 및 top-3 avg 기록
    results_by_label = {"relevant": [], "irrelevant": [], "ambiguous": []}
    all_query_data = []

    for label, query in TEST_QUERIES:
        raw = query_collection(
            query_text=query,
            collection_name=collection_name,
            persist_dir=chroma_dir,
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
            n_results=N_RESULTS,
        )
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]

        if not distances:
            print(f"  [경고] 결과 없음: {query[:50]}")
            continue

        top1 = distances[0]
        top3_avg = statistics.mean(distances[:3]) if len(distances) >= 3 else distances[0]
        top_header = (metadatas[0] or {}).get("header", "") if metadatas else ""

        results_by_label[label].append({
            "query": query,
            "top1": top1,
            "top3_avg": top3_avg,
            "top_header": top_header,
            "all_distances": distances,
        })
        all_query_data.append((label, query, top1, top3_avg, top_header))

    # ── 1. 쿼리별 상세 결과 출력 ─────────────────────────────────────────────
    for label in ["relevant", "irrelevant", "ambiguous"]:
        label_tag = {"relevant": "[O]", "irrelevant": "[X]", "ambiguous": "[?]"}[label]
        print(f"\n{label_tag}  [{label.upper()}] 쿼리 ({len(results_by_label[label])}개)")
        print("-" * 70)
        print(f"  {'top1':>6}  {'top3avg':>7}  {'질문':<40}  top 청크 헤더")
        print("-" * 70)
        for r in sorted(results_by_label[label], key=lambda x: x["top1"]):
            q_short = r["query"][:38]
            h_short = r["top_header"][:30] if r["top_header"] else "(헤더없음)"
            print(f"  {r['top1']:>6.4f}  {r['top3_avg']:>7.4f}  {q_short:<40}  {h_short}")

    # ── 2. threshold별 pass/fail 분석 ────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("THRESHOLD 분석 (top-1 distance 기준)")
    print(f"{'='*70}")
    print(f"  {'threshold':>10}  {'rel_pass':>8}  {'irr_pass':>8}  {'amb_pass':>8}  {'정밀도':>6}  {'재현율':>6}  {'F1':>6}  판정")
    print("-" * 70)

    best_f1 = 0.0
    best_threshold = None

    for thr in THRESHOLDS_TO_TEST:
        # relevant: 통과해야 정상
        rel_data = results_by_label["relevant"]
        rel_pass = sum(1 for r in rel_data if r["top1"] <= thr)
        rel_total = len(rel_data)

        # irrelevant: 차단해야 정상 (통과하면 오탐)
        irr_data = results_by_label["irrelevant"]
        irr_pass = sum(1 for r in irr_data if r["top1"] <= thr)
        irr_total = len(irr_data)

        # ambiguous: 통과 수만 기록 (판단 보류)
        amb_data = results_by_label["ambiguous"]
        amb_pass = sum(1 for r in amb_data if r["top1"] <= thr)
        amb_total = len(amb_data)

        # precision: 통과한 것 중 relevant 비율
        total_pass = rel_pass + irr_pass
        precision = rel_pass / total_pass if total_pass > 0 else 0.0

        # recall: relevant 중 통과 비율
        recall = rel_pass / rel_total if rel_total > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

        marker = " ◀ best F1" if thr == best_threshold else ""
        print(
            f"  {thr:>10.2f}  "
            f"{rel_pass:>3}/{rel_total:<3}   "
            f"{irr_pass:>3}/{irr_total:<3}   "
            f"{amb_pass:>3}/{amb_total:<3}   "
            f"{precision:>6.3f}  "
            f"{recall:>6.3f}  "
            f"{f1:>6.3f}"
            f"{marker}"
        )

    # ── 3. 레이블별 distance 통계 ─────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("레이블별 top-1 distance 통계")
    print(f"{'='*70}")
    for label in ["relevant", "irrelevant", "ambiguous"]:
        vals = [r["top1"] for r in results_by_label[label]]
        if not vals:
            continue
        mn  = min(vals)
        mx  = max(vals)
        avg = statistics.mean(vals)
        med = statistics.median(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  [{label:>10}]  min={mn:.4f}  max={mx:.4f}  avg={avg:.4f}  median={med:.4f}  std={std:.4f}  n={len(vals)}")

    # ── 4. 최종 권고 ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("최종 권고")
    print(f"{'='*70}")
    print(f"  F1 기준 최적 threshold: {best_threshold} (F1={best_f1:.3f})")

    rel_vals = [r["top1"] for r in results_by_label["relevant"]]
    irr_vals = [r["top1"] for r in results_by_label["irrelevant"]]

    if rel_vals and irr_vals:
        gap = min(irr_vals) - max(rel_vals)
        print(f"  relevant max top1   : {max(rel_vals):.4f}")
        print(f"  irrelevant min top1 : {min(irr_vals):.4f}")
        print(f"  두 그룹 사이 gap    : {gap:.4f}")
        if gap > 0:
            mid = max(rel_vals) + gap / 2
            print(f"  gap 중간값 (권고)   : {mid:.4f}")
            print(f"\n  → 보수적(환각 방지 우선): {max(rel_vals) + gap * 0.3:.4f}")
            print(f"  → 중립:                  {mid:.4f}")
            print(f"  → 관대(재현율 우선):     {max(rel_vals) + gap * 0.7:.4f}")
        else:
            print("  ⚠ relevant/irrelevant 분리 불명확 — 청킹 또는 임베딩 품질 점검 필요")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    run_experiment()
