# src/query_processor.py
#
# 역할: 원본 사용자 질문을 받아 "어떻게 처리할지" 결정하는 전처리 레이어.
# rag_chain.py는 이 모듈의 결과(QueryContext)를 받아 RAG 실행만 담당한다.
#
# 처리 순서:
#   1. history trim      — 오래된 대화 제거
#   2. reformulation     — 지시어 포함 시 맥락 기반 재작성 (조건부)
#   3. understanding     — 구어체/짧은 질문을 검색 최적화 쿼리로 변환 (조건부)
#   4. routing           — 질문 유형 분류 (meta / existence / content)

import logging
import os
import re
from typing import Literal

from dotenv import load_dotenv
from google.genai import types
from pydantic import BaseModel

from src.llm_api import get_client

load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================
PROCESSOR_CONFIG = {
    "gemini_model":       os.getenv("GEMINI_RAG_MODEL", "gemini-2.0-flash"),
    "understanding_model": os.getenv("GEMINI_UNDERSTANDING_MODEL", "gemini-2.0-flash"),
    "history_max_turns":  5,
}

# =============================================================================
# 출력 타입
# =============================================================================
QueryType = Literal["meta", "existence", "content", "greeting"]


class QueryContext(BaseModel):
    original_query: str          # 사용자가 입력한 원본 질문 (프롬프트 생성에 사용)
    search_query:   str          # 실제 벡터 검색에 사용할 쿼리
    reformulated:   str | None   # reformulation 발생 시 재작성 결과, 없으면 None
    understood:     str | None   # understanding 발생 시 검색 최적화 결과, 없으면 None
    query_type:     QueryType    # "meta" | "existence" | "content"


# =============================================================================
# 프롬프트 상수
# =============================================================================
_REFORMULATION_PROMPT = """\
아래는 사용자와 어시스턴트의 이전 대화다.
마지막 [현재 질문]에 지시어(그럼, 거기, 그건, 해당 등)가 포함되어 있어 맥락 없이는 의미가 불분명하다.

[이전 대화]
{history}

[현재 질문]
{query}

위 대화 맥락을 참고해 [현재 질문]을 맥락 없이도 이해할 수 있는 독립적인 질문으로 재작성하라.
재작성된 질문만 출력하고 다른 설명은 하지 말라.
"""

_UNDERSTANDING_PROMPT = """\
사용자가 사내 특허·기술 문서 검색 시스템에 아래 질문을 입력했다.
이 질문을 벡터 검색에 최적화된 형태로 변환하라.

[규칙]
1. 구어체, 조사, 불필요한 표현 제거
2. 핵심 기술 용어와 개념 중심으로 재작성
3. 날짜/번호/이름 등 정형 필드를 묻는 질문이면 해당 필드명을 명시적으로 포함
   - 특히 "청구항 N", "제N항" 같은 특허 청구항 번호는 반드시 "청구항 N" 형태(숫자 그대로)로 보존하라
4. 동의어·유사어를 병기해 검색 범위를 넓혀라. 예: "창호" → "창호 창문", "냉방" → "냉방 에어컨 냉각"
5. 한 문장으로 출력. 다른 설명 없이 변환된 문장만 출력
6. 질문이 개인 감정/식사/날씨/일상 잡담이고 특허·기술 키워드가 단 하나도 없을 때만
   "[[CHITCHAT]]" 을 그대로 출력하라 (조금이라도 업무/기술 관련이면 변환 진행)

[질문]
{query}

[변환된 검색 쿼리]
"""

# =============================================================================
# 패턴 상수
# =============================================================================
# 청구항 번호 정규화: "제N항", "N항" → "청구항 N"
# BM25 인덱스에는 "청구항 1" 형태로 저장되어 있으므로 검색 쿼리도 동일 형태로 통일한다.
# "제1항", "1항", "제 1항" 등 다양한 표현을 커버하되
# "1항목", "2항목" 처럼 뒤에 다른 문자가 이어지는 경우는 제외한다.
_CLAIM_NUM_PATTERN = re.compile(r"제\s*(\d+)\s*항|(?<!\w)(\d+)\s*항(?!\w)")


def _normalize_claim_terms(text: str) -> str:
    """
    특허 청구항 번호 표현을 BM25 인덱스 형태와 일치하도록 정규화한다.

    "제N항", "N항" → "청구항 N"
    "청구항 N항", "청구항 제N항" → "청구항 N" (중복 제거)
    """
    # "제N항" → "청구항 N"
    text = re.sub(r"제\s*(\d+)\s*항(?!\w)", r"청구항 \1", text)
    # "N항" (앞에 단어 없는 경우) → "청구항 N"
    text = re.sub(r"(?<!\w)(\d+)\s*항(?!\w)", r"청구항 \1", text)
    # "청구항 청구항 N" 중복 정리
    text = re.sub(r"청구항\s+청구항\s+(\d+)", r"청구항 \1", text)
    return text


# reformulation: 지시어 포함 여부 감지
_REFERENTIAL_PATTERN = re.compile(r"그럼|거기|그건|그것|해당|방금|그거|그 ")

# understanding: 구어체 어미 감지 (이 패턴 OR 짧은 질문이면 실행)
_COLLOQUIAL_PATTERN = re.compile(
    r"알려줘|알려주세요|뭐야|뭐지|어때|있어|있나|있나요|해줘|가르쳐|얼마야|언제야|어디야|뭐가|어떻게"
)

# routing — greeting/chitchat: 인사·잡담·일상 표현 (벡터 검색 없이 바로 chitchat으로)
# understanding 전에 체크해야 LLM이 업무 맥락으로 오매핑하는 것을 막는다.
_GREETING_PATTERN = re.compile(
    # 인사
    r"^(안녕|하이|헬로|ㅎㅇ|hi|hello|hey|반가워|반갑|고마워|감사|ㄱㅅ|ㅊㅋ|축하|잘있어|잘가|bye|굿모닝|좋은\s*아침|뭐해|뭐하고있어|심심|놀자)"
    r"\s*[~!?ㅎㅋ]*$"
    # 감정·컨디션 (단독 표현)
    r"|^(피곤|힘들|졸려|졸리|지쳐|지침|배고파|배고프|배불러|기뻐|슬퍼|슬프|화나|짜증|우울|설레|두근|행복|즐거|재밌|재미없|지루|답답)"
    r"[아어워해다]?\s*[~!?ㅎㅋㅠㅜ]*$"
    # 식사·메뉴 관련
    r"|.{0,6}(저녁|점심|아침|밥|메뉴)\s*(뭐|추천|먹지|먹을까|먹어|알려줘|추천해)\s*[~!?]*$"
    r"|^(뭐\s*먹|오늘\s*뭐\s*먹|점심\s*뭐|저녁\s*뭐).{0,10}$"
    # 날씨·일상
    r"|^오늘\s*(날씨|기온|비\s*와|눈\s*와).{0,10}$"
    r"|^(주말|오늘|내일).{0,6}(뭐\s*해|뭐\s*하|계획|놀자).{0,6}$",
    re.IGNORECASE,
)

# routing — meta: DB 전체 현황을 묻는 질문
_META_PATTERN = re.compile(
    r"(파일|문서).{0,6}(몇\s*개|갯수|개수|목록|리스트|뭐|무엇|어떤)"
    r"|(갯수|개수).{0,6}(파일|문서)"
    r"|어떤\s*(파일|문서)"
    r"|뭐\s*(들어|있어|있나).{0,4}(파일|문서)"
    r"|(파일|문서)\s*(뭐|무엇|어떤)\s*(있|들어)"
)

# routing — existence: 특정 주제 자료 존재 여부를 묻는 질문
_EXISTENCE_PATTERN = re.compile(
    r".{1,20}(관련|관한)\s*(특허|자료|문서|파일)?[가은는이]?\s*(있어|있나|있나요|있습니까)\??"
    r"|.{1,20}(특허|자료|문서|파일)[가은는이]?\s*(있어|있나|있나요)\??"
)

# routing — 토픽 필터 목록 질문: "에너지 관련 파일 뭐있어" 같이
# meta 패턴에도 걸리지만 실제로는 토픽 필터 검색이어야 하는 쿼리
# meta보다 먼저 체크해서 existence로 라우팅한다.
_TOPIC_LIST_PATTERN = re.compile(
    r".{1,15}(관련|관한)\s*.{0,5}(파일|문서)\s*(뭐|무엇|어떤|몇)"
)


# =============================================================================
# Step 0 — history trim
# =============================================================================
def trim_history(history: list, max_turns: int = PROCESSOR_CONFIG["history_max_turns"]) -> list:
    """
    chat_history를 최근 max_turns 턴만 유지한다.
    1턴 = user + assistant 한 쌍 = 메시지 2개.
    role 구조 검증은 API 입력 스키마(Pydantic)에서 강제하므로 여기선 슬라이싱만 한다.
    """
    return history[-(max_turns * 2):]


# =============================================================================
# Step 1 — Query Reformulation (조건부)
# =============================================================================
def should_reformulate(query: str, history: list | None) -> bool:
    """
    지시어가 포함되어 있고 이전 대화가 있을 때만 True 반환.
    둘 중 하나라도 없으면 LLM 호출 없이 스킵한다.
    """
    if not history:
        return False
    return bool(_REFERENTIAL_PATTERN.search(query))


def reformulate_query(query: str, history: list, req_id: str) -> str:
    """
    이전 대화 맥락을 참고해 지시어가 포함된 질문을 독립적인 형태로 재작성한다.
    실패 시 원본 query 반환 (fallback 필수 — 이 단계 실패가 전체 흐름을 막아선 안 된다).
    """
    history_lines = [
        f"{'사용자' if m.get('role') == 'user' else '어시스턴트'}: {m.get('content', '')}"
        for m in history
    ]
    prompt = _REFORMULATION_PROMPT.format(
        history="\n".join(history_lines),
        query=query,
    )
    try:
        resp = get_client().models.generate_content(
            model=PROCESSOR_CONFIG["gemini_model"],
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        result = (getattr(resp, "text", "") or "").strip()
        if not result or len(result) < 5:
            logger.warning("[%s] reformulation 비정상 → 원본 사용: %r", req_id, result)
            return query
        logger.info("[%s] reformulation: %r → %r", req_id, query, result)
        return result
    except Exception as e:
        logger.warning("[%s] reformulation 실패 → 원본 사용: %s", req_id, e)
        return query


# =============================================================================
# Step 2 — Query Understanding (조건부)
# =============================================================================
def should_understand(query: str) -> bool:
    """
    구어체 어미가 감지되거나 질문이 짧을 때만 True 반환.

    조건부로 실행하는 이유:
      이미 기술 용어가 충분히 포함된 긴 질문은 그대로 검색해도 품질이 충분하다.
      모든 질문에 적용하면 LLM 호출이 2배가 되어 비용/지연이 쌓인다.

    15자 기준 이유:
      "에너지 사용량은?" (9자), "출원일 알려줘" (8자) 같은 짧은 질문은
      벡터 검색 정밀도가 낮아 변환이 필요하다.
    """
    if len(query) < 15:
        return True
    return bool(_COLLOQUIAL_PATTERN.search(query))


def understand_query(query: str, req_id: str) -> str:
    """
    구어체 질문을 벡터 검색에 최적화된 키워드 중심 문장으로 변환한다.
    실패 시 원본 query 반환.
    """
    prompt = _UNDERSTANDING_PROMPT.format(query=query)
    try:
        resp = get_client().models.generate_content(
            model=PROCESSOR_CONFIG["understanding_model"],
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        result = (getattr(resp, "text", "") or "").strip()
        if not result or len(result) < 5:
            logger.warning("[%s] understanding 비정상 → 원본 사용: %r", req_id, result)
            return query
        if "[[CHITCHAT]]" in result:
            logger.info("[%s] understanding: %r → chitchat 감지", req_id, query)
            return "[[CHITCHAT]]"
        logger.info("[%s] understanding: %r → %r", req_id, query, result)
        return result
    except Exception as e:
        logger.warning("[%s] understanding 실패 → 원본 사용: %s", req_id, e)
        return query


# =============================================================================
# Step 3 — Query Routing
# =============================================================================
def classify_query(query: str) -> QueryType:
    """
    질문 유형을 패턴 기반으로 분류한다. LLM 호출 없이 처리한다.

    meta      : DB 전체 현황 질문 ("파일 몇 개야?", "어떤 문서들 있어?")
    existence : 특정 주제 자료 존재 확인 ("에너지 관련 특허 있어?")
    content   : 문서 내용 기반 질의응답 (일반 RAG 대상)

    패턴 기반을 선택한 이유:
      라우팅은 단순 분류라 LLM이 필요없다.
      LLM 호출 시 매 질문마다 추가 지연/비용이 발생한다.
    """
    # 인사·잡담은 가장 먼저 잡아 벡터 검색 없이 바로 chitchat으로 보낸다.
    # understanding이 "하이"를 기술 용어로 오해하는 오분류도 방지한다.
    if _GREETING_PATTERN.match(query.strip()):
        return "greeting"
    # "에너지 관련 파일 뭐있어" 같이 토픽 수식어가 붙은 목록 질문은
    # meta 패턴보다 먼저 잡아 existence로 라우팅한다.
    # meta는 전체 목록 반환이라 토픽 필터가 무시되기 때문.
    if _TOPIC_LIST_PATTERN.search(query):
        return "existence"
    if _META_PATTERN.search(query):
        return "meta"
    if _EXISTENCE_PATTERN.search(query):
        return "existence"
    return "content"


# =============================================================================
# 메인 진입점
# =============================================================================
def process_query(
    query: str,
    chat_history: list | None = None,
    req_id: str = "",
) -> QueryContext:
    """
    원본 질문을 받아 QueryContext를 반환한다.
    rag_chain.ask()는 이 결과를 보고 처리 경로를 결정한다.

    반환값 필드 의미:
      original_query : 프롬프트 생성 시 사용 (자연어 원본 유지)
      search_query   : ChromaDB 벡터 검색 시 사용 (최적화된 쿼리)
      reformulated   : 지시어 재작성 발생 여부 (디버깅)
      understood     : 검색 최적화 변환 발생 여부 (디버깅)
      query_type     : 라우팅 결정 기준
    """
    # 1. history trim
    history = trim_history(chat_history) if chat_history else None

    # 2. reformulation (조건부)
    reformulated: str | None = None
    if should_reformulate(query, history):
        result = reformulate_query(query, history, req_id)
        if result != query:
            reformulated = result

    base_query = reformulated if reformulated else query

    # 3. routing — understanding 전에 먼저 분류
    # meta/existence는 understanding을 스킵해 LLM 호출을 줄인다.
    # understanding 이후 분류하면 "있어/있나" 키워드가 제거돼 오분류 위험도 있음.
    query_type = classify_query(base_query)

    # 4. understanding (content만 실행)
    understood: str | None = None
    if query_type == "content" and should_understand(base_query):
        result = understand_query(base_query, req_id)
        if result == "[[CHITCHAT]]":
            # understanding 단계에서 잡담 감지 → greeting으로 전환
            query_type = "greeting"
        elif result != base_query:
            understood = result

    # 5. 청구항 번호 정규화 — understanding 전후 모두 적용
    # "제N항", "N항" → "청구항 N" 으로 통일해 BM25 인덱스 형태와 일치시킨다.
    # understanding이 실행됐으면 understood 결과에, 아니면 base_query에 적용한다.
    raw_search = understood if understood else base_query
    normalized = _normalize_claim_terms(raw_search)
    if normalized != raw_search:
        logger.info("[%s] claim 정규화: %r → %r", req_id, raw_search, normalized)
        if understood is not None:
            understood = normalized
        # base_query 자체에서 왔을 경우 search_query에만 반영 (original_query는 유지)
    search_query = normalized

    return QueryContext(
        original_query=query,
        search_query=search_query,
        reformulated=reformulated,
        understood=understood,
        query_type=query_type,
    )
