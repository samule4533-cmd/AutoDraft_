# src/parent_store.py
#
# 역할: parent_index.json을 로드하고, child chunk 인접 context를 제공한다.
#
# ── 데이터 구조 ──────────────────────────────────────────────────────────────
# parent_index.json (dict keyed by parent_chunk_id):
#   {
#     "doc_p1": {
#       "chunk_id":   "doc_p1",
#       "header":     "## 청구항",
#       "intro_text": "헤더 + 첫 문단 요약 (350~800자)",
#       "children":   [{"chunk_id": "doc_p1_ch0", "text": "...", "metadata": {...}}, ...],
#       "metadata":   {...}
#     },
#     ...
#   }
#
# ── 인메모리 인덱스 ──────────────────────────────────────────────────────────
# _parent_map[parent_chunk_id]   = parent entry dict
# _child_to_parent[child_id]     = parent_chunk_id
# _child_order[child_id]         = child의 parent.children 리스트 내 인덱스
#
# ── 동시성 설계 ──────────────────────────────────────────────────────────────
# load() → 로컬에서 완전히 빌드 후 _store_lock 안에서 한 번에 swap
# 읽기(get_parent, get_adjacent_child_ids 등) → lock 안에서 레퍼런스 스냅샷만 복사

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = PROJECT_ROOT / "data" / "processed" / "parent_index.json"

_parent_map:      dict[str, dict[str, Any]] = {}
_child_to_parent: dict[str, str]            = {}
_child_order:     dict[str, int]            = {}

_store_lock = threading.Lock()


# =============================================================================
# 로드 (서버 시작 시 1회 호출)
# =============================================================================
def load(path: Path | str | None = None) -> None:
    """
    parent_index.json을 로드하고 인메모리 인덱스를 빌드한다.
    path가 None이면 DEFAULT_PATH를 사용한다.
    파일이 없으면 조용히 빈 상태로 유지한다.
    """
    global _parent_map, _child_to_parent, _child_order

    target = Path(path) if path else DEFAULT_PATH

    if not target.exists():
        logger.info("parent_index.json 없음 → parent_store 빈 상태 유지 (%s)", target)
        return

    try:
        data: dict[str, dict] = json.loads(target.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("parent_index.json 로드 실패: %s", e)
        return

    new_parent_map:      dict[str, dict] = {}
    new_child_to_parent: dict[str, str]  = {}
    new_child_order:     dict[str, int]  = {}

    for parent_id, pdata in data.items():
        new_parent_map[parent_id] = pdata
        for idx, child in enumerate(pdata.get("children", [])):
            cid = child["chunk_id"]
            new_child_to_parent[cid] = parent_id
            new_child_order[cid]     = idx

    with _store_lock:
        _parent_map      = new_parent_map
        _child_to_parent = new_child_to_parent
        _child_order     = new_child_order

    logger.info(
        "parent_store 로드 완료: %d 부모, %d 자식 (%s)",
        len(new_parent_map), len(new_child_to_parent), target,
    )


# =============================================================================
# 읽기 API
# =============================================================================
def get_parent(parent_chunk_id: str) -> dict[str, Any] | None:
    """parent_chunk_id로 parent entry를 반환한다. 없으면 None."""
    with _store_lock:
        pmap = _parent_map
    return pmap.get(parent_chunk_id)


def get_adjacent_child_ids(child_id: str, window: int = 1) -> list[str]:
    """
    hit child 기준 ±window 범위의 child_id 리스트를 반환한다.
    순서: [prev..., hit, next...]

    child_id가 등록되지 않은 경우 빈 리스트 반환.
    window=1: 바로 이전 + hit + 바로 다음 (최대 3개)
    """
    with _store_lock:
        c2p  = _child_to_parent
        cord = _child_order
        pmap = _parent_map

    parent_id = c2p.get(child_id)
    if parent_id is None:
        return []
    idx = cord.get(child_id, -1)
    parent = pmap.get(parent_id)
    if parent is None or idx < 0:
        return []
    children = parent.get("children", [])
    start = max(0, idx - window)
    end   = min(len(children), idx + window + 1)
    return [c["chunk_id"] for c in children[start:end]]


def get_child_text(child_id: str) -> str | None:
    """child_id의 텍스트를 parent_store에서 조회한다. 없으면 None."""
    with _store_lock:
        c2p  = _child_to_parent
        cord = _child_order
        pmap = _parent_map

    parent_id = c2p.get(child_id)
    if parent_id is None:
        return None
    idx = cord.get(child_id, -1)
    parent = pmap.get(parent_id)
    if parent is None or idx < 0:
        return None
    children = parent.get("children", [])
    if 0 <= idx < len(children):
        return children[idx].get("text")
    return None


# =============================================================================
# 쓰기 API (ingest 파이프라인에서 호출)
# =============================================================================
def _build_index_entry(p: dict[str, Any], child_map: dict[str, Any]) -> dict[str, Any]:
    """parent 청크 + child_map으로 parent_index.json 항목 1개를 생성한다."""
    children = [
        {
            "chunk_id": child_map[cid]["chunk_id"],
            "text":     child_map[cid].get("text", ""),
            "metadata": child_map[cid].get("metadata", {}),
        }
        for cid in p.get("child_ids", [])
        if cid in child_map
    ]
    return {
        "chunk_id":   p["chunk_id"],
        "header":     p.get("header", ""),
        "intro_text": p.get("intro_text", ""),
        "children":   children,
        "metadata":   p.get("metadata", {}),
    }


def _write_atomic(index: dict[str, Any], path: Path) -> None:
    """index dict를 path에 원자적으로 저장한다 (temp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        Path(tmp_path).rename(path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_index(
    parent_chunks: list[dict[str, Any]],
    child_map:     dict[str, dict[str, Any]],
    path:          Path | str | None = None,
) -> None:
    """
    전체 parent 청크 목록으로 parent_index.json을 교체 저장한다.
    company_vectordb.py 일괄 적재 후 호출한다.
    저장 후 인메모리 상태도 즉시 재로드한다.
    """
    target = Path(path) if path else DEFAULT_PATH
    index  = {p["chunk_id"]: _build_index_entry(p, child_map) for p in parent_chunks}
    _write_atomic(index, target)
    logger.info("parent_index.json 전체 저장: %d 부모 → %s", len(index), target)
    load(target)


def merge_parents(
    parent_chunks: list[dict[str, Any]],
    child_map:     dict[str, dict[str, Any]],
    path:          Path | str | None = None,
) -> None:
    """
    새 parent 청크를 기존 parent_index.json에 병합(덮어쓰기)한다.
    API 단건 ingest 후 호출한다.
    저장 후 인메모리 상태도 즉시 재로드한다.
    """
    target = Path(path) if path else DEFAULT_PATH

    # 기존 인덱스 로드
    existing: dict[str, Any] = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("기존 parent_index.json 읽기 실패 (빈 상태로 시작): %s", e)

    for p in parent_chunks:
        existing[p["chunk_id"]] = _build_index_entry(p, child_map)

    _write_atomic(existing, target)
    logger.info("parent_index.json 병합: +%d 부모 (합계 %d) → %s", len(parent_chunks), len(existing), target)
    load(target)


def remove_by_document(document_id: str, path: Path | str | None = None) -> None:
    """
    특정 document_id의 parent 항목을 parent_index.json에서 제거한다.
    API DELETE /ingest/{file_id} 후 호출한다 (document_id = file_id prefix).
    저장 후 인메모리 상태도 즉시 재로드한다.
    """
    target = Path(path) if path else DEFAULT_PATH
    if not target.exists():
        return

    try:
        existing: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("parent_index.json 읽기 실패: %s", e)
        return

    before = len(existing)
    # parent_chunk_id 형식: {document_id}_p{N} → document_id 접두사로 필터
    existing = {k: v for k, v in existing.items() if not k.startswith(f"{document_id}_")}
    removed  = before - len(existing)

    if removed == 0:
        logger.debug("remove_by_document: '%s' 항목 없음", document_id)
        return

    _write_atomic(existing, target)
    logger.info("parent_index.json 제거: %d 부모 삭제 (document_id=%s) → %s", removed, document_id, target)
    load(target)
