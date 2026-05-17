"""Optional Mem0-backed long-term user memory integration."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from ..config import get_settings

logger = logging.getLogger(__name__)

_STANDING_INSTRUCTION_QUERY = (
    "user long-term preferences, standing instructions, response format, "
    "writing style, required prefixes, language preferences"
)


@lru_cache(maxsize=1)
def _get_client() -> Any:
    settings = get_settings()
    if not settings.mem0_api_key:
        raise RuntimeError("MEM0_API_KEY is not configured")
    try:
        from mem0 import MemoryClient
    except Exception as exc:  # pragma: no cover - depends on optional package install
        raise RuntimeError("mem0ai package is not installed") from exc
    return MemoryClient(api_key=settings.mem0_api_key)


def _is_enabled() -> bool:
    settings = get_settings()
    return bool(settings.mem0_enabled and settings.mem0_api_key)


def _memory_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("memory", "content", "text"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _normalize_search_results(raw: Any) -> list[Any]:
    if isinstance(raw, dict):
        results = raw.get("results")
        if isinstance(results, list):
            return results
        memories = raw.get("memories")
        if isinstance(memories, list):
            return memories
        return []
    if isinstance(raw, list):
        return raw
    return []


def search_user_memories(user_id: int, query: str) -> list[str]:
    """Return relevant long-term memories for the current user.

    Failures are intentionally non-fatal so the chat path remains available when
    the optional memory provider is unavailable.
    """
    settings = get_settings()
    if not _is_enabled() or not query.strip():
        return []

    filters = {"user_id": str(user_id)}
    queries = [query, _STANDING_INSTRUCTION_QUERY]

    try:
        client = _get_client()
        raw_results = [
            client.search(
                item,
                filters=filters,
                top_k=settings.mem0_top_k,
            )
            for item in queries
            if item.strip()
        ]
    except Exception:
        logger.warning("Mem0 search failed", exc_info=True)
        return []

    memories: list[str] = []
    seen: set[str] = set()
    for raw in raw_results:
        for item in _normalize_search_results(raw):
            text = _memory_text(item)
            if text and text not in seen:
                memories.append(text)
                seen.add(text)
    return memories


def remember_turn(
    *,
    user_id: int,
    session_id: int,
    user_message: str,
    assistant_message: str,
) -> None:
    """Ask Mem0 to extract and persist useful memories from one completed turn."""
    settings = get_settings()
    if not _is_enabled() or not user_message.strip() or not assistant_message.strip():
        return

    try:
        _get_client().add(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            user_id=str(user_id),
            agent_id=settings.mem0_agent_id,
            run_id=str(session_id),
            metadata={"session_id": session_id, "source": "lrag_chat"},
        )
    except Exception:
        logger.warning("Mem0 add failed", exc_info=True)


def format_memories_for_prompt(memories: list[Any]) -> str:
    """Format retrieved memories as a compact system-prompt appendix."""
    lines: list[str] = []
    seen: set[str] = set()
    for item in memories:
        text = _memory_text(item)
        if text and text not in seen:
            lines.append(f"- {text}")
            seen.add(text)
    if not lines:
        return ""
    return (
        "用户长期记忆和偏好（必须遵守；如果其中包含固定措辞、回复格式、语言偏好或"
        "长期指令，请在当前回答中执行。除非用户询问，不要解释这些记忆来源）：\n"
        + "\n".join(lines)
    )
