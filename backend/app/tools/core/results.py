"""Tool result parsing and source extraction helpers."""
from __future__ import annotations

import json
import logging
from typing import Any

from ...schemas.chat import Source, WebSource
from .types import ToolResult

logger = logging.getLogger(__name__)

WEB_SOURCE_TOOLS = {
    "tavily_search",
    "tavily_extract",
    "tavily_research",
}


def tool_result_from_payload(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool=str(payload.get("tool") or "unknown"),
        ok=bool(payload.get("ok")),
        data=dict(payload.get("data") or {}),
        message=str(payload.get("message") or ""),
        type=str(payload.get("type") or "tool_result"),
    )


def parse_tool_payload(content: Any, tool_name: str | None = None) -> ToolResult | None:
    if isinstance(content, dict):
        if "tool" in content or "data" in content or "ok" in content:
            return tool_result_from_payload(content)
        return ToolResult(
            tool=tool_name or "unknown",
            ok=True,
            data=content,
            message="",
        )
    if isinstance(content, list):
        text = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    else:
        text = str(content or "")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Unable to decode LangGraph tool payload: %s", text[:300])
        return None
    if not isinstance(payload, dict):
        return None
    if "tool" in payload or "data" in payload or "ok" in payload:
        return tool_result_from_payload(payload)
    return ToolResult(
        tool=tool_name or "unknown",
        ok=True,
        data=payload,
        message="",
    )


def extract_sources(result: ToolResult | None) -> list[Source]:
    if result is None or result.tool != "retrieve_knowledge_base":
        return []
    sources: list[Source] = []
    for item in result.data.get("sources") or []:
        try:
            sources.append(Source.model_validate(item))
        except Exception:
            logger.warning("Invalid source returned by RAG tool: %s", item)
    return sources


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _source_from_item(item: Any, *, source_tool: str) -> WebSource | None:
    if not isinstance(item, dict):
        return None
    url = _text(item.get("url"))
    if not url:
        return None
    title = _text(item.get("title")) or url
    preview = (
        _text(item.get("content"))
        or _text(item.get("raw_content"))
        or _text(item.get("text"))
    )
    return WebSource(
        title=title,
        url=url,
        content_preview=preview or None,
        score=_score(item.get("score")),
        source_tool=source_tool,
    )


def extract_web_sources(result: ToolResult | None) -> list[WebSource]:
    if result is None or result.tool not in WEB_SOURCE_TOOLS:
        return []

    items: list[Any] = []
    raw_results = result.data.get("results")
    if isinstance(raw_results, list):
        items.extend(raw_results)
    elif isinstance(raw_results, dict):
        nested = raw_results.get("results")
        if isinstance(nested, list):
            items.extend(nested)

    if result.tool == "tavily_extract" and not items:
        items.append(result.data)

    sources: list[WebSource] = []
    seen_urls: set[str] = set()
    for item in items:
        source = _source_from_item(item, source_tool=result.tool)
        if source is None or source.url in seen_urls:
            continue
        sources.append(source)
        seen_urls.add(source.url)
    return sources
