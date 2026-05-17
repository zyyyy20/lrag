"""Normalize web-search tool results into UI-friendly source records."""
from __future__ import annotations

from typing import Any

from ..schemas.chat import WebSource
from ..tools.types import ToolResult

WEB_SOURCE_TOOLS = {
    "tavily_search",
    "tavily_extract",
    "tavily_research",
}


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


def extract_web_sources(result: ToolResult) -> list[WebSource]:
    if result.tool not in WEB_SOURCE_TOOLS:
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
