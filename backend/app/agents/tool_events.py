"""Tool event formatting and payload parsing for streamed agent runs."""
from __future__ import annotations

import json
from typing import Any

from ..tools.core.results import parse_tool_payload, tool_result_from_payload


def safe_tool_args(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def tool_display_title(tool_name: str) -> str:
    titles = {
        "retrieve_knowledge_base": "查询知识库",
        "generate_conversation_invoice": "生成对话发票",
        "tavily_search": "联网搜索",
        "tavily_extract": "读取网页",
        "tavily_crawl": "爬取网站",
        "tavily_map": "生成站点地图",
        "tavily_research": "联网研究",
    }
    return titles.get(tool_name, tool_name)


def tool_call_events(message: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    raw_calls = list(getattr(message, "tool_calls", None) or [])
    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    if isinstance(additional_kwargs, dict):
        raw_calls.extend(additional_kwargs.get("tool_calls") or [])

    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(
            call.get("name")
            or call.get("tool")
            or function.get("name")
            or "unknown"
        )
        call_id = str(call.get("id") or f"{name}:{len(events)}")
        args = call.get("args")
        if args is None:
            args = function.get("arguments")
        events.append(
            {
                "id": call_id,
                "type": "tool_call",
                "tool": name,
                "title": tool_display_title(name),
                "args": safe_tool_args(args or {}),
                "status": "running",
            }
        )
    return events
