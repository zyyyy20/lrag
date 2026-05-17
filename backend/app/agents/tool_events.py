"""Tool event formatting and payload parsing for streamed agent runs."""
from __future__ import annotations

import json
import logging
from typing import Any

from ..tools.types import ToolResult

logger = logging.getLogger(__name__)


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
