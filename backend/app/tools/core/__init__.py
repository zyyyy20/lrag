"""Core tool contracts and orchestration helpers."""
from __future__ import annotations

from .manager import ToolManager, ToolProviderSpec, build_agent_tools
from .results import (
    WEB_SOURCE_TOOLS,
    extract_sources,
    extract_web_sources,
    parse_tool_payload,
    tool_result_from_payload,
)
from .runtime import (
    clear_tool_context,
    get_tool_context,
    register_tool_context,
    set_tool_context,
    unregister_tool_context,
)
from .types import ToolContext, ToolResult

__all__ = [
    "ToolContext",
    "ToolManager",
    "ToolProviderSpec",
    "ToolResult",
    "WEB_SOURCE_TOOLS",
    "build_agent_tools",
    "clear_tool_context",
    "extract_sources",
    "extract_web_sources",
    "get_tool_context",
    "parse_tool_payload",
    "register_tool_context",
    "set_tool_context",
    "tool_result_from_payload",
    "unregister_tool_context",
]
