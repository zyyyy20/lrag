"""Tool registration entrypoint."""
from __future__ import annotations

from .invoice import invoice_tool
from .rag import retrieve_knowledge_base_tool
from .registry import register_tools, tool_registry


def ensure_tools_registered() -> None:
    register_tools([invoice_tool, retrieve_knowledge_base_tool])


__all__ = ["ensure_tools_registered", "tool_registry"]
