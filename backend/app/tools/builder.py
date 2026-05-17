"""Build per-request LangChain tools with backend context closed over."""
from __future__ import annotations

from ..config import get_settings
from ..services.mcp_tools import get_cached_mcp_tools
from .invoice import (
    build_runtime_generate_conversation_invoice_tool,
)
from .rag import (
    build_runtime_retrieve_knowledge_base_tool,
)


def build_runtime_tools() -> list:
    tools = [
        build_runtime_generate_conversation_invoice_tool(),
        build_runtime_retrieve_knowledge_base_tool(),
    ]
    if get_settings().mcp_tavily_enabled:
        tools.extend(get_cached_mcp_tools())
    return tools
