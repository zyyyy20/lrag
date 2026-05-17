"""Build per-request LangChain tools with backend context closed over."""
from __future__ import annotations

from .local import build_local_tools
from .mcp import get_cached_mcp_tools


def build_runtime_tools() -> list:
    return [*build_local_tools(), *get_cached_mcp_tools()]
