"""Unified access point for tools exposed to the LangGraph agent."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..local import build_local_tools
from ..mcp import get_cached_mcp_tools

ToolProvider = Callable[[], list[Any]]


@dataclass(frozen=True)
class ToolProviderSpec:
    name: str
    category: str
    provider: ToolProvider


class ToolManager:
    """Collect enabled tools from local and external providers."""

    def __init__(
        self,
        *,
        local_provider: ToolProvider | None = None,
        mcp_provider: ToolProvider | None = None,
    ) -> None:
        self._local_provider = local_provider or build_local_tools
        self._mcp_provider = mcp_provider or get_cached_mcp_tools
        self._providers = [
            ToolProviderSpec("local", "built_in", self._local_provider),
            ToolProviderSpec("mcp", "external", self._mcp_provider),
        ]

    def list_providers(self) -> list[ToolProviderSpec]:
        return list(self._providers)

    def get_agent_tools(self) -> list[Any]:
        tools: list[Any] = []
        for provider in self._providers:
            tools.extend(provider.provider())
        return tools


def build_agent_tools() -> list[Any]:
    return ToolManager(
        local_provider=build_local_tools,
        mcp_provider=get_cached_mcp_tools,
    ).get_agent_tools()
