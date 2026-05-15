"""In-process registry for safe, allow-listed tools."""
from __future__ import annotations

from collections.abc import Iterable

from .base import ToolContext, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name:
            raise ValueError("Tool name is required")
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def list_available(self, ctx: ToolContext) -> list[ToolSpec]:
        available: list[ToolSpec] = []
        for tool in self._tools.values():
            if tool.is_available is None or tool.is_available(ctx):
                available.append(tool)
        return available

    def names(self) -> set[str]:
        return set(self._tools)


tool_registry = ToolRegistry()


def register_tools(specs: Iterable[ToolSpec]) -> None:
    for spec in specs:
        if spec.name not in tool_registry.names():
            tool_registry.register(spec)
