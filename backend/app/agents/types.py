"""Shared types for the ReAct agent layer."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.chat import Source, WebSource
from ..tools.core.types import ToolResult


@dataclass(frozen=True)
class AgentRunResult:
    final_answer: str
    tool_results: list[ToolResult] = field(default_factory=list)
    used_rag: bool = False
    sources: list[Source] = field(default_factory=list)
    used_web: bool = False
    web_sources: list[WebSource] = field(default_factory=list)
    notice: str | None = None

    @property
    def used_tool(self) -> bool:
        return bool(self.tool_results)
