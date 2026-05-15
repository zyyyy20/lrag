"""Typed primitives for server-side chat tools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from sqlalchemy.orm import Session


@dataclass(frozen=True)
class ToolContext:
    """Request-scoped context injected by the backend, never by the model."""

    db: Session
    user_id: int
    session_id: int


@dataclass(frozen=True)
class ToolResult:
    """Stable result shape returned to the API and frontend."""

    tool: str
    ok: bool
    data: dict[str, Any]
    message: str
    type: str = "tool_result"

    def model_dump(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "tool": self.tool,
            "ok": self.ok,
            "data": self.data,
            "message": self.message,
        }


class ToolHandler(Protocol):
    def __call__(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        ...


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def prompt_description(self) -> str:
        return (
            f"- {self.name}: {self.description}\n"
            f"  Parameters JSON schema: {self.parameters}"
        )
