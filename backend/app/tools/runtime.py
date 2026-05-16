"""Request-scoped tool context for singleton LangChain tools."""
from __future__ import annotations

from contextvars import ContextVar, Token

from .base import ToolContext

_current_tool_context: ContextVar[ToolContext | None] = ContextVar(
    "current_tool_context",
    default=None,
)


def set_tool_context(ctx: ToolContext) -> Token[ToolContext | None]:
    return _current_tool_context.set(ctx)


def reset_tool_context(token: Token[ToolContext | None]) -> None:
    _current_tool_context.reset(token)


def clear_tool_context() -> None:
    _current_tool_context.set(None)


def get_tool_context() -> ToolContext:
    ctx = _current_tool_context.get()
    if ctx is None:
        raise RuntimeError("Tool context is not bound to the current request")
    return ctx
