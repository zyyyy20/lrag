"""Request-scoped tool context for singleton LangChain tools."""
from __future__ import annotations

from contextvars import ContextVar, Token
from threading import RLock
from typing import Any

from .types import ToolContext

_current_tool_context: ContextVar[ToolContext | None] = ContextVar(
    "current_tool_context",
    default=None,
)
_tool_contexts: dict[str, ToolContext] = {}
_tool_contexts_lock = RLock()


def set_tool_context(ctx: ToolContext) -> Token[ToolContext | None]:
    return _current_tool_context.set(ctx)


def clear_tool_context() -> None:
    _current_tool_context.set(None)


def register_tool_context(key: str, ctx: ToolContext) -> None:
    with _tool_contexts_lock:
        _tool_contexts[key] = ctx


def unregister_tool_context(key: str) -> None:
    with _tool_contexts_lock:
        _tool_contexts.pop(key, None)


def _context_key_from_config(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    key = configurable.get("tool_context_key") or configurable.get("thread_id")
    return str(key) if key else None


def get_tool_context(config: dict[str, Any] | None = None) -> ToolContext:
    ctx = _current_tool_context.get()
    if ctx is not None:
        return ctx

    key = _context_key_from_config(config)
    if key:
        with _tool_contexts_lock:
            ctx = _tool_contexts.get(key)
        if ctx is not None:
            return ctx

    raise RuntimeError("Tool context is not bound to the current request")
