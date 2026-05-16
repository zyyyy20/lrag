"""LangGraph checkpoint setup and thread helpers."""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver

from ..config import get_settings

_checkpointer_cm: AbstractContextManager | None = None
_checkpointer: PostgresSaver | None = None
_setup_done = False


def _checkpoint_url() -> str:
    settings = get_settings()
    if settings.langgraph_checkpoint_url:
        return settings.langgraph_checkpoint_url
    return settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def get_checkpointer() -> PostgresSaver:
    global _checkpointer_cm, _checkpointer, _setup_done
    if _checkpointer is None:
        _checkpointer_cm = PostgresSaver.from_conn_string(_checkpoint_url())
        _checkpointer = _checkpointer_cm.__enter__()
    if not _setup_done:
        _checkpointer.setup()
        _setup_done = True
    return _checkpointer


def build_thread_id(user_id: int, session_id: int) -> str:
    return f"user:{user_id}:session:{session_id}"


def build_thread_config(user_id: int, session_id: int) -> dict[str, Any]:
    return {"configurable": {"thread_id": build_thread_id(user_id, session_id)}}


def checkpoint_exists(user_id: int, session_id: int) -> bool:
    cp = get_checkpointer()
    return cp.get_tuple(build_thread_config(user_id, session_id)) is not None
