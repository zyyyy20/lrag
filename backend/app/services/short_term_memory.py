"""Short-term conversation memory helpers."""
from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Message
from ..models import Session as ChatSession
from .llm import get_llm

logger = logging.getLogger(__name__)


def _format_summary(summary: str | None) -> dict[str, str] | None:
    text = (summary or "").strip()
    if not text:
        return None
    return {"role": "system", "content": f"当前会话摘要：\n{text}"}


def _is_current_user_message(message: Message, current_user_message: str) -> bool:
    return message.role == "user" and (message.content or "") == current_user_message


def compose_short_term_messages(
    *,
    summary: str | None,
    messages: Sequence[Message],
    current_user_message: str,
    recent_turns: int,
) -> list[dict[str, str]]:
    """Compose summary, recent turns, and current user input for the agent."""
    rows = list(messages)
    current = {"role": "user", "content": current_user_message}
    if rows and _is_current_user_message(rows[-1], current_user_message):
        history_rows = rows[:-1]
    else:
        history_rows = rows

    max_history_messages = max(recent_turns, 0) * 2
    recent_rows = history_rows[-max_history_messages:] if max_history_messages else []
    result: list[dict[str, str]] = []
    summary_message = _format_summary(summary)
    if summary_message is not None:
        result.append(summary_message)
    result.extend(
        {"role": row.role, "content": row.content or ""}
        for row in recent_rows
        if row.role in {"user", "assistant"}
    )
    result.append(current)
    return result


def build_short_term_messages(
    db: Session,
    *,
    user_id: int,
    session_id: int,
    current_user_message: str,
    recent_turns: int | None = None,
) -> list[dict[str, str]]:
    settings = get_settings()
    turns = (
        recent_turns
        if recent_turns is not None
        else settings.short_memory_recent_turns
    )
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .one_or_none()
    )
    rows = (
        db.query(Message)
        .filter(
            Message.user_id == user_id,
            Message.session_id == session_id,
            Message.role.in_(["user", "assistant"]),
        )
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    return compose_short_term_messages(
        summary=getattr(session, "summary", None),
        messages=rows,
        current_user_message=current_user_message,
        recent_turns=turns,
    )


def _summary_prompt(existing_summary: str | None, rows: Sequence[Message]) -> str:
    transcript = "\n".join(
        f"{row.role}: {row.content or ''}"
        for row in rows
        if row.role in {"user", "assistant"}
    )
    existing = (existing_summary or "").strip() or "无"
    return (
        "请将当前会话中较早的对话压缩成给助手使用的短期记忆摘要。\n"
        "要求：保留用户目标、已确认事实、偏好、当前进展和待办；不要编造；用中文要点输出。\n\n"
        f"已有摘要：\n{existing}\n\n"
        f"新增对话：\n{transcript}"
    )


def maybe_update_session_summary(
    db: Session,
    *,
    user_id: int,
    session_id: int,
) -> None:
    settings = get_settings()
    if not settings.short_memory_enabled:
        return

    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .one_or_none()
    )
    if session is None:
        return

    rows = (
        db.query(Message)
        .filter(
            Message.user_id == user_id,
            Message.session_id == session_id,
            Message.role.in_(["user", "assistant"]),
        )
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    threshold = max(settings.short_memory_summary_threshold, 1)
    if len(rows) <= threshold:
        return

    last_summarized_id = getattr(session, "summary_message_id", None) or 0
    unsummarized = [row for row in rows if row.id and row.id > last_summarized_id]
    protected_count = max(settings.short_memory_recent_turns, 1) * 2
    candidates = unsummarized[: max(len(unsummarized) - protected_count, 0)]
    if not candidates:
        return

    batch_size = max(settings.short_memory_summary_batch_size, 1)
    batch = candidates[:batch_size]
    try:
        summary = get_llm().chat(
            [
                {"role": "system", "content": "你负责维护对话短期记忆摘要。"},
                {
                    "role": "user",
                    "content": _summary_prompt(getattr(session, "summary", None), batch),
                },
            ],
            temperature=0.2,
            max_tokens=600,
        )
    except Exception:
        logger.warning("Short-term memory summary failed", exc_info=True)
        return

    text = summary.strip()
    if not text:
        return
    session.summary = text
    session.summary_message_id = batch[-1].id
