"""Bootstrap LangGraph thread memory from persisted chat messages."""
from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import Message
from .checkpoints import checkpoint_exists


def build_input_messages(
    db: Session,
    *,
    user_id: int,
    session_id: int,
    current_user_message: str,
) -> list[dict[str, str]]:
    """Return bootstrap history for new threads, otherwise only the current turn."""
    if checkpoint_exists(user_id, session_id):
        return [{"role": "user", "content": current_user_message}]

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
    messages = [{"role": row.role, "content": row.content or ""} for row in rows]
    if not messages or messages[-1]["role"] != "user" or messages[-1]["content"] != current_user_message:
        messages.append({"role": "user", "content": current_user_message})
    return messages
