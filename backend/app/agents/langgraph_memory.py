"""Bootstrap LangGraph thread memory from persisted chat messages."""
from __future__ import annotations

from sqlalchemy.orm import Session

from ..services.short_term_memory import build_short_term_messages


def build_input_messages(
    db: Session,
    *,
    user_id: int,
    session_id: int,
    current_user_message: str,
    recent_turns: int | None = None,
) -> list[dict[str, str]]:
    """Return explicit short-term memory context for the current turn."""
    return build_short_term_messages(
        db,
        user_id=user_id,
        session_id=session_id,
        current_user_message=current_user_message,
        recent_turns=recent_turns,
    )
