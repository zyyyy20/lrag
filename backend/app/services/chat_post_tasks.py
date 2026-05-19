"""Background post-processing tasks for completed chat turns."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from ..database import session_scope
from ..models import Session as ChatSession
from .chat_title import set_title_if_needed
from .long_term_memory import remember_turn
from .short_term_memory import maybe_update_session_summary

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chat-post-task")


def schedule_chat_post_tasks(
    *,
    user_id: int,
    session_id: int,
    user_message: str,
    assistant_message: str,
    is_first_message: bool,
) -> None:
    """Schedule non-critical chat post-processing outside the SSE critical path."""
    _executor.submit(
        run_chat_post_tasks,
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        assistant_message=assistant_message,
        is_first_message=is_first_message,
    )


def run_chat_post_tasks(
    *,
    user_id: int,
    session_id: int,
    user_message: str,
    assistant_message: str,
    is_first_message: bool,
) -> None:
    """Update derived chat state and long-term memory after the answer is sent."""
    _update_title_and_summary(
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        is_first_message=is_first_message,
    )
    _write_long_term_memory(
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        assistant_message=assistant_message,
    )


def _update_title_and_summary(
    *,
    user_id: int,
    session_id: int,
    user_message: str,
    is_first_message: bool,
) -> None:
    try:
        with session_scope() as db:
            session = (
                db.query(ChatSession)
                .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
                .one_or_none()
            )
            if session is None or session.is_deleted:
                return
            set_title_if_needed(
                session,
                is_first_message=is_first_message,
                user_message=user_message,
            )
            maybe_update_session_summary(
                db,
                user_id=user_id,
                session_id=session_id,
            )
    except Exception:
        logger.warning("Chat title/summary post task failed", exc_info=True)


def _write_long_term_memory(
    *,
    user_id: int,
    session_id: int,
    user_message: str,
    assistant_message: str,
) -> None:
    try:
        remember_turn(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )
    except Exception:
        logger.warning("Chat long-term memory post task failed", exc_info=True)
