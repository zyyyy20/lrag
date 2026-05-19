"""Conversation title generation helpers."""
from __future__ import annotations

import logging

from ..models import Session as ChatSession
from .llm import get_llm

logger = logging.getLogger(__name__)


def generate_title(first_user_message: str) -> str:
    fallback = first_user_message.strip().splitlines()[0][:40] or "新会话"
    try:
        prompt = (
            "Generate a short, descriptive title (max 6 words, no quotes, no trailing "
            "punctuation) for a conversation that starts with this user message:\n\n"
            f"{first_user_message.strip()[:500]}"
        )
        title = get_llm().chat(
            [
                {"role": "system", "content": "You produce concise titles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=30,
        )
        title = title.strip().strip("\"'").strip()
        return title[:80] if title else fallback
    except Exception:
        logger.warning("Title generation failed; using fallback", exc_info=True)
        return fallback


def set_title_if_needed(
    session: ChatSession,
    *,
    is_first_message: bool,
    user_message: str,
) -> None:
    if not is_first_message:
        return
    session.title = generate_title(user_message)
