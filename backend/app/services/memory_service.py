"""Token estimation helpers used by tools."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ENC: object | None = None


def _get_encoder():
    global _ENC
    if _ENC is not None:
        return _ENC
    try:
        import tiktoken

        _ENC = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning("tiktoken unavailable; falling back to char-based token estimate")
        _ENC = False
    return _ENC


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _get_encoder()
    if not enc:
        return max(1, len(text) // 2)
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 2)
