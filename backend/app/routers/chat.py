"""Chat HTTP endpoints."""
from __future__ import annotations

import json
import logging
from typing import Generator

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..dependencies.auth import get_current_user
from ..models import User
from ..schemas.chat import ChatRequest
from ..services.chat_workflow import (
    ChatSessionNotFound,
    StreamEvent,
    start_stream_chat,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.ReadTimeout):
        return (
            "模型响应超时，请稍后重试；如果是在生成长篇草稿或调用工具，"
            "可以适当缩短内容或调大 LLM_TIMEOUT_SECONDS。"
        )
    return str(exc) or "internal_error"


def _encode_sse_events(events: Generator[StreamEvent, None, None]) -> Generator[str, None, None]:
    try:
        for event, data in events:
            yield _sse(event, data)
    except Exception as exc:
        logger.exception("Stream chat failed")
        yield _sse("error", {"message": _stream_error_message(exc)})


@router.post("/chat/stream")
def chat_stream(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """SSE streaming chat endpoint scoped to the current debug guest user."""
    try:
        handle = start_stream_chat(user_id=current_user.id, payload=payload)
    except ChatSessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return StreamingResponse(
        _encode_sse_events(handle.events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
