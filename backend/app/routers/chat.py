"""Chat HTTP endpoints."""
from __future__ import annotations

import json
import logging
from typing import Generator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies.auth import get_current_user
from ..models import User
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.chat_workflow import (
    ChatSessionNotFound,
    StreamEvent,
    complete_chat,
    start_stream_chat,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """Non-streaming chat endpoint scoped to the current debug guest user."""
    try:
        return complete_chat(db, user_id=current_user.id, payload=payload)
    except ChatSessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _encode_sse_events(events: Generator[StreamEvent, None, None]) -> Generator[str, None, None]:
    try:
        for event, data in events:
            yield _sse(event, data)
    except Exception as exc:
        logger.exception("Stream chat failed")
        yield _sse("error", {"message": str(exc) or "internal_error"})


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
