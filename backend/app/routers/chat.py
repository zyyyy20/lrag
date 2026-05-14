"""Chat HTTP endpoints."""
from __future__ import annotations

import json
import logging
from typing import Generator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db, session_scope
from ..dependencies.auth import get_current_user
from ..models import ChatMode, Message, User
from ..models import Session as ChatSession
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.chat import (
    answer_question,
    answer_question_stream,
    generate_title,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """Non-streaming chat endpoint scoped to the current debug guest user."""
    user_id = current_user.id
    if payload.session_id is None:
        session = ChatSession(
            user_id=user_id,
            title="New chat",
            chat_mode=ChatMode.general,
            knowledge_base_id=None,
        )
        db.add(session)
        db.flush()
        is_first_message = True
    else:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == payload.session_id, ChatSession.user_id == user_id)
            .one_or_none()
        )
        if session is None or session.is_deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        is_first_message = (
            db.query(Message)
            .filter(Message.session_id == session.id, Message.user_id == user_id)
            .count()
            == 0
        )

    db.add(
        Message(
            user_id=user_id,
            session_id=session.id,
            role="user",
            content=payload.message,
        )
    )
    db.flush()

    answer, used_rag, sources, notice = answer_question(db, session, payload.message)

    db.add(
        Message(
            user_id=user_id,
            session_id=session.id,
            role="assistant",
            content=answer,
            used_rag=used_rag,
            sources=[s.model_dump(mode="json") for s in sources] if sources else None,
        )
    )

    if is_first_message:
        session.title = generate_title(payload.message)

    db.commit()

    return ChatResponse(
        session_id=session.id,
        answer=answer,
        used_rag=used_rag,
        sources=sources,
        notice=notice,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """SSE streaming chat endpoint scoped to the current debug guest user."""
    user_id = current_user.id
    session_id: int
    is_first_message: bool

    with session_scope() as db:
        if payload.session_id is None:
            new_session = ChatSession(
                user_id=user_id,
                title="New chat",
                chat_mode=ChatMode.general,
                knowledge_base_id=None,
            )
            db.add(new_session)
            db.flush()
            session_id = new_session.id
            is_first_message = True
        else:
            existing = (
                db.query(ChatSession)
                .filter(
                    ChatSession.id == payload.session_id,
                    ChatSession.user_id == user_id,
                )
                .one_or_none()
            )
            if existing is None or existing.is_deleted:
                raise HTTPException(status_code=404, detail="Session not found")
            session_id = existing.id
            is_first_message = (
                db.query(Message)
                .filter(Message.session_id == session_id, Message.user_id == user_id)
                .count()
                == 0
            )

        db.add(
            Message(
                user_id=user_id,
                session_id=session_id,
                role="user",
                content=payload.message,
            )
        )

    user_message_text = payload.message

    def event_generator() -> Generator[str, None, None]:
        used_rag = False
        sources_payload: list = []
        notice: str | None = None
        full_text_parts: list[str] = []
        final_title: str | None = None
        try:
            with session_scope() as db:
                session_obj = (
                    db.query(ChatSession)
                    .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
                    .one_or_none()
                )
                if session_obj is None or session_obj.is_deleted:
                    yield _sse("error", {"message": "Session not found"})
                    return

                for kind, data in answer_question_stream(db, session_obj, user_message_text):
                    if kind == "meta":
                        used_rag = bool(data.get("used_rag"))
                        sources_payload = list(data.get("sources") or [])
                        notice = data.get("notice")
                        yield _sse(
                            "meta",
                            {
                                "session_id": session_id,
                                "used_rag": used_rag,
                                "sources": sources_payload,
                                "notice": notice,
                            },
                        )
                    elif kind == "delta":
                        yield _sse("delta", {"content": data.get("content", "")})
                    elif kind == "final":
                        full_text_parts.append(data.get("content", ""))

                full_text = "".join(full_text_parts).strip()
                db.add(
                    Message(
                        user_id=user_id,
                        session_id=session_id,
                        role="assistant",
                        content=full_text,
                        used_rag=used_rag,
                        sources=sources_payload or None,
                    )
                )

                if is_first_message:
                    try:
                        session_obj.title = generate_title(user_message_text)
                    except Exception:
                        logger.warning("Title generation failed", exc_info=True)

                final_title = session_obj.title

            yield _sse("done", {"session_id": session_id, "title": final_title})
        except Exception as e:
            logger.exception("Stream chat failed")
            yield _sse("error", {"message": str(e) or "internal_error"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
