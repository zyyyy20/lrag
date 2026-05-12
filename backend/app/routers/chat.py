from __future__ import annotations

import json
import logging
import uuid
from typing import Generator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db, session_scope
from ..models import ChatMode, Message
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
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Non-streaming chat endpoint (kept for backward compatibility)."""
    if payload.session_id is None:
        session = ChatSession(title="新会话", chat_mode=ChatMode.general, knowledge_base_id=None)
        db.add(session)
        db.flush()
        is_first_message = True
    else:
        session = db.get(ChatSession, payload.session_id)
        if session is None or session.is_deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        is_first_message = (
            db.query(Message).filter(Message.session_id == session.id).count() == 0
        )

    user_msg = Message(session_id=session.id, role="user", content=payload.message)
    db.add(user_msg)
    db.flush()

    answer, used_rag, sources, notice = answer_question(db, session, payload.message)

    assistant_msg = Message(
        session_id=session.id,
        role="assistant",
        content=answer,
        used_rag=used_rag,
        sources=[s.model_dump(mode="json") for s in sources] if sources else None,
    )
    db.add(assistant_msg)

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
    """Format a single SSE event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint that emits Server-Sent Events.

    Event protocol:
      - event: meta   data: {session_id, used_rag, sources, notice}
      - event: delta  data: {content}                  (many)
      - event: done   data: {session_id, title}
      - event: error  data: {message}

    Note: this endpoint deliberately does NOT use Depends(get_db). FastAPI tears
    down generator-based dependencies right after the route function returns,
    which would close the DB session before the StreamingResponse body is
    actually iterated. Instead we manage DB sessions explicitly via
    `session_scope()` inside the generator.
    """
    # --- Phase 1: validate input and persist user message in a short-lived session ---
    session_id: uuid.UUID
    is_first_message: bool
    with session_scope() as db:
        if payload.session_id is None:
            new_session = ChatSession(
                title="新会话", chat_mode=ChatMode.general, knowledge_base_id=None
            )
            db.add(new_session)
            db.flush()
            session_id = new_session.id
            is_first_message = True
        else:
            existing = db.get(ChatSession, payload.session_id)
            if existing is None or existing.is_deleted:
                raise HTTPException(status_code=404, detail="Session not found")
            session_id = existing.id
            is_first_message = (
                db.query(Message).filter(Message.session_id == session_id).count() == 0
            )

        db.add(Message(session_id=session_id, role="user", content=payload.message))
        # commit happens implicitly on scope exit

    session_id_str = str(session_id)
    user_message_text = payload.message

    def event_generator() -> Generator[str, None, None]:
        used_rag = False
        sources_payload: list = []
        notice: str | None = None
        full_text_parts: list[str] = []
        final_title: str | None = None
        try:
            # --- Phase 2: retrieve + stream + persist within a single DB session ---
            with session_scope() as db:
                session_obj = db.get(ChatSession, session_id)
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
                                "session_id": session_id_str,
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
                # commit happens on scope exit

            yield _sse("done", {"session_id": session_id_str, "title": final_title})
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
