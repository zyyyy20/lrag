from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ChatMode, Message
from ..models import Session as ChatSession
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.chat import answer_question, generate_title

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
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
