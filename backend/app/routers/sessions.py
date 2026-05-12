from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ChatMode, KnowledgeBase, Message
from ..models import Session as ChatSession
from ..schemas import MessageOut, SessionCreate, SessionDetail, SessionOut

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _kb_name(db: Session, kb_id: uuid.UUID | None) -> str | None:
    if kb_id is None:
        return None
    kb = db.get(KnowledgeBase, kb_id)
    return kb.name if kb is not None else None


def _to_out(db: Session, s: ChatSession) -> SessionOut:
    return SessionOut(
        id=s.id,
        title=s.title,
        chat_mode=s.chat_mode,
        knowledge_base_id=s.knowledge_base_id,
        knowledge_base_name=_kb_name(db, s.knowledge_base_id),
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: SessionCreate | None = None, db: Session = Depends(get_db)
) -> SessionOut:
    payload = payload or SessionCreate()
    chat_mode = payload.chat_mode
    kb_id = payload.knowledge_base_id

    if chat_mode == ChatMode.rag:
        if kb_id is None:
            raise HTTPException(
                status_code=400,
                detail="knowledge_base_id is required when chat_mode is 'rag'",
            )
        kb = db.get(KnowledgeBase, kb_id)
        if kb is None or kb.is_deleted:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
    else:
        # plain chat must not be bound to a KB
        kb_id = None

    title = (payload.title or "新会话").strip() or "新会话"
    s = ChatSession(title=title, chat_mode=chat_mode, knowledge_base_id=kb_id)
    db.add(s)
    db.commit()
    db.refresh(s)
    return _to_out(db, s)


@router.get("", response_model=List[SessionOut])
def list_sessions(db: Session = Depends(get_db)) -> List[SessionOut]:
    rows = (
        db.query(ChatSession)
        .filter(ChatSession.is_deleted.is_(False))
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return [_to_out(db, r) for r in rows]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: uuid.UUID, db: Session = Depends(get_db)) -> SessionDetail:
    s = db.get(ChatSession, session_id)
    if s is None or s.is_deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    out = _to_out(db, s)
    return SessionDetail(
        **out.model_dump(),
        messages=[MessageOut.model_validate(m) for m in s.messages],
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_session(session_id: uuid.UUID, db: Session = Depends(get_db)) -> Response:
    s = db.get(ChatSession, session_id)
    if s is None or s.is_deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    s.is_deleted = True
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
