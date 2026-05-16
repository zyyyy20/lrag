"""Public knowledge-base management routes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..dependencies.auth import get_current_user
from ..models import Document, DocumentStatus, KnowledgeBase, User
from ..schemas import KnowledgeBaseCreate, KnowledgeBaseOut, KnowledgeBaseUpdate
from ..services.documents import soft_delete_documents_under_kb

router = APIRouter(prefix="/api/knowledge-bases", tags=["knowledge_bases"])


def _ensure_public_kb_write_allowed(kb: KnowledgeBase | None, user_id: int) -> None:
    mode = get_settings().public_kb_write_mode
    if mode == "disabled":
        raise HTTPException(status_code=403, detail="Public knowledge base writes are disabled")
    if mode == "creator_only" and kb is not None and kb.created_by != user_id:
        raise HTTPException(status_code=403, detail="Only the creator can modify this knowledge base")


def _get_active_kb(db: Session, kb_id: int) -> KnowledgeBase:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).one_or_none()
    if kb is None or kb.is_deleted:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


def _to_out(kb: KnowledgeBase, document_count: int) -> KnowledgeBaseOut:
    return KnowledgeBaseOut(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        document_count=document_count,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.post("", response_model=KnowledgeBaseOut, status_code=status.HTTP_201_CREATED)
def create_kb(
    payload: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> KnowledgeBaseOut:
    _ensure_public_kb_write_allowed(None, current_user.id)
    kb = KnowledgeBase(
        user_id=current_user.id,
        created_by=current_user.id,
        name=payload.name.strip(),
        description=payload.description,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return _to_out(kb, 0)


@router.get("", response_model=List[KnowledgeBaseOut])
def list_kbs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[KnowledgeBaseOut]:
    count_subq = (
        select(Document.knowledge_base_id, func.count(Document.id).label("cnt"))
        .where(Document.status != DocumentStatus.deleted)
        .group_by(Document.knowledge_base_id)
        .subquery()
    )
    stmt = (
        select(KnowledgeBase, func.coalesce(count_subq.c.cnt, 0).label("cnt"))
        .outerjoin(count_subq, count_subq.c.knowledge_base_id == KnowledgeBase.id)
        .where(KnowledgeBase.is_deleted.is_(False))
        .order_by(KnowledgeBase.updated_at.desc())
    )
    rows = db.execute(stmt).all()
    return [_to_out(row[0], int(row[1] or 0)) for row in rows]


@router.get("/{kb_id}", response_model=KnowledgeBaseOut)
def get_kb(
    kb_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> KnowledgeBaseOut:
    kb = _get_active_kb(db, kb_id)
    cnt = (
        db.query(func.count(Document.id))
        .filter(
            Document.knowledge_base_id == kb.id,
            Document.status != DocumentStatus.deleted,
        )
        .scalar()
        or 0
    )
    return _to_out(kb, int(cnt))


@router.patch("/{kb_id}", response_model=KnowledgeBaseOut)
def update_kb(
    kb_id: int,
    payload: KnowledgeBaseUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> KnowledgeBaseOut:
    kb = _get_active_kb(db, kb_id)
    _ensure_public_kb_write_allowed(kb, current_user.id)
    if payload.name is not None:
        kb.name = payload.name.strip()
    if payload.description is not None:
        kb.description = payload.description
    db.commit()
    db.refresh(kb)
    cnt = (
        db.query(func.count(Document.id))
        .filter(
            Document.knowledge_base_id == kb.id,
            Document.status != DocumentStatus.deleted,
        )
        .scalar()
        or 0
    )
    return _to_out(kb, int(cnt))


@router.delete(
    "/{kb_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_kb(
    kb_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    kb = _get_active_kb(db, kb_id)
    _ensure_public_kb_write_allowed(kb, current_user.id)
    kb.is_deleted = True
    kb.deleted_at = datetime.now(timezone.utc)
    db.commit()
    soft_delete_documents_under_kb(kb_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
