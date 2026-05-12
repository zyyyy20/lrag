from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Document, DocumentStatus, KnowledgeBase
from ..schemas import KnowledgeBaseCreate, KnowledgeBaseOut, KnowledgeBaseUpdate
from ..services.documents import soft_delete_documents_under_kb

router = APIRouter(prefix="/api/knowledge-bases", tags=["knowledge_bases"])


def _get_active_kb(db: Session, kb_id: uuid.UUID) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, kb_id)
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
def create_kb(payload: KnowledgeBaseCreate, db: Session = Depends(get_db)) -> KnowledgeBaseOut:
    kb = KnowledgeBase(name=payload.name.strip(), description=payload.description)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return _to_out(kb, 0)


@router.get("", response_model=List[KnowledgeBaseOut])
def list_kbs(db: Session = Depends(get_db)) -> List[KnowledgeBaseOut]:
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
def get_kb(kb_id: uuid.UUID, db: Session = Depends(get_db)) -> KnowledgeBaseOut:
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
    kb_id: uuid.UUID,
    payload: KnowledgeBaseUpdate,
    db: Session = Depends(get_db),
) -> KnowledgeBaseOut:
    kb = _get_active_kb(db, kb_id)
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
def delete_kb(kb_id: uuid.UUID, db: Session = Depends(get_db)) -> Response:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None or kb.is_deleted:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    kb.is_deleted = True
    kb.deleted_at = datetime.now(timezone.utc)
    db.commit()
    # cascade soft-delete documents + chunks (uses its own session)
    soft_delete_documents_under_kb(kb_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
