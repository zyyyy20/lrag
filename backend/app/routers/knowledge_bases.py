"""知识库（Knowledge Base）管理：增删改查与软删除。

知识库为文档的逻辑容器；删除知识库会级联软删其下文档与 chunk（见
``services.documents.soft_delete_documents_under_kb``），已有会话记录保留，
后续对话由编排层检测 KB 已删并降级。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies.auth import get_current_user
from ..models import Document, DocumentStatus, KnowledgeBase, User
from ..schemas import KnowledgeBaseCreate, KnowledgeBaseOut, KnowledgeBaseUpdate
from ..services.documents import soft_delete_documents_under_kb

router = APIRouter(prefix="/api/knowledge-bases", tags=["knowledge_bases"])


def _get_active_kb(db: Session, kb_id: int, user_id: int) -> KnowledgeBase:
    """获取未软删知识库；否则 404。"""
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == user_id)
        .one_or_none()
    )
    if kb is None or kb.is_deleted:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


def _to_out(kb: KnowledgeBase, document_count: int) -> KnowledgeBaseOut:
    """ORM → API 模型，附带未删文档数量。"""
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
    """新建知识库（名称必填，描述可选）。"""
    kb = KnowledgeBase(
        user_id=current_user.id,
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
    """列出未删知识库，并左连接统计各库未删文档数。"""
    count_subq = (
        select(Document.knowledge_base_id, func.count(Document.id).label("cnt"))
        .where(
            Document.status != DocumentStatus.deleted,
            Document.user_id == current_user.id,
        )
        .group_by(Document.knowledge_base_id)
        .subquery()
    )
    stmt = (
        select(KnowledgeBase, func.coalesce(count_subq.c.cnt, 0).label("cnt"))
        .outerjoin(count_subq, count_subq.c.knowledge_base_id == KnowledgeBase.id)
        .where(
            KnowledgeBase.is_deleted.is_(False),
            KnowledgeBase.user_id == current_user.id,
        )
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
    """单个知识库详情（含文档计数）。"""
    kb = _get_active_kb(db, kb_id, current_user.id)
    cnt = (
        db.query(func.count(Document.id))
        .filter(
            Document.knowledge_base_id == kb.id,
            Document.user_id == current_user.id,
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
    """部分更新名称与/或描述。"""
    kb = _get_active_kb(db, kb_id, current_user.id)
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
            Document.user_id == current_user.id,
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
    """软删除知识库并级联软删其下文档与 chunk（独立 session 内执行级联逻辑）。"""
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == current_user.id)
        .one_or_none()
    )
    if kb is None or kb.is_deleted:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    kb.is_deleted = True
    kb.deleted_at = datetime.now(timezone.utc)
    db.commit()
    soft_delete_documents_under_kb(kb_id, current_user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
