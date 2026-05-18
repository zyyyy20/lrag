"""会话（聊天线程）CRUD：创建、列表、详情、软删除。

会话与 ``messages`` 一对多；``chat_mode`` 与 ``knowledge_base_id`` 决定后续对话
是否走 RAG。软删除通过 ``is_deleted`` 标记，列表与详情接口不返回已删会话。
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies.auth import get_current_user
from ..models import ChatMode, KnowledgeBase, User
from ..models import Session as ChatSession
from ..schemas import MessageOut, SessionCreate, SessionDetail, SessionOut

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _kb_name(db: Session, kb_id: int | None, user_id: int) -> str | None:
    """解析知识库显示名；若 KB 已删或不存在则返回 ``None``（前端可不展示）。"""
    if kb_id is None:
        return None
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.is_deleted.is_(False))
        .one_or_none()
    )
    return kb.name if kb is not None else None


def _to_out(db: Session, s: ChatSession) -> SessionOut:
    """将 ORM ``Session`` 转为 API 输出模型（含知识库名称冗余字段）。"""
    return SessionOut(
        id=s.id,
        title=s.title,
        chat_mode=s.chat_mode,
        knowledge_base_id=s.knowledge_base_id,
        knowledge_base_name=_kb_name(db, s.knowledge_base_id, s.user_id),
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: SessionCreate | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SessionOut:
    """创建新会话。

    - ``chat_mode=rag`` 时必须提供有效且未删除的 ``knowledge_base_id``；
    - ``chat_mode=general`` 时强制清空 ``knowledge_base_id``，避免误绑 KB。
    """
    payload = payload or SessionCreate()
    chat_mode = payload.chat_mode
    kb_id = payload.knowledge_base_id

    if chat_mode == ChatMode.rag:
        if kb_id is None:
            raise HTTPException(
                status_code=400,
                detail="knowledge_base_id is required when chat_mode is 'rag'",
            )
        kb = (
            db.query(KnowledgeBase)
            .filter(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.is_deleted.is_(False),
            )
            .one_or_none()
        )
        if kb is None or kb.is_deleted:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
    else:
        kb_id = None

    title = (payload.title or "新会话").strip() or "新会话"
    s = ChatSession(
        user_id=current_user.id,
        title=title,
        chat_mode=chat_mode,
        knowledge_base_id=kb_id,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return _to_out(db, s)


@router.get("", response_model=List[SessionOut])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[SessionOut]:
    """列出未软删会话，按 ``updated_at`` 倒序（最近活跃在前）。"""
    rows = (
        db.query(ChatSession)
        .filter(
            ChatSession.is_deleted.is_(False),
            ChatSession.user_id == current_user.id,
        )
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return [_to_out(db, r) for r in rows]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SessionDetail:
    """会话详情：含全部消息（前端用于刷新后恢复历史）。"""
    s = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .one_or_none()
    )
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
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    """软删除会话（``is_deleted=True``）；消息随 ORM 级联删除。"""
    s = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .one_or_none()
    )
    if s is None or s.is_deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    s.is_deleted = True
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
