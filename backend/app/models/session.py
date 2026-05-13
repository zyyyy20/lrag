"""会话表：一条聊天线程，绑定可选知识库与对话模式。"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import List

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class ChatMode(str, enum.Enum):
    """会话模式：普通聊天不检索；RAG 模式仅检索绑定的知识库。"""

    general = "general"
    rag = "rag"


class Session(Base):
    """聊天会话。软删除用 ``is_deleted``；消息通过 ``messages`` 级联删除。"""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="新会话")
    chat_mode: Mapped[ChatMode] = mapped_column(
        Enum(ChatMode, name="chat_mode"),
        nullable=False,
        default=ChatMode.general,
    )
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    messages: Mapped[List["Message"]] = relationship(  # noqa: F821
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    knowledge_base = relationship("KnowledgeBase")
