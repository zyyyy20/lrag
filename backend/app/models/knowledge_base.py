"""知识库表：文档的逻辑分组；软删除后其下文档与 chunk 一并软删。"""
from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import Boolean, DateTime, Identity, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class KnowledgeBase(Base):
    """知识库实体。会话 RAG 模式通过 ``knowledge_base_id`` 外键引用。"""

    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(
        Integer, Identity(always=False), primary_key=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    documents: Mapped[List["Document"]] = relationship(  # noqa: F821
        "Document",
        back_populates="knowledge_base",
    )
