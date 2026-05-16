"""文档表：归属知识库的上传文件及索引状态。"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import List

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Identity, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class DocumentStatus(str, enum.Enum):
    """文档生命周期：上传 → 处理中 → 已索引 / 失败；删除为软删。"""

    uploaded = "uploaded"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"
    deleted = "deleted"


class Document(Base):
    """用户上传的原始文件元数据；正文切块在 ``document_chunks``。"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        Integer, Identity(always=False), primary_key=True
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"),
        nullable=False,
        default=DocumentStatus.uploaded,
        index=True,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(default=0, nullable=False)
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

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    user = relationship("User", back_populates="documents", foreign_keys=[user_id])
    chunks: Mapped[List["DocumentChunk"]] = relationship(  # noqa: F821
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
