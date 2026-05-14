"""文档分块表：每块文本 + pgvector 向量；支持软删除不参与检索。"""
from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Identity, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..config import get_settings
from ..database import Base

_settings = get_settings()


class DocumentChunk(Base):
    """向量检索的最小单位。``embedding`` 维度由配置 ``EMBEDDING_DIM`` 决定。"""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(
        Integer, Identity(always=False), primary_key=True
    )
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[list[float]] = mapped_column(Vector(_settings.embedding_dim))
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    document = relationship("Document", back_populates="chunks")
