"""Debug guest user identity."""
from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import DateTime, Identity, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class User(Base):
    """Anonymous debug user resolved from a browser/device fingerprint."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer, Identity(always=False), primary_key=True
    )
    debug_guest_id: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    device_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    knowledge_bases: Mapped[List["KnowledgeBase"]] = relationship(  # noqa: F821
        "KnowledgeBase", back_populates="user"
    )
    sessions: Mapped[List["Session"]] = relationship(  # noqa: F821
        "Session", back_populates="user"
    )
    documents: Mapped[List["Document"]] = relationship(  # noqa: F821
        "Document", back_populates="user"
    )
    messages: Mapped[List["Message"]] = relationship(  # noqa: F821
        "Message", back_populates="user"
    )
