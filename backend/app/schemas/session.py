from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..models.session import ChatMode
from .chat import Source


class SessionCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    chat_mode: ChatMode = ChatMode.general
    knowledge_base_id: Optional[uuid.UUID] = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    sources: Optional[List[Source]] = None
    used_rag: Optional[bool] = None
    created_at: datetime


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    chat_mode: ChatMode
    knowledge_base_id: Optional[uuid.UUID] = None
    knowledge_base_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SessionDetail(SessionOut):
    messages: List[MessageOut] = []
