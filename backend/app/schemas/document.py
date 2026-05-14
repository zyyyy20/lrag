from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from ..models.document import DocumentStatus


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    knowledge_base_id: int
    filename: str
    content_type: str
    file_size: int
    status: DocumentStatus
    chunk_count: int
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
