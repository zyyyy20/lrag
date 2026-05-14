from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    knowledge_base_id: Optional[int] = None
    document_id: Optional[int] = None
    filename: str
    chunk_index: int
    score: float
    content_preview: str


class ChatRequest(BaseModel):
    session_id: Optional[int] = None
    message: str = Field(..., min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    used_rag: bool
    sources: List[Source] = []
    notice: Optional[str] = None
