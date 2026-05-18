from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    knowledge_base_id: Optional[int] = None
    document_id: Optional[int] = None
    filename: str
    chunk_index: int
    score: float
    content_preview: str


class WebSource(BaseModel):
    title: str
    url: str
    content_preview: Optional[str] = None
    score: Optional[float] = None
    source_tool: str = "web_search"


class ChatRequest(BaseModel):
    session_id: Optional[int] = None
    message: str = Field(..., min_length=1, max_length=8000)

