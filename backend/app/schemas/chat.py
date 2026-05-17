from __future__ import annotations

from typing import Any, List, Optional

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


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    used_rag: bool
    sources: List[Source] = []
    used_web: bool = False
    web_sources: List[WebSource] = []
    notice: Optional[str] = None
    tool_results: List[dict[str, Any]] = []
