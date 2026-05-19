from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models.session import ChatMode
from ..tools.core.results import extract_web_sources, tool_result_from_payload
from .chat import Source, WebSource


def _web_sources_from_tool_results(
    tool_results: list[dict[str, Any]] | None,
) -> list[WebSource]:
    sources: list[WebSource] = []
    seen_urls: set[str] = set()
    for result in tool_results or []:
        for source in extract_web_sources(tool_result_from_payload(result)):
            if source.url in seen_urls:
                continue
            sources.append(source)
            seen_urls.add(source.url)
    return sources


class SessionCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    chat_mode: ChatMode = ChatMode.general
    knowledge_base_id: Optional[int] = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    sources: Optional[List[Source]] = None
    web_sources: Optional[List[WebSource]] = None
    tool_results: Optional[List[dict[str, Any]]] = None
    used_rag: Optional[bool] = None
    used_web: Optional[bool] = None
    created_at: datetime

    @model_validator(mode="after")
    def fill_web_sources_from_tool_results(self) -> "MessageOut":
        if self.web_sources is None:
            self.web_sources = _web_sources_from_tool_results(self.tool_results)
        if self.used_web is None:
            self.used_web = bool(self.web_sources)
        return self


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    chat_mode: ChatMode
    knowledge_base_id: Optional[int] = None
    knowledge_base_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SessionDetail(SessionOut):
    messages: List[MessageOut] = []
