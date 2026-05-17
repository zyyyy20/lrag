"""Source extraction helpers for agent tool results."""
from __future__ import annotations

import logging

from ..schemas.chat import Source
from ..tools.types import ToolResult

logger = logging.getLogger(__name__)


def extract_sources(result: ToolResult) -> list[Source]:
    if result.tool != "retrieve_knowledge_base":
        return []
    sources: list[Source] = []
    for item in result.data.get("sources") or []:
        try:
            sources.append(Source.model_validate(item))
        except Exception:
            logger.warning("Invalid source returned by RAG tool: %s", item)
    return sources
