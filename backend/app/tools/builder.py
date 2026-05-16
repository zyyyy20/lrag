"""Build per-request LangChain tools with backend context closed over."""
from __future__ import annotations

from .invoice import (
    build_runtime_generate_conversation_invoice_tool,
)
from .rag import (
    build_runtime_retrieve_knowledge_base_tool,
)


def build_runtime_tools() -> list:
    return [
        build_runtime_generate_conversation_invoice_tool(),
        build_runtime_retrieve_knowledge_base_tool(),
    ]
