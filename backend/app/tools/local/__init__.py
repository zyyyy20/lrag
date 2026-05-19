"""Local LangChain tools implemented by this backend."""
from __future__ import annotations

from .invoice import build_runtime_generate_conversation_invoice_tool
from .rag import build_runtime_retrieve_knowledge_base_tool


def build_local_tools() -> list:
    return [
        build_runtime_generate_conversation_invoice_tool(),
        build_runtime_retrieve_knowledge_base_tool(),
    ]


__all__ = [
    "build_local_tools",
    "build_runtime_generate_conversation_invoice_tool",
    "build_runtime_retrieve_knowledge_base_tool",
]
