"""Knowledge-base retrieval tool for the ReAct agent."""
from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..models import ChatMode, KnowledgeBase
from ..models import Session as ChatSession
from ..schemas.chat import Source
from ..services.memory_service import count_tokens
from ..services.retrieval import RetrievedChunk, retrieve
from .base import ToolContext, ToolResult, ToolSpec

TOOL_NAME = "retrieve_knowledge_base"
MAX_CONTEXT_TOKENS = 1800
SOURCE_PREVIEW_LEN = 240


def _safe_top_k(value: Any) -> int:
    settings = get_settings()
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = settings.rag_top_k
    return max(1, min(parsed, 10))


def _load_session(ctx: ToolContext) -> ChatSession | None:
    return (
        ctx.db.query(ChatSession)
        .filter(ChatSession.id == ctx.session_id, ChatSession.user_id == ctx.user_id)
        .one_or_none()
    )


def _knowledge_base_is_valid(ctx: ToolContext, knowledge_base_id: int) -> bool:
    return (
        ctx.db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == ctx.user_id,
            KnowledgeBase.is_deleted.is_(False),
        )
        .one_or_none()
        is not None
    )


def rag_tool_available(ctx: ToolContext) -> bool:
    if not get_settings().enable_agent_rag:
        return False
    session = _load_session(ctx)
    return bool(
        session
        and not session.is_deleted
        and session.chat_mode == ChatMode.rag
        and session.knowledge_base_id is not None
        and _knowledge_base_is_valid(ctx, session.knowledge_base_id)
    )


def _to_source(chunk: RetrievedChunk) -> Source:
    preview = chunk.content.strip().replace("\n", " ")
    if len(preview) > SOURCE_PREVIEW_LEN:
        preview = preview[:SOURCE_PREVIEW_LEN] + "..."
    return Source(
        knowledge_base_id=chunk.knowledge_base_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        chunk_index=chunk.chunk_index,
        score=round(chunk.score, 4),
        content_preview=preview,
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    used_tokens = 0
    for index, chunk in enumerate(chunks, start=1):
        block = (
            f"[Source {index}] filename={chunk.filename}; "
            f"document_id={chunk.document_id}; chunk_index={chunk.chunk_index}; "
            f"score={chunk.score:.4f}\n{chunk.content.strip()}"
        )
        tokens = count_tokens(block)
        if lines and used_tokens + tokens > MAX_CONTEXT_TOKENS:
            break
        lines.append(block)
        used_tokens += tokens
    return "\n\n".join(lines)


def retrieve_knowledge_base(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    session = _load_session(ctx)
    if session is None or session.is_deleted:
        return ToolResult(
            tool=TOOL_NAME,
            ok=False,
            message="当前会话不存在或已删除。",
            data={"used_rag": False, "sources": [], "context": ""},
        )

    if (
        session.chat_mode != ChatMode.rag
        or session.knowledge_base_id is None
        or not _knowledge_base_is_valid(ctx, session.knowledge_base_id)
    ):
        return ToolResult(
            tool=TOOL_NAME,
            ok=False,
            message="当前会话未绑定可用知识库。",
            data={"used_rag": False, "sources": [], "context": ""},
        )

    query = str(arguments.get("query") or "").strip()
    if not query:
        return ToolResult(
            tool=TOOL_NAME,
            ok=False,
            message="检索问题为空，未执行知识库检索。",
            data={"used_rag": False, "sources": [], "context": ""},
        )

    settings = get_settings()
    chunks = retrieve(
        ctx.db,
        query,
        knowledge_base_id=session.knowledge_base_id,
        user_id=ctx.user_id,
        top_k=_safe_top_k(arguments.get("top_k")),
    )
    relevant = [c for c in chunks if c.score >= settings.rag_score_threshold]
    sources = [_to_source(chunk) for chunk in relevant]
    context = _format_context(relevant)

    if not relevant:
        return ToolResult(
            tool=TOOL_NAME,
            ok=True,
            message="知识库中未检索到足够相关的片段。",
            data={"used_rag": False, "sources": [], "context": ""},
        )

    return ToolResult(
        tool=TOOL_NAME,
        ok=True,
        message=f"已检索到 {len(relevant)} 条相关知识库片段。",
        data={
            "used_rag": True,
            "sources": [source.model_dump(mode="json") for source in sources],
            "context": context,
        },
    )


retrieve_knowledge_base_tool = ToolSpec(
    name=TOOL_NAME,
    description=(
        "Retrieve relevant chunks from the knowledge base bound to the current "
        "session. Use it for questions that require uploaded documents or "
        "knowledge-base content."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query rewritten from the user's latest question.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Maximum number of chunks to retrieve.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=retrieve_knowledge_base,
    is_available=rag_tool_available,
)
