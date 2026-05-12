"""Chat orchestration: history -> retrieval -> LLM."""
from __future__ import annotations

import logging
import uuid
from typing import Generator, List, Tuple

from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import ChatMode, KnowledgeBase, Message
from ..models import Session as ChatSession
from ..schemas.chat import Source
from .llm import get_llm
from .retrieval import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_BASE = (
    "You are a concise, helpful assistant. Answer the user clearly. "
    "Reply in the user's language."
)

SYSTEM_PROMPT_RAG = (
    "You are a knowledgeable assistant answering with the help of the provided context. "
    "Follow these rules strictly:\n"
    "1. Prefer information from the CONTEXT below. If the context does not contain the "
    "answer, say so honestly and answer from general knowledge where appropriate.\n"
    "2. Do NOT fabricate citations or filenames. Only refer to sources that actually appear "
    "in the CONTEXT.\n"
    "3. Be concise and accurate. Reply in the user's language."
)

PREVIEW_LEN = 240
HISTORY_TURNS = 10


def _history_messages(db: Session, session_id: uuid.UUID) -> List[dict]:
    rows = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    rows = rows[-HISTORY_TURNS * 2 :]
    return [{"role": r.role, "content": r.content} for r in rows if r.role in ("user", "assistant")]


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}] file={c.filename} chunk={c.chunk_index} score={c.score:.3f}\n{c.content}"
        )
    return "\n\n".join(parts)


def _to_sources(chunks: List[RetrievedChunk]) -> List[Source]:
    out: List[Source] = []
    for c in chunks:
        preview = c.content.strip().replace("\n", " ")
        if len(preview) > PREVIEW_LEN:
            preview = preview[:PREVIEW_LEN] + "…"
        out.append(
            Source(
                knowledge_base_id=uuid.UUID(c.knowledge_base_id),
                document_id=uuid.UUID(c.document_id),
                filename=c.filename,
                chunk_index=c.chunk_index,
                score=round(c.score, 4),
                content_preview=preview,
            )
        )
    return out


def answer_question(
    db: Session, session: ChatSession, user_message: str
) -> Tuple[str, bool, List[Source], str | None]:
    """Returns: (answer, used_rag, sources, notice).

    The session's chat_mode and knowledge_base_id determine the behavior:
      - chat_mode == general OR knowledge_base_id is None -> plain chat
      - chat_mode == rag AND KB exists & not deleted    -> RAG over that KB
      - chat_mode == rag AND KB missing/deleted         -> plain chat + notice
    """
    settings = get_settings()
    notice: str | None = None

    use_rag_intent = (
        session.chat_mode == ChatMode.rag and session.knowledge_base_id is not None
    )

    relevant: List[RetrievedChunk] = []
    if use_rag_intent:
        kb = db.get(KnowledgeBase, session.knowledge_base_id)
        if kb is None or kb.is_deleted:
            notice = "当前会话绑定的知识库已被删除，已自动降级为普通聊天。"
            use_rag_intent = False
        else:
            try:
                retrieved = retrieve(
                    db,
                    user_message,
                    knowledge_base_id=session.knowledge_base_id,
                    top_k=settings.rag_top_k,
                )
            except Exception:
                logger.exception("Retrieval failed; falling back to plain chat")
                retrieved = []
            relevant = [c for c in retrieved if c.score >= settings.rag_score_threshold]

    use_rag = bool(relevant)
    history = _history_messages(db, session.id)

    if use_rag:
        context_block = _build_context_block(relevant)
        user_with_ctx = (
            f"CONTEXT:\n{context_block}\n\n"
            f"USER QUESTION:\n{user_message}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            *history,
            {"role": "user", "content": user_with_ctx},
        ]
        sources = _to_sources(relevant)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            *history,
            {"role": "user", "content": user_message},
        ]
        sources = []

    answer = get_llm().chat(messages)
    return answer, use_rag, sources, notice


StreamEvent = Tuple[str, dict]


def answer_question_stream(
    db: Session, session: ChatSession, user_message: str
) -> Generator[StreamEvent, None, None]:
    """Streaming variant of answer_question.

    Yields events of the form (event_name, payload):
      - ("meta",  {"used_rag", "sources", "notice"})  — emitted exactly once before any deltas
      - ("delta", {"content": str})                   — emitted many times as tokens arrive
      - ("final", {"content": str})                   — emitted once at end with full assembled text
    """
    settings = get_settings()
    notice: str | None = None

    use_rag_intent = (
        session.chat_mode == ChatMode.rag and session.knowledge_base_id is not None
    )

    relevant: List[RetrievedChunk] = []
    if use_rag_intent:
        kb = db.get(KnowledgeBase, session.knowledge_base_id)
        if kb is None or kb.is_deleted:
            notice = "当前会话绑定的知识库已被删除，已自动降级为普通聊天。"
            use_rag_intent = False
        else:
            try:
                retrieved = retrieve(
                    db,
                    user_message,
                    knowledge_base_id=session.knowledge_base_id,
                    top_k=settings.rag_top_k,
                )
            except Exception:
                logger.exception("Retrieval failed; falling back to plain chat")
                retrieved = []
            relevant = [c for c in retrieved if c.score >= settings.rag_score_threshold]

    use_rag = bool(relevant)
    sources = _to_sources(relevant) if use_rag else []
    history = _history_messages(db, session.id)

    if use_rag:
        context_block = _build_context_block(relevant)
        user_with_ctx = (
            f"CONTEXT:\n{context_block}\n\n"
            f"USER QUESTION:\n{user_message}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            *history,
            {"role": "user", "content": user_with_ctx},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            *history,
            {"role": "user", "content": user_message},
        ]

    yield (
        "meta",
        {
            "used_rag": use_rag,
            "sources": [s.model_dump(mode="json") for s in sources],
            "notice": notice,
        },
    )

    parts: List[str] = []
    for delta in get_llm().chat_stream(messages):
        parts.append(delta)
        yield ("delta", {"content": delta})

    yield ("final", {"content": "".join(parts)})


def generate_title(first_user_message: str) -> str:
    """Generate a short title for a session. Falls back to a truncated version of the
    user message if the LLM call fails."""
    fallback = first_user_message.strip().splitlines()[0][:40] or "New chat"
    try:
        prompt = (
            "Generate a short, descriptive title (max 6 words, no quotes, no trailing "
            "punctuation) for a conversation that starts with this user message:\n\n"
            f"{first_user_message.strip()[:500]}"
        )
        title = get_llm().chat(
            [
                {"role": "system", "content": "You produce concise titles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=30,
        )
        title = title.strip().strip("\"'").strip()
        if not title:
            return fallback
        return title[:80]
    except Exception:
        logger.warning("Title generation failed; using fallback")
        return fallback
