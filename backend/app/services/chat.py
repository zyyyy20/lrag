"""Chat orchestration: 装配 memory + retrieval + prompt + LLM。

本模块只编排流程，不做 prompt 拼接（→ prompt_builder）或历史加载（→
memory_service）。两种调用入口：
- ``answer_question``        非流式，一次性返回 (answer, used_rag, sources, notice)
- ``answer_question_stream`` SSE 流式，按事件 (meta / delta / final) 产出。

两者共享同一个"计划阶段" :func:`_plan_answer`，确保行为完全一致。
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Generator, List, Tuple

from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import ChatMode, KnowledgeBase
from ..models import Session as ChatSession
from ..schemas.chat import Source
from .llm import get_llm
from .memory_service import (
    MemoryMessage,
    build_retrieval_query,
    load_recent_messages,
)
from .prompt_builder import BuiltPrompt, build_chat_prompt
from .retrieval import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)


PREVIEW_LEN = 240  # sources 预览文本截断长度


# ============ Source 转换 ============


def _to_sources(chunks: List[RetrievedChunk]) -> List[Source]:
    """把检索结果转换为可暴露给前端的 Source 对象（带预览）。"""
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


# ============ "计划"阶段：内部数据结构 ============


@dataclass
class _AnswerPlan:
    """一次回答所需的全部素材：最终 LLM messages、命中 sources、提示信息等。"""

    prompt: BuiltPrompt
    sources: List[Source]
    used_rag: bool
    notice: str | None
    history: List[MemoryMessage]


def _plan_answer(
    db: Session, session: ChatSession, user_message: str
) -> _AnswerPlan:
    """根据 session 状态 + 用户问题，准备好送往 LLM 的全部内容。

    流程：
        1) 加载 session 的最近历史 memory（按 token 与条数预算）；
        2) 若 session 绑定 RAG：构造检索 query（current + last user），过滤命中；
        3) 调用 prompt_builder 生成最终 messages。
    """
    settings = get_settings()
    notice: str | None = None

    # ---- 1. Memory：加载该 session 的最近历史（严格按 session 隔离）
    history = load_recent_messages(
        db,
        session.id,
        max_messages=settings.max_history_messages,
        exclude_pending_user=True,  # 当前正在被回答的 user 消息已在 DB，但不应重复进历史
    )

    # ---- 2. RAG 判定 + 检索
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
            retrieval_query = build_retrieval_query(user_message, history)
            try:
                retrieved = retrieve(
                    db,
                    retrieval_query,
                    knowledge_base_id=session.knowledge_base_id,
                    top_k=settings.rag_top_k,
                )
            except Exception:
                logger.exception("Retrieval failed; falling back to plain chat")
                retrieved = []
            relevant = [
                c for c in retrieved if c.score >= settings.rag_score_threshold
            ]

    use_rag = bool(relevant)
    sources = _to_sources(relevant) if use_rag else []

    # ---- 3. Prompt 拼接（含 token 预算控制 + 注入防护）
    prompt = build_chat_prompt(
        user_message=user_message,
        history=history,
        rag_chunks=relevant if use_rag else None,
        max_context_tokens=settings.max_context_tokens,
    )
    logger.debug(
        "chat plan: session=%s use_rag=%s history_kept=%d/%d est_tokens=%d",
        session.id,
        use_rag,
        prompt.used_history_count,
        len(history),
        prompt.estimated_input_tokens,
    )

    return _AnswerPlan(
        prompt=prompt,
        sources=sources,
        used_rag=use_rag,
        notice=notice,
        history=history,
    )


# ============ 对外入口：非流式 ============


def answer_question(
    db: Session, session: ChatSession, user_message: str
) -> Tuple[str, bool, List[Source], str | None]:
    """一次性回答。返回 (answer, used_rag, sources, notice)。"""
    plan = _plan_answer(db, session, user_message)
    answer = get_llm().chat(plan.prompt.messages)
    return answer, plan.used_rag, plan.sources, plan.notice


# ============ 对外入口：流式 (SSE) ============


StreamEvent = Tuple[str, dict]


def answer_question_stream(
    db: Session, session: ChatSession, user_message: str
) -> Generator[StreamEvent, None, None]:
    """流式回答。

    yields:
      - ("meta",  {"used_rag", "sources", "notice"})  仅一次，token 开始前
      - ("delta", {"content": str})                   每个 token 块一次
      - ("final", {"content": str})                   仅一次，完整拼接文本
    """
    plan = _plan_answer(db, session, user_message)

    yield (
        "meta",
        {
            "used_rag": plan.used_rag,
            "sources": [s.model_dump(mode="json") for s in plan.sources],
            "notice": plan.notice,
        },
    )

    parts: List[str] = []
    for delta in get_llm().chat_stream(plan.prompt.messages):
        parts.append(delta)
        yield ("delta", {"content": delta})

    yield ("final", {"content": "".join(parts)})


# ============ 标题生成（独立的小调用，不走 memory） ============


def generate_title(first_user_message: str) -> str:
    """根据首条用户消息生成会话标题。LLM 调用失败时降级为消息截断。"""
    fallback = first_user_message.strip().splitlines()[0][:40] or "新会话"
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
