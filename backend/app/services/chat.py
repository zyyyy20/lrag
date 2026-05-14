"""对话编排服务：串联 Memory、检索、Prompt 构建与大模型调用。

**模块定位**
- 本文件只做「流程编排」，不直接拼接 prompt 字符串（见 ``prompt_builder``）；
- 不直接写复杂 SQL 读历史（见 ``memory_service``）；
- 路由层（``routers/chat``）负责 HTTP、事务边界与 SSE 帧格式，本层不关心传输细节。

**两条对外入口（行为必须一致）**
1. ``answer_question``：同步一次性返回完整回复，供 ``POST /api/chat`` 使用；
2. ``answer_question_stream``：生成器流式产出事件元组，供 ``POST /api/chat/stream`` 包装为 SSE。

二者在调用 LLM 之前共享 ``_plan_answer``，确保「是否 RAG / sources / notice /
messages」完全一致，避免流式与非流式行为分叉。

**RAG 与降级（摘要）**
- 仅当 ``session.chat_mode == rag`` 且绑定有效知识库时才发起向量检索；
- 知识库已软删：不检索，``notice`` 提示前端，走普通聊天；
- 检索异常：静默降级为无命中；
- 全部 chunk 分数低于 ``RAG_SCORE_THRESHOLD``：视为未命中，走普通聊天、``sources`` 为空。
"""
from __future__ import annotations

import logging
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

PREVIEW_LEN = 240  # 前端 ``content_preview`` 最大展示长度（字符）


def _to_sources(chunks: List[RetrievedChunk]) -> List[Source]:
    """将检索命中块转为 API 层的 ``Source`` 列表（含预览文本，禁止伪造字段）。

    ``filename`` / ``chunk_index`` / ``score`` / ``document_id`` / ``knowledge_base_id``
    均来自数据库与检索结果，不由模型生成。
    """
    out: List[Source] = []
    for c in chunks:
        preview = c.content.strip().replace("\n", " ")
        if len(preview) > PREVIEW_LEN:
            preview = preview[:PREVIEW_LEN] + "…"
        out.append(
            Source(
                knowledge_base_id=c.knowledge_base_id,
                document_id=c.document_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                score=round(c.score, 4),
                content_preview=preview,
            )
        )
    return out


@dataclass
class _AnswerPlan:
    """单次回答的「计划结果」：已拼好的 LLM 输入、对外 sources、提示等。"""

    prompt: BuiltPrompt
    sources: List[Source]
    used_rag: bool
    notice: str | None
    history: List[MemoryMessage]


def _plan_answer(
    db: Session, session: ChatSession, user_message: str
) -> _AnswerPlan:
    """核心编排：加载记忆 → 条件检索 → 构建 prompt。

    详细步骤：
    1. **Memory**：按 ``MAX_HISTORY_MESSAGES`` 读取当前会话最近 user/assistant，
       并默认剔除「刚写入 DB、本轮待回答」的那条 user，避免与最终 user 重复；
    2. **RAG 意图**：仅 ``chat_mode=rag`` 且 ``knowledge_base_id`` 非空时尝试检索；
       若知识库记录不存在或已软删，设置 ``notice`` 并跳过检索；
    3. **检索 query**：``build_retrieval_query(本轮, history)``，不把整段历史塞进 embedding；
    4. **阈值过滤**：低于 ``RAG_SCORE_THRESHOLD`` 的 chunk 全部丢弃 → ``used_rag=False``；
    5. **Prompt**：调用 ``build_chat_prompt``，在 ``MAX_CONTEXT_TOKENS`` 内截断历史。

    Args:
        db: 与当前请求或流式阶段绑定的 SQLAlchemy Session（由调用方保证生命周期）。
        session: 当前会话 ORM 对象（须已绑定到 ``db``）。
        user_message: 本轮用户文本。

    Returns:
        ``_AnswerPlan``，供同步或流式 LLM 调用共用。
    """
    settings = get_settings()
    notice: str | None = None

    history = load_recent_messages(
        db,
        session.id,
        max_messages=settings.max_history_messages,
        exclude_pending_user=True,
    )

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


def answer_question(
    db: Session, session: ChatSession, user_message: str
) -> Tuple[str, bool, List[Source], str | None]:
    """非流式：返回 ``(助手全文, 是否启用RAG, 引用列表, 可选提示)``。"""
    plan = _plan_answer(db, session, user_message)
    answer = get_llm().chat(plan.prompt.messages)
    return answer, plan.used_rag, plan.sources, plan.notice


StreamEvent = Tuple[str, dict]


def answer_question_stream(
    db: Session, session: ChatSession, user_message: str
) -> Generator[StreamEvent, None, None]:
    """流式：先产出 meta（含 sources），再多次 delta，最后一条 final 为全文拼接结果。

    路由层负责把事件转为 SSE；本函数只产出 Python 元组 ``(event_name, payload)``。

    Yields:
        - ``("meta", {"used_rag", "sources", "notice"})``：一次，在首个 token 前；
        - ``("delta", {"content": str})``：零次或多次；
        - ``("final", {"content": str})``：一次，``content`` 为完整助手回复（用于校验/落库）。
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


def generate_title(first_user_message: str) -> str:
    """根据会话首条用户消息生成短标题；LLM 失败时截取首行作为兜底。

    不走 memory / RAG，仅独立一次短补全调用。
    """
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
