"""Conversation Memory service.

负责：
1. 按 session 加载最近 N 条消息（不允许跨 session）；
2. 把消息列表按 token 预算截断，保留最近上下文；
3. 基于历史构造更优的"检索查询语句"（current + 最近一轮 user message）。

设计原则：
- Memory 始终绑定到单个 session_id；删除 session 时其消息会随 ORM 级联删除，
  memory 自然失效。
- 仅服务层使用，不在 router 直接调用 ORM；保持分层。
- token 计数使用 tiktoken 的 cl100k_base（与 OpenAI 多数模型一致）。对 Qwen
  等其它模型属于近似估算，足够用于"丢历史"的预算控制。
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.orm import Session

from ..models import Message

logger = logging.getLogger(__name__)


# 允许进入 LLM prompt 的角色白名单（防止脏数据 / 注入 system 角色）
_ALLOWED_ROLES = {"user", "assistant"}


# ----- token 计数 -----

_ENC = None  # tiktoken encoder 单例


def _get_encoder():
    """惰性初始化 tiktoken encoder。若环境缺失（极少见）则降级为字符估算。"""
    global _ENC
    if _ENC is not None:
        return _ENC
    try:
        import tiktoken

        _ENC = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning("tiktoken unavailable; falling back to char-based token estimate")
        _ENC = False  # 用 False 表示已尝试过但失败，避免反复探测
    return _ENC


def count_tokens(text: str) -> int:
    """估算字符串占用的 token 数。失败时按 1 token ≈ 2 字符兜底。"""
    if not text:
        return 0
    enc = _get_encoder()
    if not enc:
        return max(1, len(text) // 2)
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 2)


# ----- Memory 数据结构 -----


@dataclass
class MemoryMessage:
    """LLM 视角的一条历史消息（只包含 role 和 content）。"""

    role: str
    content: str

    def tokens(self) -> int:
        # 4 token 的固定开销留给 chat-completion 的 role/分隔符等元数据
        return count_tokens(self.content) + 4


# ----- 加载历史 -----


def load_recent_messages(
    db: Session,
    session_id: uuid.UUID,
    max_messages: int,
    *,
    exclude_pending_user: bool = True,
) -> List[MemoryMessage]:
    """从 DB 加载某个 session 的最近 N 条历史消息（按时间升序返回）。

    Args:
        db: SQLAlchemy session。
        session_id: 必填，严格按 session 隔离，禁止跨 session 共享。
        max_messages: 最多回放多少条（仅统计 user/assistant 角色）。
        exclude_pending_user: 若最末一条是 user 消息（即"用户刚刚发出、正在被
            回答"的那条），则不计入历史 —— 因为它会作为最终的 user prompt
            单独追加，避免重复。

    Returns:
        按时间升序的 MemoryMessage 列表（最旧→最新）。
    """
    if max_messages <= 0:
        return []

    rows = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .filter(Message.role.in_(_ALLOWED_ROLES))
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    if not rows:
        return []

    if exclude_pending_user and rows[-1].role == "user":
        rows = rows[:-1]

    # 取最近 max_messages 条
    rows = rows[-max_messages:]
    return [MemoryMessage(role=r.role, content=r.content or "") for r in rows]


# ----- token 预算截断 -----


def truncate_by_tokens(
    history: List[MemoryMessage], budget_tokens: int
) -> List[MemoryMessage]:
    """从最旧端向最新端丢弃消息，直到剩余 token 总数 <= budget_tokens。

    保留时间顺序；预算非正时直接返回空列表。
    """
    if budget_tokens <= 0 or not history:
        return []

    total = sum(m.tokens() for m in history)
    if total <= budget_tokens:
        return list(history)

    # 从最早消息开始丢，直到满足预算
    kept = list(history)
    while kept and total > budget_tokens:
        dropped = kept.pop(0)
        total -= dropped.tokens()
    return kept


# ----- 检索查询构造 -----


def build_retrieval_query(
    current_message: str, history: List[MemoryMessage]
) -> str:
    """根据"当前问题 + 最近一轮 user 消息"组合出更稳的检索查询。

    用整段历史去做 embedding 容易引入噪声（话题漂移、token 浪费、上下文重要
    度被稀释），因此这里只引入"最近一次 user 提问"作为补充上下文。

    Returns:
        最终用于向量检索的 query 字符串。
    """
    current = (current_message or "").strip()
    last_user_content: Optional[str] = None
    for m in reversed(history):
        if m.role == "user" and m.content.strip():
            last_user_content = m.content.strip()
            break
    if last_user_content and last_user_content != current:
        # 最近一轮提问放前面作为"话题锚点"，当前问题置于末尾突出。
        return f"{last_user_content}\n{current}"
    return current
