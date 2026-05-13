"""会话级对话记忆（Conversation Memory）服务。

本模块不直接调用大模型，只负责从数据库读取、裁剪、格式化「与当前会话绑定的」
历史消息，以及构造用于向量检索的查询字符串。

设计原则（与需求对齐）：
1. **严格按 session 隔离**：所有查询必须带 ``session_id``，禁止跨会话共享；
2. **删除会话即记忆失效**：依赖 ORM 级联删除 ``messages``，本层不维护额外缓存；
3. **Token 预算**：与 ``prompt_builder`` 配合，用 ``count_tokens`` / ``truncate_by_tokens``
   控制进入 LLM 的历史长度；
4. **检索 query 不堆整段历史**：仅用「当前问题 + 最近一条 user 消息」组合，
   减少 embedding 噪声与话题漂移。

Token 计数默认使用 tiktoken ``cl100k_base``；对 Qwen 等模型为近似值，仅用于
「丢最早历史」的相对排序，不追求与网关计费 token 完全一致。
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.orm import Session

from ..models import Message

logger = logging.getLogger(__name__)

# 仅允许 user / assistant 进入记忆与 prompt；防止脏数据伪造 system 角色
_ALLOWED_ROLES = {"user", "assistant"}

# tiktoken 编码器单例；False 表示已尝试加载失败，改用字符估算
_ENC: object | None = None


def _get_encoder():
    """惰性加载 tiktoken 编码器；失败则标记为 False 并走字符估算分支。"""
    global _ENC
    if _ENC is not None:
        return _ENC
    try:
        import tiktoken

        _ENC = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning("tiktoken unavailable; falling back to char-based token estimate")
        _ENC = False
    return _ENC


def count_tokens(text: str) -> int:
    """估算文本 token 数（用于历史截断预算）。

    若 tiktoken 不可用或编码失败，按「约 2 字符 = 1 token」兜底，避免阻塞主流程。

    Args:
        text: 任意字符串。

    Returns:
        非负整数 token 估计值。
    """
    if not text:
        return 0
    enc = _get_encoder()
    if not enc:
        return max(1, len(text) // 2)
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 2)


@dataclass
class MemoryMessage:
    """单条可送入 LLM 历史槽位的消息（与 ORM ``Message`` 解耦，仅 role + content）。"""

    role: str
    content: str

    def tokens(self) -> int:
        """本条消息估算 token：正文 + 少量协议开销（role 分隔等）。"""
        return count_tokens(self.content) + 4


def load_recent_messages(
    db: Session,
    session_id: uuid.UUID,
    max_messages: int,
    *,
    exclude_pending_user: bool = True,
) -> List[MemoryMessage]:
    """从数据库加载指定会话的最近若干条 user/assistant 消息（时间升序）。

    **为何排除末尾 user（可选）**：路由在调用编排层之前通常已把「本轮用户
    输入」写入 ``messages`` 表。若此处不把最后一条 user 从历史中剔除，则
    ``prompt_builder`` 最终再追加一条 user 时会出现**同一句用户话重复两次**，
    既浪费 token 又可能干扰模型。因此 ``exclude_pending_user=True`` 为默认。

    Args:
        db: 数据库会话。
        session_id: 会话主键（隔离边界）。
        max_messages: 最多保留多少条历史（仅 user/assistant）。
        exclude_pending_user: 为 True 时，若最后一条是 user 则先去掉再取最近 N 条。

    Returns:
        ``MemoryMessage`` 列表，从旧到新排序。
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

    rows = rows[-max_messages:]
    return [MemoryMessage(role=r.role, content=r.content or "") for r in rows]


def truncate_by_tokens(
    history: List[MemoryMessage], budget_tokens: int
) -> List[MemoryMessage]:
    """在总 token 不超过 ``budget_tokens`` 的前提下，保留**时间上靠后**的历史。

    策略：从列表头部（最旧消息）开始逐条弹出，直到剩余总和 ≤ 预算；若预算
    ≤0 或历史为空则返回空列表。这样保证「当前轮」在 ``prompt_builder`` 里
    单独占用的 system + user（含 RAG CONTEXT）优先不被挤掉。

    Args:
        history: 时间升序的历史消息列表。
        budget_tokens: 允许历史部分占用的最大 token 数。

    Returns:
        截断后的列表，仍为时间升序。
    """
    if budget_tokens <= 0 or not history:
        return []

    total = sum(m.tokens() for m in history)
    if total <= budget_tokens:
        return list(history)

    kept = list(history)
    while kept and total > budget_tokens:
        dropped = kept.pop(0)
        total -= dropped.tokens()
    return kept


def build_retrieval_query(
    current_message: str, history: List[MemoryMessage]
) -> str:
    """构造向量检索用的查询字符串（**不是**整段 prompt）。

    若把多轮闲聊全文都塞进 embedding，向量会被无关词稀释，召回变差。因此
    只拼接「**时间顺序上最近的一条 user 内容**」与「当前用户输入」；若二者
    相同（例如连续两条都是当前句）则只返回当前句。

    典型追问场景：上一轮问「产品支持哪些格式？」本轮问「那 PDF 呢？」——
    组合后检索能同时携带主题锚点与细化条件。

    Args:
        current_message: 本轮用户原始输入。
        history: 已加载且已排除「待回答 user」的历史（见 ``load_recent_messages``）。

    Returns:
        送入 ``retrieve`` / ``embed_one`` 的查询文本。
    """
    current = (current_message or "").strip()
    last_user_content: Optional[str] = None
    for m in reversed(history):
        if m.role == "user" and m.content.strip():
            last_user_content = m.content.strip()
            break
    if last_user_content and last_user_content != current:
        # 上一轮放前作锚点，当前放后突出本轮焦点
        return f"{last_user_content}\n{current}"
    return current
