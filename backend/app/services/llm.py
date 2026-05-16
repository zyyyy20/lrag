"""大语言模型与向量嵌入客户端封装。

使用 OpenAI 官方 Python SDK（``openai`` 包）。当 ``LLM_BASE_URL`` /
``EMBEDDING_BASE_URL`` 指向 DashScope、DeepSeek 等兼容 OpenAI 协议的网关时，
无需改代码即可切换供应商。

设计要点：
- ``LLMClient``：非流式 ``chat`` 与流式 ``chat_stream``，供对话编排层调用；
- ``EmbeddingClient``：批量 ``embed`` 与单条 ``embed_one``，供检索与文档索引；
- ``get_llm`` / ``get_embedder``：进程内懒加载单例，避免重复创建 HTTP 客户端。

注意：API Key 仅从配置读取，禁止在代码中硬编码。
"""
from __future__ import annotations

import logging
from typing import Any, Generator, List

from openai import OpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)


def _build_client(api_key: str, base_url: str | None) -> OpenAI:
    """根据 API Key 与可选 base_url 构造 OpenAI 兼容客户端。

    Args:
        api_key: 供应商颁发的密钥；为空时抛出明确错误，避免静默失败。
        base_url: 自定义网关根地址，例如 ``https://dashscope.aliyuncs.com/compatible-mode/v1``。

    Returns:
        配置好的 ``OpenAI`` 实例。
    """
    if not api_key:
        raise RuntimeError(
            "Missing API key. Please configure LLM_API_KEY / EMBEDDING_API_KEY in .env"
        )
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


class LLMClient:
    """对话补全客户端（Chat Completions）。"""

    def __init__(self) -> None:
        s = get_settings()
        self.client = _build_client(s.llm_api_key, s.llm_base_url)
        self.model = s.llm_model

    def chat(self, messages: List[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """非流式对话：一次性返回完整助手回复文本（首尾已 strip）。

        Args:
            messages: OpenAI 格式的 ``[{"role","content"}, ...]`` 列表。
            temperature: 采样温度。
            max_tokens: 生成上限（由模型与网关共同约束）。

        Returns:
            助手回复的纯文本；若模型返回空内容则返回空字符串。
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def chat_with_tools(
        self,
        messages: List[dict],
        tools: List[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """Chat completion that may return structured tool_calls."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message

    def chat_stream(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """流式对话：逐块 yield 文本增量（delta）。

        空或 None 的 delta 会被跳过，调用方只需拼接非空字符串即可还原完整回复。

        Yields:
            每个流式 chunk 中的 ``content`` 字符串片段。
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) if delta else None
            if content:
                yield content


class EmbeddingClient:
    """文本向量嵌入客户端。"""

    def __init__(self) -> None:
        s = get_settings()
        self.client = _build_client(s.embedding_api_key, s.embedding_base_url)
        self.model = s.embedding_model
        self.dim = s.embedding_dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        """对一批文本生成向量；顺序与输入一一对应。

        Args:
            texts: 非空字符串列表；空列表直接返回空列表。

        Returns:
            与 ``texts`` 等长的浮点向量列表；维度由配置 ``embedding_dim`` 决定。
        """
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> List[float]:
        """单条文本嵌入，等价于 ``embed([text])[0]``。"""
        return self.embed([text])[0]


_llm: LLMClient | None = None
_emb: EmbeddingClient | None = None


def get_llm() -> LLMClient:
    """获取全局单例 ``LLMClient``。"""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def get_embedder() -> EmbeddingClient:
    """获取全局单例 ``EmbeddingClient``。"""
    global _emb
    if _emb is None:
        _emb = EmbeddingClient()
    return _emb
