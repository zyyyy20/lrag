"""LLM client wrappers. Uses the OpenAI SDK which is compatible with
DeepSeek / Qwen (DashScope compatible-mode) when given a custom base_url."""
from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)


def _build_client(api_key: str, base_url: str | None) -> OpenAI:
    if not api_key:
        raise RuntimeError(
            "Missing API key. Please configure LLM_API_KEY / EMBEDDING_API_KEY in .env"
        )
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


class LLMClient:
    def __init__(self) -> None:
        s = get_settings()
        self.client = _build_client(s.llm_api_key, s.llm_base_url)
        self.model = s.llm_model

    def chat(self, messages: List[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class EmbeddingClient:
    def __init__(self) -> None:
        s = get_settings()
        self.client = _build_client(s.embedding_api_key, s.embedding_base_url)
        self.model = s.embedding_model
        self.dim = s.embedding_dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


_llm: LLMClient | None = None
_emb: EmbeddingClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def get_embedder() -> EmbeddingClient:
    global _emb
    if _emb is None:
        _emb = EmbeddingClient()
    return _emb
