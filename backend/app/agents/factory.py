"""Agent runner factory."""
from __future__ import annotations

from functools import lru_cache

from .langgraph_runner import LangGraphAgentRunner


@lru_cache(maxsize=1)
def _get_langgraph_runner() -> LangGraphAgentRunner:
    return LangGraphAgentRunner()


def get_agent_runner() -> LangGraphAgentRunner:
    return _get_langgraph_runner()
