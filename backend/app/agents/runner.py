"""Constrained ReAct runner used by chat endpoints."""
from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from ..config import get_settings
from ..schemas.chat import Source
from ..services.llm import get_llm
from ..tools import ensure_tools_registered, tool_registry
from ..tools.base import ToolContext, ToolResult
from .parser import parse_react_decision
from .prompts import build_react_system_prompt
from .types import AgentRunResult

logger = logging.getLogger(__name__)


def _observation_payload(result: ToolResult) -> dict:
    payload = result.model_dump()
    if result.tool == "retrieve_knowledge_base":
        max_chars = get_settings().agent_rag_observation_max_chars
        data = dict(payload.get("data") or {})
        context = str(data.get("context") or "")
        if len(context) > max_chars:
            data["context"] = context[:max_chars] + "\n...[truncated]"
        payload["data"] = data
    return payload


def _extract_rag_sources(result: ToolResult) -> list[Source]:
    if result.tool != "retrieve_knowledge_base":
        return []
    raw_sources = result.data.get("sources") or []
    sources: list[Source] = []
    for item in raw_sources:
        try:
            sources.append(Source.model_validate(item))
        except Exception:
            logger.warning("Invalid RAG source payload: %s", item)
    return sources


class AgentRunner:
    """Small ReAct loop around an allow-listed tool registry."""

    def __init__(self, max_steps: int | None = None) -> None:
        ensure_tools_registered()
        settings = get_settings()
        self.max_steps = max_steps or settings.react_agent_max_steps

    def try_run_tool(
        self,
        *,
        db: Session,
        user_id: int,
        session_id: int,
        user_message: str,
    ) -> AgentRunResult | None:
        if not get_settings().enable_react_agent:
            return None

        tool_results: list[ToolResult] = []
        ctx = ToolContext(db=db, user_id=user_id, session_id=session_id)
        tools = tool_registry.list_available(ctx)
        if not tools:
            return None

        messages: list[dict] = [
            {"role": "system", "content": build_react_system_prompt(tools)},
            {
                "role": "user",
                "content": (
                    "Latest user message:\n"
                    f"{user_message.strip()}\n\n"
                    "Decide whether a tool is needed."
                ),
            },
        ]
        used_rag = False
        rag_sources: list[Source] = []
        notice: str | None = None

        for _ in range(max(1, self.max_steps)):
            raw = get_llm().chat(messages, temperature=0.0, max_tokens=500)
            decision = parse_react_decision(raw)

            if decision.final_answer:
                if decision.final_answer.strip() == "NO_TOOL" and not tool_results:
                    return None
                if decision.final_answer.strip() == "NO_TOOL" and tool_results:
                    return AgentRunResult(
                        final_answer=tool_results[-1].message,
                        tool_results=tool_results,
                        used_rag=used_rag,
                        sources=rag_sources,
                        notice=notice,
                    )
                return AgentRunResult(
                    final_answer=decision.final_answer.strip(),
                    tool_results=tool_results,
                    used_rag=used_rag,
                    sources=rag_sources,
                    notice=notice,
                )

            if not decision.tool_name:
                return None

            spec = tool_registry.get(decision.tool_name)
            if spec is None:
                logger.warning("Model requested unknown tool: %s", decision.tool_name)
                return None

            try:
                result = spec.handler(ctx, decision.arguments)
            except Exception as exc:
                logger.exception("Tool execution failed: %s", decision.tool_name)
                result = ToolResult(
                    tool=decision.tool_name,
                    ok=False,
                    data={},
                    message=str(exc) or "工具执行失败。",
                )
            tool_results.append(result)
            if result.tool == "retrieve_knowledge_base":
                used_rag = bool(result.data.get("used_rag"))
                rag_sources = _extract_rag_sources(result)
                if not used_rag:
                    notice = result.message
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Observation: "
                    + json.dumps(_observation_payload(result), ensure_ascii=False)
                    + "\nReturn the Final Answer now.",
                }
            )

        if tool_results:
            return AgentRunResult(
                final_answer=tool_results[-1].message,
                tool_results=tool_results,
                used_rag=used_rag,
                sources=rag_sources,
                notice=notice,
            )
        return None
