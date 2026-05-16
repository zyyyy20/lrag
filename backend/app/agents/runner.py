"""Constrained tool-calling agent runner used by chat endpoints."""
from __future__ import annotations

import json
import logging
from typing import Any

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
    """Small tool-calling loop around an allow-listed tool registry."""

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

        used_rag = False
        rag_sources: list[Source] = []
        notice: str | None = None

        if get_settings().agent_calling_mode == "react_text":
            return self._try_run_react_text(
                ctx=ctx,
                tools=tools,
                user_message=user_message,
                tool_results=tool_results,
                used_rag=used_rag,
                rag_sources=rag_sources,
                notice=notice,
            )

        try:
            return self._try_run_function_call(
                ctx=ctx,
                tools=tools,
                user_message=user_message,
                tool_results=tool_results,
                used_rag=used_rag,
                rag_sources=rag_sources,
                notice=notice,
            )
        except Exception:
            logger.warning(
                "Function calling agent failed; falling back to react_text",
                exc_info=True,
            )
            return self._try_run_react_text(
                ctx=ctx,
                tools=tools,
                user_message=user_message,
                tool_results=[],
                used_rag=False,
                rag_sources=[],
                notice=None,
            )

    def _run_tool(
        self,
        *,
        ctx: ToolContext,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        spec = tool_registry.get(tool_name)
        if spec is None:
            logger.warning("Model requested unknown tool: %s", tool_name)
            return ToolResult(
                tool=tool_name,
                ok=False,
                data={},
                message=f"Unknown tool: {tool_name}",
            )
        try:
            return spec.handler(ctx, arguments)
        except Exception as exc:
            logger.exception("Tool execution failed: %s", tool_name)
            return ToolResult(
                tool=tool_name,
                ok=False,
                data={},
                message=str(exc) or "工具执行失败。",
            )

    def _update_rag_state(
        self,
        result: ToolResult,
        *,
        used_rag: bool,
        rag_sources: list[Source],
        notice: str | None,
    ) -> tuple[bool, list[Source], str | None]:
        if result.tool != "retrieve_knowledge_base":
            return used_rag, rag_sources, notice
        used_rag = bool(result.data.get("used_rag"))
        rag_sources = _extract_rag_sources(result)
        if not used_rag:
            notice = result.message
        return used_rag, rag_sources, notice

    @staticmethod
    def _tool_call_to_message(call) -> dict[str, Any]:
        return {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments or "{}",
            },
        }

    @staticmethod
    def _parse_tool_arguments(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _try_run_function_call(
        self,
        *,
        ctx: ToolContext,
        tools,
        user_message: str,
        tool_results: list[ToolResult],
        used_rag: bool,
        rag_sources: list[Source],
        notice: str | None,
    ) -> AgentRunResult | None:
        messages: list[dict] = [
            {"role": "system", "content": build_react_system_prompt(tools)},
            {"role": "user", "content": user_message.strip()},
        ]
        openai_tools = [tool.to_openai_tool() for tool in tools]

        for _ in range(max(1, self.max_steps)):
            message = get_llm().chat_with_tools(
                messages,
                openai_tools,
                temperature=0.0,
                max_tokens=800,
            )
            calls = list(getattr(message, "tool_calls", None) or [])
            content = (getattr(message, "content", None) or "").strip()

            if not calls:
                if not tool_results:
                    return None
                return AgentRunResult(
                    final_answer=content or tool_results[-1].message,
                    tool_results=tool_results,
                    used_rag=used_rag,
                    sources=rag_sources,
                    notice=notice,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [self._tool_call_to_message(call) for call in calls],
                }
            )

            for call in calls:
                tool_name = call.function.name
                arguments = self._parse_tool_arguments(call.function.arguments)
                result = self._run_tool(
                    ctx=ctx,
                    tool_name=tool_name,
                    arguments=arguments,
                )
                tool_results.append(result)
                used_rag, rag_sources, notice = self._update_rag_state(
                    result,
                    used_rag=used_rag,
                    rag_sources=rag_sources,
                    notice=notice,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            _observation_payload(result), ensure_ascii=False
                        ),
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

    def _try_run_react_text(
        self,
        *,
        ctx: ToolContext,
        tools,
        user_message: str,
        tool_results: list[ToolResult],
        used_rag: bool,
        rag_sources: list[Source],
        notice: str | None,
    ) -> AgentRunResult | None:
        messages: list[dict] = [
            {"role": "system", "content": build_react_system_prompt(tools, react_text=True)},
            {
                "role": "user",
                "content": (
                    "Latest user message:\n"
                    f"{user_message.strip()}\n\n"
                    "Decide whether a tool is needed."
                ),
            },
        ]

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

            result = self._run_tool(
                ctx=ctx,
                tool_name=decision.tool_name,
                arguments=decision.arguments,
            )
            tool_results.append(result)
            used_rag, rag_sources, notice = self._update_rag_state(
                result,
                used_rag=used_rag,
                rag_sources=rag_sources,
                notice=notice,
            )
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

