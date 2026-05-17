"""LangGraph/LangChain agent runner."""
from __future__ import annotations

from typing import Any, Generator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from ..config import get_settings
from ..schemas.chat import Source, WebSource
from ..tools.builder import build_runtime_tools
from ..tools.runtime import (
    clear_tool_context,
    register_tool_context,
    reset_tool_context,
    set_tool_context,
    unregister_tool_context,
)
from ..tools.types import ToolContext, ToolResult
from .checkpoints import build_thread_config, get_checkpointer
from .langgraph_memory import build_input_messages
from .prompts import build_base_system_prompt, build_system_prompt
from .sources import extract_sources
from .tool_events import parse_tool_payload, tool_call_events
from .types import AgentRunResult
from .web_sources import extract_web_sources


def _tool_context_key(user_id: int, session_id: int) -> str:
    return str(build_thread_config(user_id, session_id)["configurable"]["thread_id"])


def _runtime_config(user_id: int, session_id: int) -> dict[str, Any]:
    config = build_thread_config(user_id, session_id)
    config["configurable"]["tool_context_key"] = _tool_context_key(user_id, session_id)
    return config



def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def _agent_step(
    step_id: str,
    title: str,
    *,
    status: str = "running",
    content: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": step_id,
        "type": "agent_step",
        "title": title,
        "status": status,
    }
    if content:
        payload["content"] = content
    return payload


def _messages_from_update(update: Any) -> list[Any]:
    messages: list[Any] = []
    if isinstance(update, dict):
        value = update.get("messages")
        if isinstance(value, list):
            messages.extend(value)
        for child in update.values():
            if isinstance(child, dict):
                messages.extend(_messages_from_update(child))
    return messages


def _merge_web_sources(
    existing: list[WebSource],
    incoming: list[WebSource],
) -> list[WebSource]:
    seen = {source.url for source in existing}
    merged = list(existing)
    for source in incoming:
        if source.url in seen:
            continue
        merged.append(source)
        seen.add(source.url)
    return merged


class LangGraphAgentRunner:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0.3,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self.base_system_prompt = build_base_system_prompt()
        self.agent = self._create_agent(self.base_system_prompt)

    def _create_agent(self, system_prompt: str):
        return create_agent(
            model=self.model,
            tools=build_runtime_tools(),
            system_prompt=system_prompt,
            checkpointer=get_checkpointer(),
        )

    def _agent_for_memory_context(self, long_term_memory_context: str | None):
        system_prompt = build_system_prompt(
            self.base_system_prompt,
            long_term_memory_context,
        )
        if system_prompt == self.base_system_prompt:
            return self.agent
        return self._create_agent(system_prompt)

    def run(
        self,
        *,
        db: Session,
        user_id: int,
        session_id: int,
        user_message: str,
        long_term_memory_context: str | None = None,
    ) -> AgentRunResult:
        input_messages = build_input_messages(
            db,
            user_id=user_id,
            session_id=session_id,
            current_user_message=user_message,
        )
        agent = self._agent_for_memory_context(long_term_memory_context)
        ctx = ToolContext(db=db, user_id=user_id, session_id=session_id)
        key = _tool_context_key(user_id, session_id)
        register_tool_context(key, ctx)
        token = set_tool_context(ctx)
        try:
            result = agent.invoke(
                {"messages": input_messages},
                config=_runtime_config(user_id, session_id),
            )
            return self._to_result(result)
        finally:
            reset_tool_context(token)
            unregister_tool_context(key)

    def stream(
        self,
        *,
        db: Session,
        user_id: int,
        session_id: int,
        user_message: str,
        long_term_memory_context: str | None = None,
    ) -> Generator[tuple[str, Any], None, AgentRunResult]:
        input_messages = build_input_messages(
            db,
            user_id=user_id,
            session_id=session_id,
            current_user_message=user_message,
        )
        agent = self._agent_for_memory_context(long_term_memory_context)

        delta_parts: list[str] = []
        final_message_text = ""
        tool_results: list[ToolResult] = []
        used_rag = False
        sources: list[Source] = []
        used_web = False
        web_sources: list[WebSource] = []
        notice: str | None = None

        ctx = ToolContext(db=db, user_id=user_id, session_id=session_id)
        key = _tool_context_key(user_id, session_id)
        register_tool_context(key, ctx)
        set_tool_context(ctx)
        seen_tool_calls: set[str] = set()
        analysis_finished = False
        answer_started = False
        yield (
            "agent_step",
            {
                "id": "analysis",
                "type": "agent_step",
                "title": "分析用户问题",
                "status": "running",
                "content": "判断是否需要调用知识库或其他工具",
            },
        )
        try:
            for mode, chunk in agent.stream(
                {"messages": input_messages},
                config=_runtime_config(user_id, session_id),
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    message_chunk, _metadata = chunk
                    if not isinstance(message_chunk, AIMessageChunk):
                        continue
                    text = _message_text(message_chunk.content)
                    if text:
                        if not analysis_finished:
                            analysis_finished = True
                            yield (
                                "agent_step",
                                _agent_step(
                                    "analysis",
                                    "分析用户问题",
                                    status="success",
                                    content="已完成工具选择与上下文判断",
                                ),
                            )
                        if not answer_started:
                            answer_started = True
                            yield (
                                "agent_step",
                                _agent_step(
                                    "answer",
                                    "生成回答",
                                    content="基于当前上下文生成流式回复",
                                ),
                            )
                        delta_parts.append(text)
                        yield ("delta", {"content": text})
                    continue

                if mode != "updates":
                    continue

                for message in _messages_from_update(chunk):
                    if isinstance(message, AIMessage):
                        for event in tool_call_events(message):
                            event_id = str(event["id"])
                            if event_id in seen_tool_calls:
                                continue
                            seen_tool_calls.add(event_id)
                            if not analysis_finished:
                                analysis_finished = True
                                yield (
                                    "agent_step",
                                    _agent_step(
                                        "analysis",
                                        "分析用户问题",
                                        status="success",
                                        content="已完成工具选择与上下文判断",
                                    ),
                                )
                            yield ("tool_call", event)
                        text = _message_text(message.content)
                        if text:
                            final_message_text = text
                        continue

                    if isinstance(message, ToolMessage):
                        parsed = parse_tool_payload(
                            message.content,
                            tool_name=getattr(message, "name", None),
                        )
                        if parsed is None:
                            continue
                        tool_results.append(parsed)
                        if parsed.tool == "retrieve_knowledge_base":
                            used_rag = bool(parsed.data.get("used_rag"))
                            sources = extract_sources(parsed)
                            if not used_rag:
                                notice = parsed.message
                        new_web_sources = extract_web_sources(parsed)
                        if new_web_sources:
                            web_sources = _merge_web_sources(
                                web_sources,
                                new_web_sources,
                            )
                            used_web = True
                        yield ("tool_result", parsed)
        finally:
            clear_tool_context()
            unregister_tool_context(key)

        final_answer = "".join(delta_parts).strip() or final_message_text.strip()
        if not analysis_finished:
            yield (
                "agent_step",
                _agent_step(
                    "analysis",
                    "分析用户问题",
                    status="success",
                    content="已完成工具选择与上下文判断",
                ),
            )
        if answer_started:
            yield ("agent_step", _agent_step("answer", "生成回答", status="success"))
        return AgentRunResult(
            final_answer=final_answer,
            tool_results=tool_results,
            used_rag=used_rag,
            sources=sources,
            used_web=used_web,
            web_sources=web_sources,
            notice=notice,
        )

    def _to_result(self, result: dict[str, Any]) -> AgentRunResult:
        messages = list(result.get("messages") or [])
        final_answer = ""
        tool_results: list[ToolResult] = []
        used_rag = False
        sources: list[Source] = []
        used_web = False
        web_sources: list[WebSource] = []
        notice: str | None = None

        for message in messages:
            if isinstance(message, ToolMessage):
                parsed = parse_tool_payload(
                    message.content,
                    tool_name=getattr(message, "name", None),
                )
                if parsed is None:
                    continue
                tool_results.append(parsed)
                if parsed.tool == "retrieve_knowledge_base":
                    used_rag = bool(parsed.data.get("used_rag"))
                    sources = extract_sources(parsed)
                    if not used_rag:
                        notice = parsed.message
                new_web_sources = extract_web_sources(parsed)
                if new_web_sources:
                    web_sources = _merge_web_sources(web_sources, new_web_sources)
                    used_web = True
            elif isinstance(message, AIMessage) and message.content:
                final_answer = (
                    message.content
                    if isinstance(message.content, str)
                    else str(message.content)
                )

        if not final_answer and tool_results:
            final_answer = tool_results[-1].message
        return AgentRunResult(
            final_answer=final_answer.strip(),
            tool_results=tool_results,
            used_rag=used_rag,
            sources=sources,
            used_web=used_web,
            web_sources=web_sources,
            notice=notice,
        )
