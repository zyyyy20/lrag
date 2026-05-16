"""LangGraph/LangChain agent runner."""
from __future__ import annotations

import json
import logging
from typing import Any, Generator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from ..config import get_settings
from ..schemas.chat import Source
from ..tools.builder import build_runtime_tools
from ..tools.runtime import clear_tool_context, reset_tool_context, set_tool_context
from ..tools.types import ToolContext, ToolResult
from .checkpoints import build_thread_config, get_checkpointer
from .langgraph_memory import build_input_messages
from .types import AgentRunResult

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    return """你是 LRAG 的对话助手。
普通问题可以直接回答。

工具规则：
- 当用户要求总结会话、生成发票、查看 token 用量、生成 HTML 票据时，调用 generate_conversation_invoice。
- retrieve_knowledge_base 只能基于当前请求绑定的会话知识库检索；如果当前会话不是知识库会话，工具会返回不可用结果。
- 当用户询问上传文档、知识库内容、文件、手册、制度、部署步骤、说明书、政策、条款等内容时，先调用 retrieve_knowledge_base，再基于工具返回的 context 回答。
- 当用户询问具体人名、组织、项目、产品、文档、术语、日期、金额、编号，或提出“X是谁 / X是什么 / Who is X / What is X”这类实体事实问题时，如果这些信息可能来自当前知识库，先调用 retrieve_knowledge_base。
- 不要声称已经检索或调用了 retrieve_knowledge_base，除非工具确实返回了结果。
- 如果 retrieve_knowledge_base 返回的 context 为空，明确说明知识库中没有检索到匹配内容。
- 不要编造引用来源。不要把 HTML 原文放进回答。"""


def _tool_result_from_payload(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool=str(payload.get("tool") or "unknown"),
        ok=bool(payload.get("ok")),
        data=dict(payload.get("data") or {}),
        message=str(payload.get("message") or ""),
        type=str(payload.get("type") or "tool_result"),
    )


def _parse_tool_payload(content: Any) -> ToolResult | None:
    if isinstance(content, dict):
        return _tool_result_from_payload(content)
    if isinstance(content, list):
        text = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    else:
        text = str(content or "")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Unable to decode LangGraph tool payload: %s", text[:300])
        return None
    if not isinstance(payload, dict):
        return None
    return _tool_result_from_payload(payload)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


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


def _extract_sources(result: ToolResult) -> list[Source]:
    if result.tool != "retrieve_knowledge_base":
        return []
    sources: list[Source] = []
    for item in result.data.get("sources") or []:
        try:
            sources.append(Source.model_validate(item))
        except Exception:
            logger.warning("Invalid source returned by RAG tool: %s", item)
    return sources


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
        self.agent = create_agent(
            model=self.model,
            tools=build_runtime_tools(),
            system_prompt=_build_system_prompt(),
            checkpointer=get_checkpointer(),
        )

    def run(
        self,
        *,
        db: Session,
        user_id: int,
        session_id: int,
        user_message: str,
    ) -> AgentRunResult:
        input_messages = build_input_messages(
            db,
            user_id=user_id,
            session_id=session_id,
            current_user_message=user_message,
        )
        ctx = ToolContext(db=db, user_id=user_id, session_id=session_id)
        token = set_tool_context(ctx)
        try:
            result = self.agent.invoke(
                {"messages": input_messages},
                config=build_thread_config(user_id, session_id),
            )
            return self._to_result(result)
        finally:
            reset_tool_context(token)

    def stream(
        self,
        *,
        db: Session,
        user_id: int,
        session_id: int,
        user_message: str,
    ) -> Generator[tuple[str, Any], None, AgentRunResult]:
        input_messages = build_input_messages(
            db,
            user_id=user_id,
            session_id=session_id,
            current_user_message=user_message,
        )

        delta_parts: list[str] = []
        final_message_text = ""
        tool_results: list[ToolResult] = []
        used_rag = False
        sources: list[Source] = []
        notice: str | None = None

        ctx = ToolContext(db=db, user_id=user_id, session_id=session_id)
        set_tool_context(ctx)
        try:
            for mode, chunk in self.agent.stream(
                {"messages": input_messages},
                config=build_thread_config(user_id, session_id),
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    message_chunk, _metadata = chunk
                    if not isinstance(message_chunk, AIMessageChunk):
                        continue
                    text = _message_text(message_chunk.content)
                    if text:
                        delta_parts.append(text)
                        yield ("delta", {"content": text})
                    continue

                if mode != "updates":
                    continue

                for message in _messages_from_update(chunk):
                    if isinstance(message, ToolMessage):
                        parsed = _parse_tool_payload(message.content)
                        if parsed is None:
                            continue
                        tool_results.append(parsed)
                        if parsed.tool == "retrieve_knowledge_base":
                            used_rag = bool(parsed.data.get("used_rag"))
                            sources = _extract_sources(parsed)
                            if not used_rag:
                                notice = parsed.message
                        yield ("tool_result", parsed)
                    elif isinstance(message, AIMessage):
                        text = _message_text(message.content)
                        if text:
                            final_message_text = text
        finally:
            clear_tool_context()

        final_answer = "".join(delta_parts).strip() or final_message_text.strip()
        return AgentRunResult(
            final_answer=final_answer,
            tool_results=tool_results,
            used_rag=used_rag,
            sources=sources,
            notice=notice,
        )

    def _to_result(self, result: dict[str, Any]) -> AgentRunResult:
        messages = list(result.get("messages") or [])
        final_answer = ""
        tool_results: list[ToolResult] = []
        used_rag = False
        sources: list[Source] = []
        notice: str | None = None

        for message in messages:
            if isinstance(message, ToolMessage):
                parsed = _parse_tool_payload(message.content)
                if parsed is None:
                    continue
                tool_results.append(parsed)
                if parsed.tool == "retrieve_knowledge_base":
                    used_rag = bool(parsed.data.get("used_rag"))
                    sources = _extract_sources(parsed)
                    if not used_rag:
                        notice = parsed.message
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
            notice=notice,
        )
