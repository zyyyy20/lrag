"""Application service for chat request orchestration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator

from sqlalchemy.orm import Session

from ..agents.factory import get_agent_runner
from ..database import session_scope
from ..models import ChatMode, Message
from ..models import Session as ChatSession
from ..schemas.chat import ChatRequest
from .llm import get_llm
from .long_term_memory import (
    remember_turn,
)
from .short_term_memory import maybe_update_session_summary
from ..tools.core.results import extract_web_sources, tool_result_from_payload

logger = logging.getLogger(__name__)

StreamEvent = tuple[str, dict]


class ChatSessionNotFound(Exception):
    """Raised when the requested chat session is missing or unavailable."""


@dataclass(frozen=True)
class StreamChatHandle:
    session_id: int
    events: Generator[StreamEvent, None, None]


def _get_or_create_session(
    db: Session,
    *,
    user_id: int,
    payload: ChatRequest,
) -> tuple[ChatSession, bool]:
    if payload.session_id is None:
        session = ChatSession(
            user_id=user_id,
            title="New chat",
            chat_mode=ChatMode.general,
            knowledge_base_id=None,
        )
        db.add(session)
        db.flush()
        return session, True

    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == payload.session_id, ChatSession.user_id == user_id)
        .one_or_none()
    )
    if session is None or session.is_deleted:
        raise ChatSessionNotFound("Session not found")

    is_first_message = (
        db.query(Message)
        .filter(Message.session_id == session.id, Message.user_id == user_id)
        .count()
        == 0
    )
    return session, is_first_message


def _save_user_message(
    db: Session,
    *,
    user_id: int,
    session_id: int,
    content: str,
) -> None:
    db.add(
        Message(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=content,
        )
    )
    db.flush()


def _save_assistant_message(
    db: Session,
    *,
    user_id: int,
    session_id: int,
    content: str,
    used_rag: bool,
    sources: list[dict] | None = None,
    tool_results: list[dict] | None = None,
) -> None:
    db.add(
        Message(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=content,
            used_rag=used_rag,
            sources=sources or None,
            tool_results=tool_results or None,
        )
    )


def _set_title_if_needed(session: ChatSession, *, is_first_message: bool, user_message: str) -> None:
    if not is_first_message:
        return
    try:
        session.title = generate_title(user_message)
    except Exception:
        logger.warning("Title generation failed", exc_info=True)


def _empty_answer_message() -> str:
    return "抱歉，本次没有生成有效回复，请稍后重试。"


def generate_title(first_user_message: str) -> str:
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
        return title[:80] if title else fallback
    except Exception:
        logger.warning("Title generation failed; using fallback", exc_info=True)
        return fallback


def _rag_meta_from_tool_payload(session_id: int, payload: dict) -> dict | None:
    if payload.get("tool") != "retrieve_knowledge_base":
        return None

    data = payload.get("data") or {}
    used_rag = bool(data.get("used_rag"))
    sources = list(data.get("sources") or [])
    notice = None if used_rag else payload.get("message")
    return {
        "session_id": session_id,
        "used_rag": used_rag,
        "sources": sources,
        "notice": notice,
    }


def _sources_event_from_tool_payload(session_id: int, payload: dict) -> dict | None:
    if payload.get("tool") == "retrieve_knowledge_base":
        data = payload.get("data") or {}
        sources = list(data.get("sources") or [])
        if not sources and not payload.get("message"):
            return None
        return {
            "session_id": session_id,
            "kind": "rag",
            "used_rag": bool(data.get("used_rag")),
            "sources": sources,
            "notice": None if data.get("used_rag") else payload.get("message"),
        }

    web_sources = extract_web_sources(tool_result_from_payload(payload))
    if not web_sources:
        return None
    return {
        "session_id": session_id,
        "kind": "web",
        "used_web": True,
        "web_sources": [
            source.model_dump(mode="json") for source in web_sources
        ],
    }


def _action_required_from_tool_payload(session_id: int, payload: dict) -> dict | None:
    data = payload.get("data") or {}
    action = data.get("action_required")
    if not action:
        return None
    return {
        "session_id": session_id,
        "action": action,
        "provider": data.get("provider"),
        "credential_type": data.get("credential_type"),
        "draft": data.get("draft"),
        "message": payload.get("message"),
    }


def _web_meta_from_result(session_id: int, agent_result) -> dict:
    return {
        "session_id": session_id,
        "used_rag": agent_result.used_rag,
        "sources": [
            source.model_dump(mode="json") for source in agent_result.sources
        ],
        "used_web": agent_result.used_web,
        "web_sources": [
            source.model_dump(mode="json") for source in agent_result.web_sources
        ],
        "notice": agent_result.notice,
    }


def start_stream_chat(
    *,
    user_id: int,
    payload: ChatRequest,
) -> StreamChatHandle:
    """Prepare a streaming chat turn and return a request-scoped event generator."""
    with session_scope() as db:
        session, is_first_message = _get_or_create_session(
            db,
            user_id=user_id,
            payload=payload,
        )
        session_id = session.id
        _save_user_message(
            db,
            user_id=user_id,
            session_id=session_id,
            content=payload.message,
        )

    events = _stream_chat_events(
        user_id=user_id,
        session_id=session_id,
        user_message=payload.message,
        is_first_message=is_first_message,
    )
    return StreamChatHandle(session_id=session_id, events=events)


def _stream_chat_events(
    *,
    user_id: int,
    session_id: int,
    user_message: str,
    is_first_message: bool,
) -> Generator[StreamEvent, None, None]:
    used_rag = False
    used_web = False
    sources_payload: list[dict] = []
    web_sources_payload: list[dict] = []
    notice: str | None = None
    full_text_parts: list[str] = []
    final_title: str | None = None

    yield (
        "meta",
        {
            "session_id": session_id,
            "used_rag": used_rag,
            "sources": sources_payload,
            "used_web": used_web,
            "web_sources": web_sources_payload,
            "notice": notice,
        },
    )

    with session_scope() as db:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .one_or_none()
        )
        if session is None or session.is_deleted:
            yield ("error", {"message": "Session not found"})
            return

        runner = get_agent_runner()
        tool_results_payload: list[dict] = []
        stream = runner.stream(
            db=db,
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
        )
        agent_result = None
        while True:
            try:
                kind, data = next(stream)
            except StopIteration as done:
                agent_result = done.value
                break

            if kind == "delta":
                content = data.get("content", "")
                if content:
                    full_text_parts.append(content)
                    yield ("delta", {"content": content})
            elif kind == "tool_result":
                payload = data.model_dump()
                tool_results_payload.append(payload)
                yield ("tool_result", payload)
                action_required = _action_required_from_tool_payload(
                    session_id,
                    payload,
                )
                if action_required is not None:
                    yield ("action_required", action_required)
                sources_event = _sources_event_from_tool_payload(session_id, payload)
                if sources_event is not None:
                    yield ("sources", sources_event)
                rag_meta = _rag_meta_from_tool_payload(session_id, payload)
                if rag_meta is not None:
                    used_rag = bool(rag_meta["used_rag"])
                    sources_payload = list(rag_meta["sources"])
                    notice = rag_meta["notice"]
                    yield ("meta", rag_meta)
            elif kind in {"agent_step", "tool_call"}:
                yield (kind, data)

        if agent_result is None:
            raise RuntimeError("Agent stream ended without a final result")

        full_text = (
            agent_result.final_answer.strip()
            or "".join(full_text_parts).strip()
            or _empty_answer_message()
        )
        used_rag = agent_result.used_rag
        used_web = agent_result.used_web
        sources_payload = [
            source.model_dump(mode="json") for source in agent_result.sources
        ]
        web_sources_payload = [
            source.model_dump(mode="json") for source in agent_result.web_sources
        ]
        notice = agent_result.notice
        if (used_rag or sources_payload or notice) and not any(
            item.get("tool") == "retrieve_knowledge_base"
            for item in tool_results_payload
        ):
            yield (
                "meta",
                {
                    "session_id": session_id,
                    "used_rag": used_rag,
                    "sources": sources_payload,
                    "used_web": used_web,
                    "web_sources": web_sources_payload,
                    "notice": notice,
                },
            )
        if used_web or web_sources_payload:
            yield ("meta", _web_meta_from_result(session_id, agent_result))

        _save_assistant_message(
            db,
            user_id=user_id,
            session_id=session_id,
            content=full_text,
            used_rag=used_rag,
            sources=sources_payload,
            tool_results=tool_results_payload,
        )
        _set_title_if_needed(
            session,
            is_first_message=is_first_message,
            user_message=user_message,
        )
        maybe_update_session_summary(
            db,
            user_id=user_id,
            session_id=session_id,
        )
        remember_turn(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=full_text,
        )
        final_title = session.title
        yield ("done", {"session_id": session_id, "title": final_title})
