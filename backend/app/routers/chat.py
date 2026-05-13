"""聊天相关 HTTP 接口：非流式 ``/api/chat`` 与 SSE 流式 ``/api/chat/stream``。

**非流式**：使用 ``Depends(get_db)``，在一次请求生命周期内完成：建会话（可选）、
写用户消息、调用 ``answer_question``、写助手消息、生成标题、``commit``。

**流式（SSE）**：**不能**在路由函数参数里依赖 ``get_db`` 生成器——FastAPI 会在
路由返回 ``StreamingResponse`` 后立即关闭 DB Session，而 body 生成器此时尚未
执行，会导致 ORM 对象脱离 Session。因此采用两阶段 ``session_scope``：
1. Phase1：短事务内创建/校验会话并持久化用户消息；
2. Phase2：在 ``event_generator`` 内新开长事务，跑 ``answer_question_stream``、
   写助手消息、更新标题，再 ``yield`` SSE ``done``。
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Generator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db, session_scope
from ..models import ChatMode, Message
from ..models import Session as ChatSession
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.chat import (
    answer_question,
    answer_question_stream,
    generate_title,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """非流式对话：一次请求返回完整 JSON（含 ``answer`` / ``used_rag`` / ``sources``）。

    流程：无 ``session_id`` 时创建普通会话；校验会话；写入用户消息；调用编排层；
    写入助手消息；首条消息时异步生成标题；提交事务。
    """
    if payload.session_id is None:
        session = ChatSession(title="新会话", chat_mode=ChatMode.general, knowledge_base_id=None)
        db.add(session)
        db.flush()
        is_first_message = True
    else:
        session = db.get(ChatSession, payload.session_id)
        if session is None or session.is_deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        is_first_message = (
            db.query(Message).filter(Message.session_id == session.id).count() == 0
        )

    user_msg = Message(session_id=session.id, role="user", content=payload.message)
    db.add(user_msg)
    db.flush()

    answer, used_rag, sources, notice = answer_question(db, session, payload.message)

    assistant_msg = Message(
        session_id=session.id,
        role="assistant",
        content=answer,
        used_rag=used_rag,
        sources=[s.model_dump(mode="json") for s in sources] if sources else None,
    )
    db.add(assistant_msg)

    if is_first_message:
        session.title = generate_title(payload.message)

    db.commit()

    return ChatResponse(
        session_id=session.id,
        answer=answer,
        used_rag=used_rag,
        sources=sources,
        notice=notice,
    )


def _sse(event: str, data: dict) -> str:
    """将事件名与 JSON 负载编码为一条标准 SSE 文本帧（以双换行结尾）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """SSE 流式对话：响应体为 ``text/event-stream``。

    事件约定：
    - ``meta``：``session_id``、``used_rag``、``sources``、``notice``（与前端首包对齐）；
    - ``delta``：增量 ``content``；
    - ``done``：``session_id``、``title``（标题可能刚被生成）；
    - ``error``：异常信息字符串。

    详见模块文档字符串中关于 DB Session 生命周期的说明。
    """
    session_id: uuid.UUID
    is_first_message: bool
    with session_scope() as db:
        if payload.session_id is None:
            new_session = ChatSession(
                title="新会话", chat_mode=ChatMode.general, knowledge_base_id=None
            )
            db.add(new_session)
            db.flush()
            session_id = new_session.id
            is_first_message = True
        else:
            existing = db.get(ChatSession, payload.session_id)
            if existing is None or existing.is_deleted:
                raise HTTPException(status_code=404, detail="Session not found")
            session_id = existing.id
            is_first_message = (
                db.query(Message).filter(Message.session_id == session_id).count() == 0
            )

        db.add(Message(session_id=session_id, role="user", content=payload.message))

    session_id_str = str(session_id)
    user_message_text = payload.message

    def event_generator() -> Generator[str, None, None]:
        used_rag = False
        sources_payload: list = []
        notice: str | None = None
        full_text_parts: list[str] = []
        final_title: str | None = None
        try:
            with session_scope() as db:
                session_obj = db.get(ChatSession, session_id)
                if session_obj is None or session_obj.is_deleted:
                    yield _sse("error", {"message": "Session not found"})
                    return

                for kind, data in answer_question_stream(db, session_obj, user_message_text):
                    if kind == "meta":
                        used_rag = bool(data.get("used_rag"))
                        sources_payload = list(data.get("sources") or [])
                        notice = data.get("notice")
                        yield _sse(
                            "meta",
                            {
                                "session_id": session_id_str,
                                "used_rag": used_rag,
                                "sources": sources_payload,
                                "notice": notice,
                            },
                        )
                    elif kind == "delta":
                        yield _sse("delta", {"content": data.get("content", "")})
                    elif kind == "final":
                        full_text_parts.append(data.get("content", ""))

                full_text = "".join(full_text_parts).strip()

                db.add(
                    Message(
                        session_id=session_id,
                        role="assistant",
                        content=full_text,
                        used_rag=used_rag,
                        sources=sources_payload or None,
                    )
                )

                if is_first_message:
                    try:
                        session_obj.title = generate_title(user_message_text)
                    except Exception:
                        logger.warning("Title generation failed", exc_info=True)

                final_title = session_obj.title

            yield _sse("done", {"session_id": session_id_str, "title": final_title})
        except Exception as e:
            logger.exception("Stream chat failed")
            yield _sse("error", {"message": str(e) or "internal_error"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
