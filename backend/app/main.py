"""FastAPI 应用入口。

组装：日志、CORS、全局异常处理、各业务路由；在 ``startup`` 事件中初始化数据库
（扩展 + 建表）。业务逻辑均在 ``routers`` 与 ``services`` 中实现，本文件保持
精简。
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import init_db
from .errors import register_exception_handlers
from .routers import chat, documents, health, knowledge_bases, sessions

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Lightweight RAG Chat", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(knowledge_bases.router)
app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.on_event("startup")
def on_startup() -> None:
    """应用进程启动时执行：创建 pgvector 扩展与 ORM 表（若不存在）。"""
    init_db()
