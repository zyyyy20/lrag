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
    init_db()
