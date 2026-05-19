"""数据库连接与会话管理。

职责：
- 创建 SQLAlchemy ``Engine`` 与 ``SessionLocal``；
- 提供 FastAPI 依赖注入用的 ``get_db``（请求级会话，用完即关）；
- 提供 ``session_scope`` 上下文管理器（手动 commit/rollback，用于后台任务、
  SSE 流式响应等不能依赖 ``Depends(get_db)`` 生命周期的场景）；
- 启动时 ``init_db``：启用 pgvector 扩展并 ``create_all`` 建表。

注意：``Base.metadata.create_all`` 不会自动迁移已存在表的列变更；模型大改时需
配合迁移或清空数据卷重建。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # 连接池取出前 ping，避免用过期连接
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """所有 ORM 模型的声明式基类。"""

    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：为每个 HTTP 请求提供一个数据库 Session。

    Yields:
        绑定到当前请求的 ``Session``；请求结束后在 ``finally`` 中关闭。

    典型用法：路由函数参数 ``db: Session = Depends(get_db)``。
    注意：流式响应若在路由返回后仍需要访问 DB，不能使用本依赖，应改用
    ``session_scope`` 在生成器内部自行管理会话生命周期。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """手动管理的数据库会话上下文（带自动 commit/rollback）。

    成功退出 ``with`` 块时 ``commit``；块内抛出任意异常时 ``rollback`` 后
    重新抛出；最后始终 ``close``。

    Yields:
        一个短生命周期的 ``Session``。

    适用场景：
    - FastAPI ``BackgroundTasks`` 中的异步文档索引；
    - ``StreamingResponse`` 的 body 生成器（路由返回后 ``get_db`` 已关闭）。
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def ensure_pgvector() -> None:
    """确保 PostgreSQL 已安装并启用 ``vector`` 扩展（幂等）。"""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def ensure_session_memory_columns() -> None:
    """Add short-term memory columns when upgrading an existing development DB."""
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary TEXT"))
        conn.execute(
            text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary_message_id INTEGER")
        )


def init_db() -> None:
    """应用启动时调用：创建扩展并根据 ORM 元数据创建缺失的表。

    通过 ``from . import models`` 触发所有模型类注册到 ``Base.metadata``，
    再执行 ``create_all``。若表已存在则跳过，不会删表。
    """
    from . import models  # noqa: F401

    ensure_pgvector()
    Base.metadata.create_all(bind=engine)
    ensure_session_memory_columns()
