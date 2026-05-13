"""健康检查接口：用于编排与负载均衡探活。"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import get_db

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health(db: Session = Depends(get_db)) -> dict:
    """执行 ``SELECT 1`` 探测数据库连通性。

    Returns:
        ``{"status": "ok"|"degraded", "database": bool}`` —— 数据库异常时
        ``status`` 为 ``degraded`` 且 ``database`` 为 ``False``。
    """
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}
