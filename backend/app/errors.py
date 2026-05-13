"""API 统一异常处理。

将常见异常类型映射为 JSON 响应体 ``{"error": {"code", "message", "detail?"}}``，
便于前端与监控统一解析。未捕获的异常记录完整堆栈后返回 500，不向客户端泄露
内部实现细节。
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def _payload(code: str, message: str, detail=None) -> dict:
    """构造标准错误响应体。"""
    body = {"error": {"code": code, "message": message}}
    if detail is not None:
        body["error"]["detail"] = detail
    return body


def register_exception_handlers(app: FastAPI) -> None:
    """向 FastAPI 应用注册全局异常处理器。

    Args:
        app: 已创建的 ``FastAPI`` 实例。

    注册的处理器：
    - ``HTTPException``：透传状态码与 ``detail``；
    - ``RequestValidationError``：422，附带 Pydantic 校验错误列表；
    - ``ValueError``：视为客户端错误 400；
    - 其它 ``Exception``：500，日志记录完整异常链。
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload("http_error", str(exc.detail)),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_payload("validation_error", "Request validation failed", exc.errors()),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content=_payload("bad_request", str(exc)),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_payload("internal_error", "Internal server error"),
        )
