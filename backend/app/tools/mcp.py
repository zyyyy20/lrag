"""MCP tool registry, loading, and sync adaptation."""
from __future__ import annotations

import asyncio
import logging
from threading import RLock, Thread
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langchain_core.tools import StructuredTool

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

McpToolLoader = Callable[[dict[str, dict[str, str]]], Awaitable[list[Any]]]

_cached_mcp_tools: list[Any] = []
_cache_lock = RLock()


def build_mcp_server_config(
    *,
    url: str,
    transport: str,
    query_params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    for key, value in (query_params or {}).items():
        if value and key not in query:
            query[key] = value
    config: dict[str, Any] = {
        "url": urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        ),
        "transport": transport,
    }
    if headers:
        config["headers"] = headers
    return config


def build_enabled_mcp_server_configs(settings: Settings) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    if settings.mcp_tavily_enabled and (
        settings.mcp_tavily_api_key or "tavilyApiKey=" in settings.mcp_tavily_url
    ):
        configs["tavily"] = build_mcp_server_config(
            url=settings.mcp_tavily_url,
            transport=settings.mcp_tavily_transport,
            query_params={"tavilyApiKey": settings.mcp_tavily_api_key},
        )
    return configs


async def _load_mcp_tools(server_configs: dict[str, dict[str, Any]]) -> list[Any]:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ModuleNotFoundError as exc:
        raise RuntimeError("langchain-mcp-adapters package is not installed") from exc

    client = MultiServerMCPClient(server_configs)
    return list(await client.get_tools())


def _run_async(coro: Awaitable[list[Any]]) -> list[Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] | None = None
    error: BaseException | None = None

    def runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive bridge
            error = exc

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return result or []


def _run_async_value(coro: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Any = None
    error: BaseException | None = None

    def runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive bridge
            error = exc

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return result


def _sync_wrap_tool(tool: Any) -> Any:
    if not isinstance(tool, StructuredTool) or getattr(tool, "func", None) is not None:
        return tool

    coroutine = getattr(tool, "coroutine", None)
    if coroutine is None:
        return tool

    def sync_func(**kwargs: Any) -> Any:
        return _run_async_value(coroutine(**kwargs))

    return StructuredTool.from_function(
        func=sync_func,
        coroutine=coroutine,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        return_direct=tool.return_direct,
        response_format=tool.response_format,
    )


def _sync_wrap_tools(tools: list[Any]) -> list[Any]:
    return [_sync_wrap_tool(tool) for tool in tools]


def clear_mcp_tools() -> None:
    with _cache_lock:
        _cached_mcp_tools.clear()


def get_cached_mcp_tools() -> list[Any]:
    with _cache_lock:
        return list(_cached_mcp_tools)


def initialize_mcp_tools(
    *,
    settings: Settings | None = None,
    loader: McpToolLoader = _load_mcp_tools,
) -> None:
    settings = settings or get_settings()
    server_configs = build_enabled_mcp_server_configs(settings)
    if not server_configs:
        if settings.mcp_tavily_enabled:
            logger.warning("Tavily MCP is enabled but MCP_TAVILY_API_KEY is empty")
        clear_mcp_tools()
        return

    try:
        tools = _sync_wrap_tools(_run_async(loader(server_configs)))
    except Exception:
        logger.exception("Failed to initialize MCP tools")
        clear_mcp_tools()
        return

    with _cache_lock:
        _cached_mcp_tools[:] = tools
    logger.info("Initialized %d MCP tool(s)", len(tools))
