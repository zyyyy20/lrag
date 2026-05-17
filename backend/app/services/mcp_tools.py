"""MCP tool loading and process-wide cache."""
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


def build_tavily_server_config(
    url: str,
    api_key: str,
    transport: str,
) -> dict[str, str]:
    """Build a Tavily MCP server config for langchain-mcp-adapters."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    if api_key and "tavilyApiKey" not in query:
        query["tavilyApiKey"] = api_key
    full_url = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )
    return {"url": full_url, "transport": transport}


async def _load_mcp_tools(server_configs: dict[str, dict[str, str]]) -> list[Any]:
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
    if not settings.mcp_tavily_enabled:
        clear_mcp_tools()
        return
    if not settings.mcp_tavily_api_key and "tavilyApiKey=" not in settings.mcp_tavily_url:
        logger.warning("Tavily MCP is enabled but MCP_TAVILY_API_KEY is empty")
        clear_mcp_tools()
        return

    server_configs = {
        "tavily": build_tavily_server_config(
            settings.mcp_tavily_url,
            settings.mcp_tavily_api_key,
            settings.mcp_tavily_transport,
        )
    }
    try:
        tools = _sync_wrap_tools(_run_async(loader(server_configs)))
    except Exception:
        logger.exception("Failed to initialize MCP tools")
        clear_mcp_tools()
        return

    with _cache_lock:
        _cached_mcp_tools[:] = tools
    logger.info("Initialized %d MCP tool(s)", len(tools))
