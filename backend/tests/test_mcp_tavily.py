import unittest
from unittest.mock import Mock, patch


class McpTavilyTests(unittest.TestCase):
    def test_build_mcp_server_config_uses_query_params(self):
        from app.tools.mcp import build_mcp_server_config

        config = build_mcp_server_config(
            url="https://mcp.tavily.com/mcp/?existing=1",
            transport="streamable_http",
            query_params={"tavilyApiKey": "tvly-test-key"},
        )

        self.assertEqual(config["transport"], "streamable_http")
        self.assertEqual(
            config["url"],
            "https://mcp.tavily.com/mcp/?existing=1&tavilyApiKey=tvly-test-key",
        )

    def test_initialize_mcp_tools_caches_loaded_tools(self):
        from app.services import mcp_tools

        settings = Mock(
            mcp_tavily_enabled=True,
            mcp_tavily_url="https://mcp.tavily.com/mcp/",
            mcp_tavily_api_key="tvly-test-key",
            mcp_tavily_transport="streamable_http",
        )
        loaded_tool = Mock(name="tavily_search")

        async def fake_loader(server_configs):
            self.assertIn("tavily", server_configs)
            return [loaded_tool]

        mcp_tools.clear_mcp_tools()
        mcp_tools.initialize_mcp_tools(settings=settings, loader=fake_loader)

        self.assertEqual(mcp_tools.get_cached_mcp_tools(), [loaded_tool])

    def test_initialize_mcp_tools_wraps_async_only_tools_for_sync_agent(self):
        from langchain_core.tools import StructuredTool

        from app.services import mcp_tools

        async def async_tool(query: str):
            return {"answer": query}

        tool = StructuredTool.from_function(
            coroutine=async_tool,
            name="async_only",
            description="Async only test tool.",
            args_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        settings = Mock(
            mcp_tavily_enabled=True,
            mcp_tavily_url="https://mcp.tavily.com/mcp/",
            mcp_tavily_api_key="tvly-test-key",
            mcp_tavily_transport="streamable_http",
        )

        async def fake_loader(_server_configs):
            return [tool]

        mcp_tools.clear_mcp_tools()
        mcp_tools.initialize_mcp_tools(settings=settings, loader=fake_loader)
        wrapped = mcp_tools.get_cached_mcp_tools()[0]

        self.assertEqual(wrapped.invoke({"query": "hello"}), {"answer": "hello"})

    def test_build_runtime_tools_includes_cached_mcp_tools_when_enabled(self):
        from app.tools import builder

        cached_tool = Mock(name="cached_mcp_tool")

        with (
            patch("app.tools.builder.build_local_tools", return_value=[]),
            patch("app.tools.builder.get_cached_mcp_tools", return_value=[cached_tool]),
        ):
            tools = builder.build_runtime_tools()

        self.assertIn(cached_tool, tools)


if __name__ == "__main__":
    unittest.main()
