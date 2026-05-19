import unittest
import json
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
        from app.tools import mcp

        settings = Mock(
            mcp_servers_json="",
            mcp_tavily_enabled=True,
            mcp_tavily_url="https://mcp.tavily.com/mcp/",
            mcp_tavily_api_key="tvly-test-key",
            mcp_tavily_transport="streamable_http",
        )
        loaded_tool = Mock(name="tavily_search")

        async def fake_loader(server_configs):
            self.assertIn("tavily", server_configs)
            return [loaded_tool]

        mcp.clear_mcp_tools()
        mcp.initialize_mcp_tools(settings=settings, loader=fake_loader)

        self.assertEqual(mcp.get_cached_mcp_tools(), [loaded_tool])

    def test_initialize_mcp_tools_wraps_async_only_tools_for_sync_agent(self):
        from langchain_core.tools import StructuredTool

        from app.tools import mcp

        async def async_tool(query: str):
            return {"answer": query}

        tool = StructuredTool.from_function(
            coroutine=async_tool,
            name="async_only",
            description="Async only test tool.",
            args_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        settings = Mock(
            mcp_servers_json="",
            mcp_tavily_enabled=True,
            mcp_tavily_url="https://mcp.tavily.com/mcp/",
            mcp_tavily_api_key="tvly-test-key",
            mcp_tavily_transport="streamable_http",
        )

        async def fake_loader(_server_configs):
            return [tool]

        mcp.clear_mcp_tools()
        mcp.initialize_mcp_tools(settings=settings, loader=fake_loader)
        wrapped = mcp.get_cached_mcp_tools()[0]

        self.assertEqual(wrapped.invoke({"query": "hello"}), {"answer": "hello"})

    def test_build_enabled_mcp_server_configs_merges_generic_stdio_servers(self):
        from app.tools.mcp import build_enabled_mcp_server_configs

        settings = Mock(
            mcp_servers_json=json.dumps(
                {
                    "custom_tool_hub": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["-m", "mcp_tools_hub"],
                        "cwd": "C:/Users/zy/Desktop/lrag/python-mcp-tools",
                        "env": {
                            "PYTHONPATH": "C:/Users/zy/Desktop/lrag/python-mcp-tools/src",
                            "CUSTOM_TOOL_TOKEN": "token-value",
                        },
                    }
                }
            ),
            mcp_tavily_enabled=False,
            mcp_tavily_url="https://mcp.tavily.com/mcp/",
            mcp_tavily_api_key="",
            mcp_tavily_transport="streamable_http",
        )

        configs = build_enabled_mcp_server_configs(settings)

        self.assertEqual(configs["custom_tool_hub"]["transport"], "stdio")
        self.assertEqual(configs["custom_tool_hub"]["command"], "python")
        self.assertEqual(configs["custom_tool_hub"]["args"], ["-m", "mcp_tools_hub"])
        self.assertEqual(
            configs["custom_tool_hub"]["env"]["PYTHONPATH"],
            "C:/Users/zy/Desktop/lrag/python-mcp-tools/src",
        )

    def test_build_agent_tools_includes_cached_mcp_tools_when_enabled(self):
        from app.tools.core.manager import build_agent_tools

        cached_tool = Mock(name="cached_mcp_tool")

        with (
            patch("app.tools.core.manager.build_local_tools", return_value=[]),
            patch("app.tools.core.manager.get_cached_mcp_tools", return_value=[cached_tool]),
        ):
            tools = build_agent_tools()

        self.assertIn(cached_tool, tools)


if __name__ == "__main__":
    unittest.main()
