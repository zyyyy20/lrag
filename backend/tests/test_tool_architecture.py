import unittest
from unittest.mock import Mock, patch


class ToolArchitectureTests(unittest.TestCase):
    def test_mcp_registry_builds_enabled_server_configs(self):
        from app.tools.mcp import build_enabled_mcp_server_configs

        settings = Mock(
            mcp_tavily_enabled=True,
            mcp_tavily_url="https://mcp.tavily.com/mcp/",
            mcp_tavily_api_key="tvly-test-key",
            mcp_tavily_transport="streamable_http",
        )

        configs = build_enabled_mcp_server_configs(settings)

        self.assertEqual(
            configs,
            {
                "tavily": {
                    "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-test-key",
                    "transport": "streamable_http",
                }
            },
        )

    def test_builder_combines_local_and_cached_mcp_tools_without_provider_checks(self):
        from app.tools import builder

        local_tool = Mock(name="local_tool")
        mcp_tool = Mock(name="mcp_tool")

        with (
            patch("app.tools.builder.build_local_tools", return_value=[local_tool]),
            patch("app.tools.builder.get_cached_mcp_tools", return_value=[mcp_tool]),
        ):
            self.assertEqual(builder.build_runtime_tools(), [local_tool, mcp_tool])

    def test_runner_helpers_are_split_into_agent_modules(self):
        from app.agents.prompts import build_base_system_prompt, build_system_prompt
        from app.agents.sources import extract_sources
        from app.agents.tool_events import tool_display_title

        self.assertIn("LRAG", build_base_system_prompt())
        self.assertIn("memory", build_system_prompt("base", "memory"))
        self.assertEqual(tool_display_title("tavily_search"), "联网搜索")
        self.assertEqual(extract_sources(Mock(tool="other", data={})), [])


if __name__ == "__main__":
    unittest.main()
