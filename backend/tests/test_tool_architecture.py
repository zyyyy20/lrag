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

    def test_tool_manager_exposes_agent_tools_from_local_and_mcp_providers(self):
        from app.tools.core.manager import ToolManager

        local_tool = Mock(name="local_tool")
        mcp_tool = Mock(name="mcp_tool")

        manager = ToolManager(
            local_provider=lambda: [local_tool],
            mcp_provider=lambda: [mcp_tool],
        )

        self.assertEqual(manager.get_agent_tools(), [local_tool, mcp_tool])

    def test_tool_manager_exposes_provider_metadata(self):
        from app.tools.core.manager import ToolManager

        manager = ToolManager(local_provider=lambda: [], mcp_provider=lambda: [])

        providers = manager.list_providers()

        self.assertEqual([provider.name for provider in providers], ["local", "mcp"])
        self.assertEqual([provider.category for provider in providers], ["built_in", "external"])

    def test_tool_results_module_owns_payload_and_source_extraction(self):
        from app.tools.core.results import (
            extract_sources,
            extract_web_sources,
            parse_tool_payload,
        )

        rag_result = parse_tool_payload(
            {
                "tool": "retrieve_knowledge_base",
                "ok": True,
                "message": "ok",
                "data": {
                    "sources": [
                        {
                            "knowledge_base_id": 1,
                            "document_id": 2,
                            "filename": "Doc",
                            "chunk_index": 0,
                            "score": 0.9,
                            "content_preview": "chunk",
                        }
                    ]
                },
            }
        )
        web_result = parse_tool_payload(
            {
                "tool": "tavily_search",
                "ok": True,
                "message": "ok",
                "data": {
                    "results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                            "content": "preview",
                        }
                    ]
                },
            }
        )

        self.assertEqual(extract_sources(rag_result)[0].filename, "Doc")
        self.assertEqual(extract_web_sources(web_result)[0].url, "https://example.com")

    def test_tools_are_grouped_into_core_local_and_mcp_packages(self):
        from app.tools.core.manager import ToolManager
        from app.tools.core.results import parse_tool_payload
        from app.tools.local import build_local_tools
        from app.tools.mcp import build_enabled_mcp_server_configs

        self.assertIsNotNone(ToolManager)
        self.assertIsNotNone(parse_tool_payload)
        self.assertIsNotNone(build_local_tools)
        self.assertIsNotNone(build_enabled_mcp_server_configs)

    def test_runner_helpers_are_split_into_agent_modules(self):
        from app.agents.prompts import build_base_system_prompt, build_system_prompt
        from app.agents.tool_events import tool_display_title
        from app.tools.core.results import extract_sources

        self.assertIn("LRAG", build_base_system_prompt())
        self.assertIn("memory", build_system_prompt("base", "memory"))
        self.assertEqual(tool_display_title("tavily_search"), "联网搜索")
        self.assertEqual(extract_sources(Mock(tool="other", data={})), [])


if __name__ == "__main__":
    unittest.main()
