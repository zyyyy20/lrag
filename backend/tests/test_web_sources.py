import unittest

from app.agents.tool_events import parse_tool_payload
from app.agents.web_sources import extract_web_sources
from app.schemas.session import MessageOut
from app.services.chat_workflow import _sources_event_from_tool_payload
from app.tools.types import ToolResult


class WebSourcesTests(unittest.TestCase):
    def test_parse_mcp_raw_json_result_preserves_tool_name_and_data(self):
        result = parse_tool_payload(
            '{"results":[{"title":"Example","url":"https://example.com"}]}',
            tool_name="tavily_search",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.tool, "tavily_search")
        self.assertTrue(result.ok)
        self.assertEqual(result.data["results"][0]["url"], "https://example.com")

    def test_extract_web_sources_from_tavily_search_results(self):
        result = ToolResult(
            tool="tavily_search",
            ok=True,
            message="",
            data={
                "results": [
                    {
                        "title": "上海海洋大学",
                        "url": "https://www.shou.edu.cn/",
                        "content": "上海海洋大学是...",
                        "score": 0.91,
                    }
                ]
            },
        )

        sources = extract_web_sources(result)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].title, "上海海洋大学")
        self.assertEqual(sources[0].url, "https://www.shou.edu.cn/")
        self.assertEqual(sources[0].source_tool, "tavily_search")

    def test_extract_web_sources_deduplicates_urls(self):
        result = ToolResult(
            tool="tavily_search",
            ok=True,
            message="",
            data={
                "results": [
                    {"title": "A", "url": "https://example.com"},
                    {"title": "B", "url": "https://example.com"},
                ]
            },
        )

        self.assertEqual(len(extract_web_sources(result)), 1)

    def test_message_out_restores_web_sources_from_saved_tool_results(self):
        message = MessageOut.model_validate(
            {
                "id": 1,
                "role": "assistant",
                "content": "answer",
                "created_at": "2026-05-17T10:00:00Z",
                "tool_results": [
                    {
                        "type": "tool_result",
                        "tool": "tavily_search",
                        "ok": True,
                        "message": "",
                        "data": {
                            "results": [
                                {
                                    "title": "Example",
                                    "url": "https://example.com",
                                }
                            ]
                        },
                    }
                ],
            }
        )

        self.assertTrue(message.used_web)
        self.assertEqual(message.web_sources[0].url, "https://example.com")

    def test_sources_event_from_tavily_tool_payload(self):
        event = _sources_event_from_tool_payload(
            7,
            {
                "type": "tool_result",
                "tool": "tavily_search",
                "ok": True,
                "message": "",
                "data": {
                    "results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                        }
                    ]
                },
            },
        )

        self.assertEqual(event["kind"], "web")
        self.assertTrue(event["used_web"])
        self.assertEqual(event["web_sources"][0]["url"], "https://example.com")

    def test_sources_event_from_rag_tool_payload(self):
        event = _sources_event_from_tool_payload(
            7,
            {
                "type": "tool_result",
                "tool": "retrieve_knowledge_base",
                "ok": True,
                "message": "",
                "data": {
                    "used_rag": True,
                    "sources": [
                        {
                            "filename": "doc.pdf",
                            "chunk_index": 1,
                            "score": 0.8,
                            "content_preview": "text",
                        }
                    ],
                },
            },
        )

        self.assertEqual(event["kind"], "rag")
        self.assertTrue(event["used_rag"])
        self.assertEqual(event["sources"][0]["filename"], "doc.pdf")


if __name__ == "__main__":
    unittest.main()
