import unittest

import httpx

from mcp_tools_hub.publishing.csdn_client import (
    CsdnArticle,
    CsdnClient,
    CsdnConfig,
    CsdnConfigError,
)


class CsdnClientTests(unittest.TestCase):
    def test_csdn_client_requires_cookie(self):
        with self.assertRaisesRegex(CsdnConfigError, "CSDN_COOKIE"):
            CsdnConfig(cookie="")

    def test_save_article_builds_csdn_payload_and_redacts_cookie(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "success",
                    "data": {
                        "id": 123,
                        "url": "https://blog.csdn.net/test/article/details/123",
                        "title": "LRAG MCP 实践",
                        "description": "对话总结",
                    },
                },
            )

        config = CsdnConfig(
            cookie="UserToken=secret-token; UserName=tester",
            category="AI 工程",
            default_tags=("AI", "MCP"),
        )
        client = CsdnClient(
            config,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        result = client.save_article(
            CsdnArticle(
                title="LRAG MCP 实践",
                markdown_content="# 标题\n\n正文",
                description="对话总结",
                tags=("RAG", "MCP"),
                publish_status="draft",
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.article_id, "123")
        self.assertEqual(
            result.url,
            "https://blog.csdn.net/test/article/details/123",
        )
        self.assertEqual(
            result.cookie_preview,
            "UserToken=<redacted>; UserName=<redacted>",
        )

        payload = requests[0].read().decode("utf-8")
        self.assertIn('"title":"LRAG MCP 实践"', payload)
        self.assertIn('"markdowncontent":"# 标题\\n\\n正文"', payload)
        self.assertIn('"content":"', payload)
        self.assertIn('"tags":"RAG,MCP"', payload)
        self.assertIn('"categories":"AI 工程"', payload)
        self.assertIn('"pubStatus":"draft"', payload)
        self.assertEqual(
            requests[0].headers["cookie"],
            "UserToken=secret-token; UserName=tester",
        )

    def test_save_article_returns_structured_error_for_http_failure(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"code": 401, "msg": "unauthorized"})

        client = CsdnClient(
            CsdnConfig(cookie="UserToken=expired"),
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        result = client.save_article(
            CsdnArticle(
                title="失败测试",
                markdown_content="正文",
                description="简介",
                tags=("AI",),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.code, 401)
        self.assertIn("unauthorized", result.message)


if __name__ == "__main__":
    unittest.main()
