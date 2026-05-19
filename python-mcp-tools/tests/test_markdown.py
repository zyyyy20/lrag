import unittest

from mcp_tools_hub.publishing.markdown import render_markdown


class MarkdownTests(unittest.TestCase):
    def test_render_markdown_converts_markdown_to_html(self):
        html = render_markdown("# 标题\n\n- A\n- B")

        self.assertIn("<h1>标题</h1>", html)
        self.assertIn("<li>A</li>", html)
        self.assertIn("<li>B</li>", html)


if __name__ == "__main__":
    unittest.main()
