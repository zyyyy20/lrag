from __future__ import annotations

from markdown_it import MarkdownIt


def render_markdown(markdown_content: str) -> str:
    return MarkdownIt("commonmark").render(markdown_content or "")
