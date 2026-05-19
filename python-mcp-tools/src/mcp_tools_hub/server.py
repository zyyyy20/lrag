from mcp.server.fastmcp import FastMCP

from .publishing.csdn_client import CsdnArticle, CsdnClient, CsdnConfig


def build_mcp_server() -> FastMCP:
    mcp = FastMCP(
        "python-mcp-tools",
        instructions=(
            "Reusable external tools. Publishing tools may perform side effects; "
            "call them only after the user has explicitly confirmed."
        ),
    )

    @mcp.tool(
        name="publish_csdn_article",
        description=(
            "Save a Markdown article to CSDN. Defaults to draft; use only after "
            "the user confirms the article title, description, tags, and content."
        ),
    )
    def publish_csdn_article(
        title: str,
        markdown_content: str,
        description: str,
        tags: list[str],
        category: str | None = None,
        publish_status: str = "draft",
    ) -> dict:
        config = CsdnConfig.from_env()
        client = CsdnClient(config)
        result = client.save_article(
            CsdnArticle(
                title=title,
                markdown_content=markdown_content,
                description=description,
                tags=tuple(tags),
                category=category,
                publish_status=publish_status,
            )
        )
        return result.to_dict()

    return mcp


def main() -> None:
    build_mcp_server().run(transport="stdio")
