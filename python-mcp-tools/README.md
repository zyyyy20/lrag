# Python MCP Tools

Reusable Python MCP tools for external integrations. The first provider is CSDN
article publishing over MCP stdio.

## Run

```bash
set CSDN_COOKIE=your-cookie
set CSDN_CATEGORY=AI Engineering
python -m mcp_tools_hub
```

The server communicates over stdio by default and exposes:

- `publish_csdn_article`

## LRAG stdio config example

```json
{
  "csdn": {
    "transport": "stdio",
    "command": "python",
    "args": ["-m", "mcp_tools_hub"],
    "cwd": "C:/Users/zy/Desktop/lrag/python-mcp-tools",
    "env": {
      "PYTHONPATH": "C:/Users/zy/Desktop/lrag/python-mcp-tools/src",
      "CSDN_COOKIE": "your-cookie",
      "CSDN_CATEGORY": "AI Engineering"
    }
  }
}
```
