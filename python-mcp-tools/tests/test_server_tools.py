import unittest

from mcp_tools_hub.server import build_mcp_server


class ServerToolTests(unittest.TestCase):
    def test_build_mcp_server_registers_csdn_publish_tool(self):
        mcp = build_mcp_server()

        tool_names = {tool.name for tool in mcp._tool_manager.list_tools()}

        self.assertIn("publish_csdn_article", tool_names)


if __name__ == "__main__":
    unittest.main()
