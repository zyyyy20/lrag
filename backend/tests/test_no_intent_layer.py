import unittest

from app.agents.langgraph_runner import LangGraphAgentRunner
from app.services import chat_workflow


class NoIntentLayerTests(unittest.TestCase):
    def test_runner_does_not_expose_intent_graph(self):
        self.assertFalse(hasattr(LangGraphAgentRunner, "_create_intent_graph"))

    def test_chat_workflow_does_not_emit_intent_payloads(self):
        self.assertFalse(hasattr(chat_workflow, "_intent_event_payload"))


if __name__ == "__main__":
    unittest.main()
