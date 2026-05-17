import unittest
from unittest.mock import Mock, patch


class LongTermMemoryTests(unittest.TestCase):
    def test_format_memories_for_prompt_returns_empty_for_no_memories(self):
        from app.services.long_term_memory import format_memories_for_prompt

        self.assertEqual(format_memories_for_prompt([]), "")

    def test_format_memories_for_prompt_extracts_memory_field(self):
        from app.services.long_term_memory import format_memories_for_prompt

        prompt = format_memories_for_prompt(
            [
                {"memory": "User prefers concise Chinese technical explanations."},
                {"content": "User works on a FastAPI project."},
                "User is evaluating long-term memory design.",
            ]
        )

        self.assertIn("长期记忆", prompt)
        self.assertIn("必须遵守", prompt)
        self.assertIn("User prefers concise Chinese technical explanations.", prompt)
        self.assertIn("User works on a FastAPI project.", prompt)
        self.assertIn("User is evaluating long-term memory design.", prompt)

    @patch("app.services.long_term_memory.get_settings")
    def test_search_user_memories_returns_empty_when_disabled(self, get_settings):
        from app.services.long_term_memory import search_user_memories

        get_settings.return_value = Mock(mem0_enabled=False)

        self.assertEqual(search_user_memories(1, "hello"), [])

    @patch("app.services.long_term_memory.get_settings")
    def test_search_user_memories_merges_query_and_preference_results(self, get_settings):
        from app.services.long_term_memory import search_user_memories

        get_settings.return_value = Mock(
            mem0_enabled=True,
            mem0_api_key="test-key",
            mem0_agent_id="lrag",
            mem0_top_k=5,
        )
        client = Mock()
        client.search.side_effect = [
            {"results": [{"memory": "Assistant is LRAG."}]},
            {"results": [{"memory": "Assistant must prepend 我的回答是： to every reply."}]},
        ]

        with patch("app.services.long_term_memory._get_client", return_value=client):
            memories = search_user_memories(1, "你是谁")

        for call in client.search.call_args_list:
            self.assertEqual(call.kwargs["filters"], {"user_id": "1"})
        self.assertEqual(
            memories,
            [
                "Assistant is LRAG.",
                "Assistant must prepend 我的回答是： to every reply.",
            ],
        )
        self.assertEqual(client.search.call_count, 2)

    @patch("app.services.long_term_memory.get_settings")
    def test_remember_turn_swallows_client_errors(self, get_settings):
        from app.services.long_term_memory import remember_turn

        get_settings.return_value = Mock(
            mem0_enabled=True,
            mem0_api_key="test-key",
            mem0_agent_id="lrag",
        )

        with patch("app.services.long_term_memory._get_client") as get_client:
            get_client.side_effect = RuntimeError("mem0 unavailable")
            with patch("app.services.long_term_memory.logger.warning") as warning:
                remember_turn(
                    user_id=1,
                    session_id=2,
                    user_message="Remember I prefer Chinese.",
                    assistant_message="Got it.",
                )
                warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
