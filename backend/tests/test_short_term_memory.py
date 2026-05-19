import unittest

from app.agents.langgraph_memory import build_input_messages
from app.agents.langgraph_runner import _runtime_config
from app.models import ChatMode, Message
from app.models import Session as ChatSession


class _FakeQuery:
    def __init__(self, value):
        self.value = value

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def all(self):
        return self.value

    def one_or_none(self):
        return self.value


class _FakeDb:
    def __init__(self, *, session, messages):
        self.session = session
        self.messages = messages

    def query(self, model):
        if model is ChatSession:
            return _FakeQuery(self.session)
        if model is Message:
            return _FakeQuery(self.messages)
        raise AssertionError(f"Unexpected model: {model!r}")


def _message(role, content, message_id):
    row = Message(
        id=message_id,
        user_id=1,
        session_id=10,
        role=role,
        content=content,
    )
    return row


class ShortTermMemoryTests(unittest.TestCase):
    def test_build_input_messages_uses_summary_and_recent_window(self):
        session = ChatSession(
            id=10,
            user_id=1,
            title="Memory test",
            chat_mode=ChatMode.general,
        )
        session.summary = "用户正在排查短期记忆设计。"
        messages = [
            _message("user", "old user 1", 1),
            _message("assistant", "old assistant 1", 2),
            _message("user", "recent user 1", 3),
            _message("assistant", "recent assistant 1", 4),
            _message("user", "recent user 2", 5),
            _message("assistant", "recent assistant 2", 6),
            _message("user", "current question", 7),
        ]
        db = _FakeDb(session=session, messages=messages)

        result = build_input_messages(
            db,
            user_id=1,
            session_id=10,
            current_user_message="current question",
            recent_turns=2,
        )

        self.assertEqual(
            result,
            [
                {
                    "role": "system",
                    "content": "当前会话摘要：\n用户正在排查短期记忆设计。",
                },
                {"role": "user", "content": "recent user 1"},
                {"role": "assistant", "content": "recent assistant 1"},
                {"role": "user", "content": "recent user 2"},
                {"role": "assistant", "content": "recent assistant 2"},
                {"role": "user", "content": "current question"},
            ],
        )

    def test_build_input_messages_always_includes_explicit_short_term_context(self):
        session = ChatSession(
            id=10,
            user_id=1,
            title="Memory test",
            chat_mode=ChatMode.general,
        )
        session.summary = "用户偏好中文回答。"
        db = _FakeDb(
            session=session,
            messages=[
                _message("user", "recent user", 1),
                _message("assistant", "recent assistant", 2),
            ],
        )

        result = build_input_messages(
            db,
            user_id=1,
            session_id=10,
            current_user_message="new question",
            recent_turns=1,
        )

        self.assertEqual(result[0]["role"], "system")
        self.assertIn("用户偏好中文回答", result[0]["content"])
        self.assertEqual(result[-1], {"role": "user", "content": "new question"})

    def test_runtime_config_uses_turn_scoped_checkpoint_thread(self):
        first = _runtime_config(user_id=1, session_id=10)
        second = _runtime_config(user_id=1, session_id=10)

        first_thread = first["configurable"]["thread_id"]
        second_thread = second["configurable"]["thread_id"]

        self.assertNotEqual(first_thread, second_thread)
        self.assertTrue(first_thread.startswith("user:1:session:10:turn:"))


if __name__ == "__main__":
    unittest.main()
