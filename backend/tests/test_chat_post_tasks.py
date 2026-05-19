import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from app.models import ChatMode
from app.models import Session as ChatSession
from app.services import chat_post_tasks


class _FakeQuery:
    def __init__(self, value):
        self.value = value

    def filter(self, *args, **kwargs):
        return self

    def one_or_none(self):
        return self.value


class _FakeDb:
    def __init__(self, session):
        self.session = session

    def query(self, model):
        if model is ChatSession:
            return _FakeQuery(self.session)
        raise AssertionError(f"Unexpected model: {model!r}")


@contextmanager
def _fake_session_scope(db):
    yield db


class ChatPostTaskTests(unittest.TestCase):
    def test_run_chat_post_tasks_updates_title_summary_and_long_term_memory(self):
        session = ChatSession(
            id=10,
            user_id=1,
            title="New chat",
            chat_mode=ChatMode.general,
        )
        db = _FakeDb(session)

        with (
            patch.object(chat_post_tasks, "session_scope", lambda: _fake_session_scope(db)),
            patch.object(chat_post_tasks, "set_title_if_needed") as set_title,
            patch.object(chat_post_tasks, "maybe_update_session_summary") as update_summary,
            patch.object(chat_post_tasks, "remember_turn") as remember,
        ):
            chat_post_tasks.run_chat_post_tasks(
                user_id=1,
                session_id=10,
                user_message="hello",
                assistant_message="answer",
                is_first_message=True,
            )

        set_title.assert_called_once_with(
            session,
            is_first_message=True,
            user_message="hello",
        )
        update_summary.assert_called_once_with(db, user_id=1, session_id=10)
        remember.assert_called_once_with(
            user_id=1,
            session_id=10,
            user_message="hello",
            assistant_message="answer",
        )

    def test_schedule_chat_post_tasks_submits_to_executor(self):
        executor = Mock()

        with patch.object(chat_post_tasks, "_executor", executor):
            chat_post_tasks.schedule_chat_post_tasks(
                user_id=1,
                session_id=10,
                user_message="hello",
                assistant_message="answer",
                is_first_message=True,
            )

        executor.submit.assert_called_once_with(
            chat_post_tasks.run_chat_post_tasks,
            user_id=1,
            session_id=10,
            user_message="hello",
            assistant_message="answer",
            is_first_message=True,
        )


if __name__ == "__main__":
    unittest.main()
