import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from app.agents.types import AgentRunResult
from app.models import ChatMode
from app.models import Session as ChatSession
from app.services import chat_workflow
from app.tools.core.types import ToolResult


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
        self.added = []

    def query(self, model):
        if model is ChatSession:
            return _FakeQuery(self.session)
        raise AssertionError(f"Unexpected model: {model!r}")

    def add(self, value):
        self.added.append(value)


class _FakeRunner:
    def __init__(self, tool_result=None):
        self.tool_result = tool_result

    def stream(self, **kwargs):
        yield ("delta", {"content": "answer"})
        if self.tool_result is not None:
            yield ("tool_result", self.tool_result)
        return AgentRunResult(
            final_answer="answer",
            used_rag=self.tool_result is not None,
            sources=[],
            used_web=False,
            web_sources=[],
            notice=None,
        )


@contextmanager
def _fake_session_scope(db):
    yield db


@contextmanager
def _tracked_session_scope(db, state):
    try:
        yield db
    finally:
        state["closed"] = True


class ChatStreamPostTaskTests(unittest.TestCase):
    def test_stream_emits_started_sources_completed_meta_without_intermediate_meta(self):
        session = ChatSession(
            id=10,
            user_id=1,
            title="New chat",
            chat_mode=ChatMode.general,
        )
        db = _FakeDb(session)
        tool_result = ToolResult(
            tool="retrieve_knowledge_base",
            ok=True,
            message="found",
            data={
                "used_rag": True,
                "sources": [
                    {
                        "knowledge_base_id": 1,
                        "document_id": 2,
                        "filename": "doc.md",
                        "chunk_index": 0,
                        "score": 0.9,
                        "content_preview": "preview",
                    }
                ],
            },
        )

        with (
            patch.object(chat_workflow, "session_scope", lambda: _fake_session_scope(db)),
            patch.object(chat_workflow, "get_agent_runner", return_value=_FakeRunner(tool_result)),
            patch.object(chat_workflow, "schedule_chat_post_tasks"),
        ):
            events = list(
                chat_workflow._stream_chat_events(
                    user_id=1,
                    session_id=10,
                    user_message="hello",
                    is_first_message=True,
                )
            )

        meta_events = [data for kind, data in events if kind == "meta"]
        self.assertEqual([item["phase"] for item in meta_events], ["started", "completed"])
        source_events = [data for kind, data in events if kind == "sources"]
        self.assertEqual(source_events[0]["kind"], "rag")

    def test_stream_schedules_title_summary_and_mem0_after_persisting_answer(self):
        session = ChatSession(
            id=10,
            user_id=1,
            title="New chat",
            chat_mode=ChatMode.general,
        )
        db = _FakeDb(session)
        scheduled = Mock()

        with (
            patch.object(chat_workflow, "session_scope", lambda: _fake_session_scope(db)),
            patch.object(chat_workflow, "get_agent_runner", return_value=_FakeRunner()),
            patch.object(chat_workflow, "schedule_chat_post_tasks", scheduled),
        ):
            events = list(
                chat_workflow._stream_chat_events(
                    user_id=1,
                    session_id=10,
                    user_message="hello",
                    is_first_message=True,
                )
            )

        self.assertEqual(events[-1][0], "done")
        self.assertFalse(hasattr(chat_workflow, "_set_title_if_needed"))
        self.assertFalse(hasattr(chat_workflow, "maybe_update_session_summary"))
        self.assertFalse(hasattr(chat_workflow, "remember_turn"))
        scheduled.assert_called_once_with(
            user_id=1,
            session_id=10,
            user_message="hello",
            assistant_message="answer",
            is_first_message=True,
        )

    def test_stream_schedules_post_tasks_after_main_session_scope_exits(self):
        session = ChatSession(
            id=10,
            user_id=1,
            title="New chat",
            chat_mode=ChatMode.general,
        )
        db = _FakeDb(session)
        state = {"closed": False}

        def scheduled(**kwargs):
            self.assertTrue(state["closed"])

        with (
            patch.object(
                chat_workflow,
                "session_scope",
                lambda: _tracked_session_scope(db, state),
            ),
            patch.object(chat_workflow, "get_agent_runner", return_value=_FakeRunner()),
            patch.object(chat_workflow, "schedule_chat_post_tasks", scheduled),
        ):
            events = list(
                chat_workflow._stream_chat_events(
                    user_id=1,
                    session_id=10,
                    user_message="hello",
                    is_first_message=True,
                )
            )

        self.assertEqual(events[-1][0], "done")


if __name__ == "__main__":
    unittest.main()
