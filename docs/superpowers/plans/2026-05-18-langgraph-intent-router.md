# LangGraph Intent Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LangGraph-native intent recognition and routing layer so each chat request is classified before tool, RAG, web, memory, or direct-answer execution.

**Architecture:** Introduce a typed `IntentDecision` model, an intent classifier node, a router node, and explicit downstream branches inside the existing LangGraph runner. Preserve the current SSE behavior by emitting an early `intent` event, then continuing to stream `tool_call`, `tool_result`, `sources`, `delta`, and `done` events as today.

**Tech Stack:** FastAPI, LangGraph, LangChain, Pydantic, SQLAlchemy, Mem0, MCP tools, Tavily, unittest, Next.js, TypeScript, SSE.

---

## Target Flow

```text
chat_workflow._stream_chat_events
  -> runner.stream(...)
    -> intent_classifier node
    -> route_by_intent node
      -> memory_context node
      -> direct_chat / rag_agent / web_agent / tool_agent / mixed_agent / clarify
    -> stream events back to chat_workflow
  -> save assistant message
  -> remember turn
  -> done
```

## Intent Types

Use these first-version intent values:

```text
direct_chat
rag_qa
web_search
memory_query
memory_update
tool_task
summary
mixed
clarify
```

## Files

- Create: `backend/app/schemas/intent.py`
  - Owns typed intent values, confidence, routing flags, recommended tools, rewritten query, and explanation.
- Create: `backend/app/agents/intent.py`
  - Owns rule-based pre-classification, LLM fallback classification, and prompt text.
- Create: `backend/tests/test_intent_classifier.py`
  - Covers deterministic rules, JSON parsing, fallback behavior, and confidence thresholds.
- Modify: `backend/app/agents/types.py`
  - Add `intent: IntentDecision | None` to `AgentRunResult`.
- Modify: `backend/app/agents/prompts.py`
  - Add intent-aware prompt section builder.
- Modify: `backend/app/agents/langgraph_runner.py`
  - Replace the current single `create_agent(...)` stream path with a compiled graph containing intent classification and routing.
- Modify: `backend/app/services/chat_workflow.py`
  - Forward early `intent` events to the frontend and keep existing final save behavior.
- Modify: `backend/app/schemas/chat.py`
  - Add optional intent payload to streaming/chat response schemas if needed by the frontend.
- Modify: `frontend/lib/types.ts`
  - Add `IntentDecision` and `ChatStreamIntent`.
- Modify: `frontend/lib/api.ts`
  - Parse `event: intent`.
- Modify: `frontend/app/page.tsx`
  - Store intent event in the current assistant message trace.
- Modify: `frontend/components/ChatArea.tsx`
  - Render intent trace as an Agent step.

---

## Task 1: Add Intent Schema

**Files:**
- Create: `backend/app/schemas/intent.py`
- Test: `backend/tests/test_intent_classifier.py`

- [ ] **Step 1: Write failing schema tests**

Add tests that assert:

```python
from app.schemas.intent import IntentDecision, IntentType


def test_intent_decision_defaults():
    decision = IntentDecision(intent=IntentType.DIRECT_CHAT)

    assert decision.intent == IntentType.DIRECT_CHAT
    assert decision.confidence == 0.0
    assert decision.need_rag is False
    assert decision.need_web is False
    assert decision.need_memory is False
    assert decision.tools == []
    assert decision.rewrite_query is None


def test_intent_decision_clamps_confidence():
    low = IntentDecision(intent=IntentType.DIRECT_CHAT, confidence=-1)
    high = IntentDecision(intent=IntentType.WEB_SEARCH, confidence=2)

    assert low.confidence == 0.0
    assert high.confidence == 1.0
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
$env:PYTHONPATH='C:\Users\zy\Desktop\lrag\backend'; .\.venv\Scripts\python.exe -m unittest backend.tests.test_intent_classifier
```

Expected: import failure because `app.schemas.intent` does not exist.

- [ ] **Step 3: Implement schema**

Create `backend/app/schemas/intent.py`:

```python
"""Intent recognition schemas."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class IntentType(StrEnum):
    DIRECT_CHAT = "direct_chat"
    RAG_QA = "rag_qa"
    WEB_SEARCH = "web_search"
    MEMORY_QUERY = "memory_query"
    MEMORY_UPDATE = "memory_update"
    TOOL_TASK = "tool_task"
    SUMMARY = "summary"
    MIXED = "mixed"
    CLARIFY = "clarify"


class IntentDecision(BaseModel):
    intent: IntentType
    confidence: float = 0.0
    need_rag: bool = False
    need_web: bool = False
    need_memory: bool = False
    tools: list[str] = Field(default_factory=list)
    rewrite_query: str | None = None
    reason: str = ""

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
```

- [ ] **Step 4: Run test and verify pass**

Run the same unittest command. Expected: pass.

---

## Task 2: Add Rule-Based Intent Classifier

**Files:**
- Create: `backend/app/agents/intent.py`
- Test: `backend/tests/test_intent_classifier.py`

- [ ] **Step 1: Add failing tests for obvious intents**

Add tests:

```python
from app.agents.intent import classify_intent_by_rules
from app.schemas.intent import IntentType


def test_rule_classifies_web_search():
    decision = classify_intent_by_rules("联网搜索一下今天上海天气")

    assert decision is not None
    assert decision.intent == IntentType.WEB_SEARCH
    assert decision.need_web is True
    assert "tavily_search" in decision.tools


def test_rule_classifies_rag_question():
    decision = classify_intent_by_rules("根据知识库回答这个文档讲了什么")

    assert decision is not None
    assert decision.intent == IntentType.RAG_QA
    assert decision.need_rag is True


def test_rule_returns_none_for_ambiguous_input():
    assert classify_intent_by_rules("这个怎么看") is None
```

- [ ] **Step 2: Implement minimal rules**

Create `backend/app/agents/intent.py` with:

```python
"""Intent classification helpers for LangGraph routing."""
from __future__ import annotations

from ..schemas.intent import IntentDecision, IntentType


WEB_KEYWORDS = ("联网", "搜索", "最新", "今天", "现在", "新闻", "官网", "网址")
RAG_KEYWORDS = ("知识库", "文档", "上传", "资料", "文件里", "根据文件")
MEMORY_QUERY_KEYWORDS = ("还记得", "之前说过", "我以前", "历史记忆")
MEMORY_UPDATE_KEYWORDS = ("记住", "以后都", "我的偏好", "我喜欢", "我不喜欢")
SUMMARY_KEYWORDS = ("总结", "归纳", "概括")
TOOL_KEYWORDS = ("生成发票", "开发票", "生成html", "生成 HTML", "做一个页面")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def classify_intent_by_rules(message: str) -> IntentDecision | None:
    text = message.strip()
    if not text:
        return IntentDecision(
            intent=IntentType.CLARIFY,
            confidence=1.0,
            reason="用户输入为空，需要追问。",
        )

    if _contains_any(text, WEB_KEYWORDS):
        return IntentDecision(
            intent=IntentType.WEB_SEARCH,
            confidence=0.9,
            need_web=True,
            need_memory=True,
            tools=["tavily_search"],
            rewrite_query=text,
            reason="命中联网搜索关键词。",
        )

    if _contains_any(text, RAG_KEYWORDS):
        return IntentDecision(
            intent=IntentType.RAG_QA,
            confidence=0.85,
            need_rag=True,
            need_memory=True,
            rewrite_query=text,
            reason="命中知识库问答关键词。",
        )

    if _contains_any(text, MEMORY_QUERY_KEYWORDS):
        return IntentDecision(
            intent=IntentType.MEMORY_QUERY,
            confidence=0.85,
            need_memory=True,
            rewrite_query=text,
            reason="命中长期记忆查询关键词。",
        )

    if _contains_any(text, MEMORY_UPDATE_KEYWORDS):
        return IntentDecision(
            intent=IntentType.MEMORY_UPDATE,
            confidence=0.85,
            need_memory=True,
            rewrite_query=text,
            reason="命中长期记忆写入关键词。",
        )

    if _contains_any(text, SUMMARY_KEYWORDS):
        return IntentDecision(
            intent=IntentType.SUMMARY,
            confidence=0.8,
            need_memory=True,
            rewrite_query=text,
            reason="命中总结类关键词。",
        )

    if _contains_any(text, TOOL_KEYWORDS):
        return IntentDecision(
            intent=IntentType.TOOL_TASK,
            confidence=0.85,
            need_memory=True,
            rewrite_query=text,
            reason="命中任务工具关键词。",
        )

    return None
```

- [ ] **Step 3: Run tests**

Run:

```powershell
$env:PYTHONPATH='C:\Users\zy\Desktop\lrag\backend'; .\.venv\Scripts\python.exe -m unittest backend.tests.test_intent_classifier
```

Expected: pass.

---

## Task 3: Add LLM Fallback Classifier

**Files:**
- Modify: `backend/app/agents/intent.py`
- Test: `backend/tests/test_intent_classifier.py`

- [ ] **Step 1: Add tests using a fake LLM**

Test that invalid JSON falls back to `direct_chat`, and valid JSON becomes `IntentDecision`.

- [ ] **Step 2: Implement `classify_intent`**

Add a function with this behavior:

```text
1. Try classify_intent_by_rules(message).
2. If rule confidence >= 0.85, return it.
3. Otherwise call LLM with a strict JSON-only prompt.
4. Parse JSON into IntentDecision.
5. On parser/model error, return direct_chat with low confidence and reason.
```

- [ ] **Step 3: Use a strict prompt**

The fallback prompt must list allowed intents and force this JSON shape:

```json
{
  "intent": "direct_chat",
  "confidence": 0.0,
  "need_rag": false,
  "need_web": false,
  "need_memory": false,
  "tools": [],
  "rewrite_query": null,
  "reason": "..."
}
```

- [ ] **Step 4: Run classifier tests**

Expected: pass without real network calls.

---

## Task 4: Extend Agent Result and Prompt Context

**Files:**
- Modify: `backend/app/agents/types.py`
- Modify: `backend/app/agents/prompts.py`
- Test: `backend/tests/test_intent_classifier.py`

- [ ] **Step 1: Add result type test**

Assert `AgentRunResult(intent=decision)` keeps the decision.

- [ ] **Step 2: Add prompt helper test**

Assert an intent-aware prompt contains:

```text
当前识别意图
是否需要知识库
是否需要联网搜索
推荐工具
```

- [ ] **Step 3: Modify `AgentRunResult`**

Add:

```python
from ..schemas.intent import IntentDecision

intent: IntentDecision | None = None
```

- [ ] **Step 4: Modify prompt builder**

Add a helper like:

```python
def build_intent_context_prompt(intent: IntentDecision | None) -> str:
    if intent is None:
        return ""
    return (
        "\n\n当前识别意图：\n"
        f"- intent: {intent.intent}\n"
        f"- confidence: {intent.confidence:.2f}\n"
        f"- need_rag: {intent.need_rag}\n"
        f"- need_web: {intent.need_web}\n"
        f"- need_memory: {intent.need_memory}\n"
        f"- tools: {', '.join(intent.tools) if intent.tools else '无'}\n"
        f"- rewrite_query: {intent.rewrite_query or '无'}\n"
        f"- reason: {intent.reason or '无'}\n"
    )
```

---

## Task 5: Refactor Runner Into Explicit LangGraph State Machine

**Files:**
- Modify: `backend/app/agents/langgraph_runner.py`
- Test: `backend/tests/test_intent_graph.py`

- [ ] **Step 1: Create graph-level tests**

Add tests for routing decisions without calling real tools:

```text
web_search -> web/tool branch
rag_qa -> agent branch with RAG allowed
direct_chat -> direct answer branch
clarify -> clarification answer
```

- [ ] **Step 2: Define graph state**

Use a typed state containing:

```python
messages
user_id
session_id
user_message
intent
long_term_memory_context
final_answer
tool_results
used_rag
sources
used_web
web_sources
notice
events
```

- [ ] **Step 3: Add `intent_classifier` node**

Node responsibilities:

```text
Input: user_message
Output: intent
Side effect: append one streamable intent event to events
```

- [ ] **Step 4: Add `memory_context` node**

Only call Mem0 when:

```python
intent.need_memory is True
```

For `memory_update`, avoid unnecessary search unless the answer needs prior context.

- [ ] **Step 5: Add route function**

Routing rule:

```text
clarify -> clarify_node
direct_chat -> agent_node
rag_qa -> agent_node
web_search -> agent_node
memory_query -> agent_node
memory_update -> agent_node
tool_task -> agent_node
summary -> agent_node
mixed -> agent_node
```

First version can still use one `agent_node`, but the graph route must be explicit so later branches can split cleanly.

- [ ] **Step 6: Preserve streaming**

Keep `agent.stream(..., stream_mode=["messages", "updates"])` inside the downstream agent node. Convert its chunks to the same event types currently emitted:

```text
intent
agent_step
tool_call
tool_result
delta
```

- [ ] **Step 7: Return `AgentRunResult(intent=...)`**

Final result must still include existing fields:

```text
final_answer
tool_results
used_rag
sources
used_web
web_sources
notice
intent
```

---

## Task 6: Stream Intent Event Through Chat Workflow

**Files:**
- Modify: `backend/app/services/chat_workflow.py`
- Test: `backend/tests/test_chat_workflow_stream.py`

- [ ] **Step 1: Add stream test**

Mock runner events:

```text
("intent", IntentDecision(...))
("delta", "hello")
StopIteration(AgentRunResult(...))
```

Assert SSE events include `intent` before `delta`.

- [ ] **Step 2: Modify event loop**

Add:

```python
elif kind == "intent":
    payload = data.model_dump(mode="json") if hasattr(data, "model_dump") else dict(data)
    yield ("intent", payload)
```

- [ ] **Step 3: Keep final compatibility**

Do not remove existing `meta`, `sources`, `tool_result`, or `done` behavior.

---

## Task 7: Frontend Intent Event Support

**Files:**
- Modify: `frontend/lib/types.ts`
- Modify: `frontend/lib/api.ts`
- Modify: `frontend/app/page.tsx`
- Modify: `frontend/components/ChatArea.tsx`

- [ ] **Step 1: Add TypeScript types**

Add:

```ts
export type IntentType =
  | "direct_chat"
  | "rag_qa"
  | "web_search"
  | "memory_query"
  | "memory_update"
  | "tool_task"
  | "summary"
  | "mixed"
  | "clarify";

export interface IntentDecision {
  intent: IntentType;
  confidence: number;
  need_rag: boolean;
  need_web: boolean;
  need_memory: boolean;
  tools: string[];
  rewrite_query?: string | null;
  reason: string;
}
```

- [ ] **Step 2: Parse SSE `intent` event**

In `frontend/lib/api.ts`, add `onIntent?: (intent: IntentDecision) => void` and dispatch it when `eventName === "intent"`.

- [ ] **Step 3: Add trace event**

In `frontend/app/page.tsx`, convert intent to an agent trace event:

```text
识别意图：联网搜索
置信度：0.90
原因：命中联网搜索关键词
```

- [ ] **Step 4: Render in ChatArea**

Reuse existing Agent trace UI. Do not add a new large panel.

- [ ] **Step 5: Run frontend type check**

Run:

```powershell
npx tsc --noEmit
```

Expected: pass.

---

## Task 8: End-to-End Verification

**Files:**
- No new files unless tests reveal missing coverage.

- [ ] **Step 1: Run backend tests**

```powershell
$env:PYTHONPATH='C:\Users\zy\Desktop\lrag\backend'; .\.venv\Scripts\python.exe -m unittest discover -s backend\tests
```

Expected: all tests pass.

- [ ] **Step 2: Compile backend**

```powershell
$env:PYTHONPATH='C:\Users\zy\Desktop\lrag\backend'; .\.venv\Scripts\python.exe -m compileall backend\app
```

Expected: compile succeeds.

- [ ] **Step 3: Run frontend type check**

```powershell
npx tsc --noEmit
```

Expected: pass.

- [ ] **Step 4: Manual smoke tests**

Run the app and send these prompts:

```text
你是谁
根据知识库介绍一下项目
联网搜索上海海洋大学官网
你还记得我之前让你怎么回答吗
以后所有回答前面加上“我的回答是：”
总结一下当前会话
```

Expected:

```text
direct_chat prompt does not call web/RAG tools
rag_qa prompt calls knowledge base tool
web_search prompt calls Tavily and streams web sources
memory_query prompt searches Mem0
memory_update prompt writes Mem0 after response
summary prompt uses conversation context
frontend shows intent before answer text
```

---

## Rollout Order

1. Add schema and classifier tests.
2. Add rule classifier.
3. Add LLM fallback classifier.
4. Add prompt and result typing.
5. Add LangGraph intent node and routing.
6. Wire stream event through backend.
7. Wire stream event through frontend.
8. Run full verification.

## Risk Controls

- Keep the old `create_agent` behavior inside the downstream agent node for the first version.
- Keep final `meta` and `sources` events for compatibility.
- Do not persist intent to DB in the first version; store it only in runtime trace unless a later requirement needs history replay.
- Do not split every intent into separate specialized nodes immediately; start with explicit routing and one shared agent branch, then split branches after behavior is stable.

## Self-Review

- Spec coverage: covers schema, classifier, graph routing, stream event, frontend rendering, and verification.
- Placeholder scan: no `TBD` or open-ended implementation placeholders remain.
- Type consistency: `IntentDecision`, `IntentType`, `intent`, `need_rag`, `need_web`, `need_memory`, and `tools` are consistently named across backend and frontend.
