"""Parser for the constrained ReAct text protocol."""
from __future__ import annotations

import json
import re
from typing import Any

from .types import AgentDecision

_ACTION_RE = re.compile(r"^Action:\s*([a-zA-Z0-9_:-]+)\s*$", re.MULTILINE)
_ACTION_INPUT_RE = re.compile(
    r"^Action Input:\s*(\{.*?\})\s*$", re.MULTILINE | re.DOTALL
)
_FINAL_RE = re.compile(r"^Final Answer:\s*(.*)\s*$", re.MULTILINE | re.DOTALL)


def _loads_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_react_decision(text: str) -> AgentDecision:
    """Parse one model turn from a deliberately small ReAct grammar."""
    final_match = _FINAL_RE.search(text)
    if final_match:
        final = final_match.group(1).strip()
        if final:
            return AgentDecision(tool_name=None, final_answer=final)

    action_match = _ACTION_RE.search(text)
    if not action_match:
        return AgentDecision(tool_name=None, final_answer="NO_TOOL")

    input_match = _ACTION_INPUT_RE.search(text)
    arguments = _loads_object(input_match.group(1)) if input_match else {}
    return AgentDecision(tool_name=action_match.group(1).strip(), arguments=arguments)
