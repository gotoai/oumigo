"""The intercept points a guard chain runs at — the vocabulary a ``Guard`` selects on.

A guardrail chain observes a request at five conceptual points, each a distinct method of
:class:`~oumigo.api.agent.chat.OumigoChat`. The tool point has a *pre* and a *post* moment
(``TOOL_CALL`` / ``TOOL_RESULT``), so five points, six members:

1. ``USER_INPUT``       — the caller's message, before the turn starts (``request``).
2. ``ASSEMBLED_PROMPT`` — the full ``messages`` body about to go to the model, every
   round-trip (``_complete_turn`` → ``_payload``).
3. ``TOOL_CALL`` / ``TOOL_RESULT`` — a tool's parsed arguments before execution, and its
   result string after (``_execute_tool``). The highest-risk point: tools have side effects.
4. ``ASSISTANT_TURN``   — a model turn's prose output (``_complete_turn`` / ``_stream_turn``).
5. ``FINAL_ANSWER``     — the answer handed back to the caller (``_run``).

A ``Guard`` may narrow itself to a subset by exposing a ``points`` attribute; absent that, it
is consulted at every point (and returns ``ALLOW`` where it has nothing to say).
"""

from __future__ import annotations

from enum import Enum


class InterceptPoint(str, Enum):
    """Where in a request a guard is consulted. ``str`` value = the audit/log label."""

    USER_INPUT = "user_input"
    ASSEMBLED_PROMPT = "assembled_prompt"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ASSISTANT_TURN = "assistant_turn"
    FINAL_ANSWER = "final_answer"


# The two points that screen model *output*. A chat with any guard active here must buffer a
# streamed turn (collect it, evaluate, then release) rather than pass deltas through live.
OUTPUT_POINTS = frozenset({InterceptPoint.ASSISTANT_TURN, InterceptPoint.FINAL_ANSWER})
