"""The Agent tier — a capability bundle that spawns conversations.

An :class:`OumigoAgent` groups the tools and sampling defaults shared by every chat it
creates, bound to one manager's data plane (``data_url`` + token). It is the entry point
of the inference surface: build one with ``manager.create_agent(...)``, then call
:meth:`OumigoAgent.create_chat` to start a stateful :class:`~oumigo.api.agent.chat.OumigoChat`.

Later versions attach the guardrail profile here too (see the module docstring of
``oumigo.api.agent.chat`` for where the request path is intercepted).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from oumigo.api.agent.chat import OumigoChat
from oumigo.api.agent.tool import Tool

# Default cap on model round-trips within one request() (runaway tool-loop guard).
DEFAULT_MAX_ITERATIONS = 5


class OumigoAgent:
    """A capability bundle (tools + sampling defaults) bound to one manager's data plane.

    Build via :meth:`oumigo.api.OumigoManager.create_agent`. Each :meth:`create_chat`
    spawns a fresh conversation that shares this agent's tools and settings.
    """

    def __init__(
        self,
        *,
        data_url: str,
        token: str | None = None,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        sampling: dict[str, Any] | None = None,
    ) -> None:
        self.data_url = data_url.rstrip("/")
        self.token = token
        self.max_iterations = max(1, int(max_iterations))
        self.sampling = dict(sampling or {})
        self.tools: dict[str, Tool] = _index_tools(tools or [])

    def create_chat(
        self,
        system: str | None = None,
        max_history_turns: int = 3,
    ) -> OumigoChat:
        """Start a conversation.

        Args:
            system: System-role content, prepended to every request in this chat.
            max_history_turns: How many prior (user, assistant) exchanges to carry into
                each request. ``0`` disables memory. Default 3.

        Returns:
            A stateful :class:`~oumigo.api.agent.chat.OumigoChat`. Not thread-safe: one
            session, one chat.
        """
        return OumigoChat(self, system=system, max_history_turns=max_history_turns)


def _index_tools(tools: Sequence[Tool | Callable[..., Any]]) -> dict[str, Tool]:
    """Coerce callables to Tools (strict validation) and index by name, rejecting dupes."""
    indexed: dict[str, Tool] = {}
    for t in tools:
        tool_obj = t if isinstance(t, Tool) else Tool.from_function(t)
        if tool_obj.name in indexed:
            raise ValueError(f"duplicate tool name {tool_obj.name!r} in this agent")
        indexed[tool_obj.name] = tool_obj
    return indexed
