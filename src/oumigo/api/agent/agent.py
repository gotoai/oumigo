"""The Agent tier — a capability bundle that spawns conversations.

An :class:`OumiGoAgent` groups the tools, sampling defaults, and optional guardrail profile
shared by every chat it creates, bound to one manager's data plane (``data_url`` + token). It
is the entry point of the inference surface: build one with ``manager.create_agent(...)``,
then call :meth:`OumiGoAgent.create_chat` to start a stateful
:class:`~oumigo.api.agent.chat.OumiGoChat`.

The ``profile`` (a :class:`oumigo.guard.GuardProfile`) is the guardrail bundle every chat
inherits; the request path is intercepted inside ``oumigo.api.agent.chat`` (see its module
docstring). A ``None``/empty profile is a strict no-op.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from oumigo.api.agent.chat import OumiGoChat
from oumigo.api.agent.tool import Tool
from oumigo.guard import GuardProfile

log = logging.getLogger("oumigo.api.agent.agent")

# Default cap on model round-trips within one request() (runaway tool-loop guard).
DEFAULT_MAX_ITERATIONS = 5

# Distinguishes "caller said nothing" (ask the fleet) from an explicit `None`
# (caller wants no timeout, do not ask).
UNSET: Any = object()

# How long to wait on the fleet for its declared defaults. Deliberately short: this
# is a convenience lookup, and a slow/absent manager must not stall agent setup.
_DEFAULTS_FETCH_TIMEOUT_S = 5.0


class OumiGoAgent:
    """A capability bundle (tools + sampling defaults) bound to one manager's data plane.

    Build via :meth:`oumigo.api.OumiGoManager.create_agent`. Each :meth:`create_chat`
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
        profile: GuardProfile | None = None,
        turn_timeout: float | None = UNSET,
        stall_timeout: float | None = UNSET,
    ) -> None:
        self.data_url = data_url.rstrip("/")
        self.token = token
        self.max_iterations = max(1, int(max_iterations))
        self.sampling = dict(sampling or {})
        self.tools: dict[str, Tool] = _index_tools(tools or [])
        self.profile = profile
        # Timeouts resolve lazily (first chat), so constructing an agent stays offline
        # and a manager that is not up yet costs nothing here.
        self._turn_timeout = turn_timeout
        self._stall_timeout = stall_timeout
        self._defaults_fetched = turn_timeout is not UNSET and stall_timeout is not UNSET

    @property
    def turn_timeout(self) -> float | None:
        """Wall-clock budget for one `request()`, across every tool-loop round-trip."""
        self._resolve_timeouts()
        return self._turn_timeout

    @property
    def stall_timeout(self) -> float | None:
        """How long one round-trip may produce *no bytes* before it is abandoned."""
        self._resolve_timeouts()
        return self._stall_timeout

    def _resolve_timeouts(self) -> None:
        """Fill in whatever the caller left unset from the fleet's `agent:` block.

        Asked once, best-effort: a manager that is down, old, or silent leaves the
        unset values at `None` (wait indefinitely — the behavior before timeouts
        existed), because failing to reach an optional config endpoint must never
        break inference.
        """
        if self._defaults_fetched:
            return
        self._defaults_fetched = True
        declared: dict[str, Any] = {}
        try:
            resp = httpx.get(
                f"{self.data_url}/agent-defaults",
                headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
                timeout=_DEFAULTS_FETCH_TIMEOUT_S,
            )
            if resp.status_code == 200:
                body = resp.json()
                if isinstance(body, dict):
                    declared = body
        except (httpx.HTTPError, ValueError) as exc:
            log.debug("no fleet agent defaults from %s: %s", self.data_url, exc)

        if self._turn_timeout is UNSET:
            self._turn_timeout = _as_seconds(declared.get("turn_timeout"))
        if self._stall_timeout is UNSET:
            self._stall_timeout = _as_seconds(declared.get("stall_timeout"))
        log.debug(
            "agent timeouts: turn=%s stall=%s", self._turn_timeout, self._stall_timeout
        )

    def create_chat(
        self,
        system: str | None = None,
        max_history_turns: int = 3,
        history: list[dict[str, Any]] | None = None,
    ) -> OumiGoChat:
        """Start a conversation.

        Args:
            system: System-role content, prepended to every request in this chat.
            max_history_turns: How many prior (user, assistant) exchanges to carry into
                each request. ``0`` disables memory. Default 3.
            history: Prior conversation to seed (for a stateless server that rehydrates a
                chat per request from a trusted store). Only ``user``/``assistant`` turns
                are accepted — a ``system``/``tool`` role is rejected — so a store/client
                blob can't inject a fake system prompt or tool result. Read the updated
                conversation back via :attr:`OumiGoChat.history` to persist it.

        Returns:
            A stateful :class:`~oumigo.api.agent.chat.OumiGoChat`, inheriting this agent's
            guardrail ``profile``. Not thread-safe: one session, one chat.
        """
        return OumiGoChat(
            self, system=system, max_history_turns=max_history_turns, history=history
        )


def _index_tools(tools: Sequence[Tool | Callable[..., Any]]) -> dict[str, Tool]:
    """Coerce callables to Tools (strict validation) and index by name, rejecting dupes."""
    indexed: dict[str, Tool] = {}
    for t in tools:
        tool_obj = t if isinstance(t, Tool) else Tool.from_function(t)
        if tool_obj.name in indexed:
            raise ValueError(f"duplicate tool name {tool_obj.name!r} in this agent")
        indexed[tool_obj.name] = tool_obj
    return indexed


def _as_seconds(value: object) -> float | None:
    """A positive number of seconds, or None for anything else (never raises)."""
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None
