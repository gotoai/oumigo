"""The ``Guard`` protocol and the ``GuardContext`` handed to it.

A guard is any object with an ``intercept(point, ctx) -> Verdict`` method — a
:class:`typing.Protocol`, so no base class or registration is needed (the same structural-
typing idiom as :mod:`oumigo.protocol` and the provider ``Provider`` protocol). A plain
callable of the same shape also works; :mod:`oumigo.guard.chain` adapts it.

A guard may optionally expose a ``points`` attribute (an iterable of :class:`InterceptPoint`)
to declare which points it cares about; without it, the guard is consulted everywhere.

Concrete guards (regex denylists, PII scrubbers, model-based classifiers) live *outside*
oumigo and plug in here — this module ships only the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from oumigo.guard.points import InterceptPoint
    from oumigo.guard.verdict import Verdict


@dataclass
class GuardContext:
    """What a guard sees at an intercept point.

    ``content`` is the primary subject and varies by point: the user's string
    (``USER_INPUT``), the ``messages`` list (``ASSEMBLED_PROMPT``), a tool's parsed arguments
    (``TOOL_CALL``), a tool's result string (``TOOL_RESULT``), or the model's answer text
    (``ASSISTANT_TURN`` / ``FINAL_ANSWER``). The remaining fields are situational context, not
    always populated:

    * ``tool_name`` — the tool being called (tool points only).
    * ``messages``  — the conversation so far, when available.
    * ``system``    — the chat's system prompt, if any.
    * ``metadata``  — an open dict for guard-specific extras.
    """

    content: Any
    tool_name: str | None = None
    messages: list[dict[str, Any]] | None = None
    system: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Guard(Protocol):
    """Anything that can screen content at an intercept point.

    Implementations return :meth:`Verdict.allow` for points/content they take no view on.
    Raising is treated as a ``BLOCK`` by the chain (fail-closed), so a crashing guard denies
    rather than silently passing content through.
    """

    def intercept(self, point: InterceptPoint, ctx: GuardContext) -> Verdict:
        """Screen ``ctx.content`` at ``point`` and return a decision."""
        ...
