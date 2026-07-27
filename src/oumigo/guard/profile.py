"""``GuardProfile`` — a named bundle of guards attached to an agent.

A profile is what ``create_agent(profile=...)`` takes: a :class:`GuardChain` plus a name for
logs/audit. The default profile is empty — :attr:`is_empty` is True — and an empty profile is
a strict no-op, so wiring a profile through changes nothing until guards are added.

Per-point routing (the "wildcard-first enable map" from the memo) is expressed by each guard's
optional ``points`` attribute; :meth:`active_points` reports the union, which the chat uses to
decide whether a streamed turn must be buffered for output screening.
"""

from __future__ import annotations

from collections.abc import Iterable

from oumigo.guard.chain import ChainOutcome, GuardChain, GuardLike
from oumigo.guard.guard import GuardContext
from oumigo.guard.points import InterceptPoint


class GuardProfile:
    """A named set of guards. Pass to :meth:`OumigoManager.create_agent`."""

    def __init__(self, guards: Iterable[GuardLike] = (), *, name: str = "default") -> None:
        self.name = name
        self._chain = GuardChain(guards)

    @property
    def is_empty(self) -> bool:
        """True when no guards are registered — evaluation is a strict no-op."""
        return not self._chain

    def active_points(self) -> frozenset[InterceptPoint]:
        """The intercept points at least one guard is consulted at."""
        return self._chain.active_points()

    def evaluate(self, point: InterceptPoint, ctx: GuardContext) -> ChainOutcome:
        """Evaluate the chain at ``point`` (see :meth:`GuardChain.evaluate`)."""
        return self._chain.evaluate(point, ctx)

    def __repr__(self) -> str:
        return f"GuardProfile(name={self.name!r}, guards={len(self._chain)})"
