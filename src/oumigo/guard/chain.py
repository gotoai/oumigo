"""``GuardChain`` — an ordered set of guards evaluated at one intercept point.

The chain encodes the decision semantics from the design memo:

* ``TRANSFORM`` **accumulates** — the rewritten content is fed to the next guard.
* ``FLAG`` is collected and the chain continues.
* ``SKIP`` / ``BLOCK`` / ``STOP`` **short-circuit** — evaluation stops and that verdict is
  returned.
* A guard that **raises fails closed**: the exception becomes a ``BLOCK`` (a crashing
  moderator denies rather than leaks).

The result is a :class:`ChainOutcome` bundling the decisive verdict, the content after any
accumulated transforms, and the flags raised along the way.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from oumigo.guard.guard import Guard, GuardContext
from oumigo.guard.points import InterceptPoint
from oumigo.guard.verdict import Verdict

log = logging.getLogger("oumigo.guard")

# A guard is an object with .intercept, or a bare callable of the same shape.
GuardLike = Guard | Callable[[InterceptPoint, GuardContext], Verdict]


@dataclass
class ChainOutcome:
    """The result of evaluating a chain at one point.

    ``verdict`` is the decisive decision (``ALLOW`` if no guard halted). ``content`` is the
    subject after any accumulated ``TRANSFORM``s. ``flags`` are the ``FLAG`` verdicts raised,
    in order, for audit.
    """

    verdict: Verdict
    content: Any
    flags: list[Verdict]


class _CallableGuard:
    """Adapts a bare ``(point, ctx) -> Verdict`` callable to the ``Guard`` protocol."""

    def __init__(self, fn: Callable[[InterceptPoint, GuardContext], Verdict]) -> None:
        self._fn = fn
        # Mirror an optional `points` attribute the callable may carry.
        self.points = getattr(fn, "points", None)

    def intercept(self, point: InterceptPoint, ctx: GuardContext) -> Verdict:
        return self._fn(point, ctx)


def _coerce(guard: GuardLike) -> Guard:
    """Return a ``Guard`` for ``guard``, wrapping a bare callable if needed."""
    if hasattr(guard, "intercept"):
        return guard  # type: ignore[return-value]
    if callable(guard):
        return _CallableGuard(guard)
    raise TypeError(
        f"a guard must have an .intercept(point, ctx) method or be callable, got {guard!r}"
    )


def _applies(guard: Guard, point: InterceptPoint) -> bool:
    """True if ``guard`` is consulted at ``point`` — all points unless it declares ``points``."""
    declared = getattr(guard, "points", None)
    return declared is None or point in declared


class GuardChain:
    """An ordered, immutable collection of guards, evaluated per intercept point."""

    def __init__(self, guards: Iterable[GuardLike] = ()) -> None:
        self._guards: list[Guard] = [_coerce(g) for g in guards]

    def __bool__(self) -> bool:
        return bool(self._guards)

    def __len__(self) -> int:
        return len(self._guards)

    def active_points(self) -> frozenset[InterceptPoint]:
        """The union of points any guard is consulted at (all points if any is unscoped)."""
        points: set[InterceptPoint] = set()
        for g in self._guards:
            declared = getattr(g, "points", None)
            if declared is None:
                return frozenset(InterceptPoint)  # an unscoped guard covers everything
            points.update(declared)
        return frozenset(points)

    def evaluate(self, point: InterceptPoint, ctx: GuardContext) -> ChainOutcome:
        """Run the guards consulted at ``point``, applying accumulate/short-circuit semantics."""
        content = ctx.content
        flags: list[Verdict] = []
        for guard in self._guards:
            if not _applies(guard, point):
                continue
            verdict = self._consult(guard, point, replace(ctx, content=content))
            if verdict.is_allow:
                continue
            if verdict.is_flag:
                flags.append(verdict)
                continue
            if verdict.is_transform:
                content = verdict.content
                continue
            return ChainOutcome(verdict=verdict, content=content, flags=flags)  # halt
        return ChainOutcome(verdict=Verdict.allow(), content=content, flags=flags)

    @staticmethod
    def _consult(guard: Guard, point: InterceptPoint, ctx: GuardContext) -> Verdict:
        """Call one guard, converting a raised exception into a fail-closed ``BLOCK``."""
        try:
            return guard.intercept(point, ctx)
        except Exception as exc:  # noqa: BLE001 - a crashing guard must deny, not leak
            log.warning("guard %r raised at %s: %s", guard, point.value, exc)
            return Verdict.block(f"guard error: {type(exc).__name__}: {exc}")
