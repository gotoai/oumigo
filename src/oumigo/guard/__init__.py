"""oumigo guardrail waist — the neutral leaf both ``api`` and ``service`` import.

Defines the guardrail *contract* (types only, no policy): the :class:`Guard` protocol and
its :class:`GuardContext`, the :class:`Verdict` decisions, the :class:`InterceptPoint` a guard
selects on, and the :class:`GuardChain` / :class:`GuardProfile` that bundle guards. Concrete
guards live outside oumigo and plug in via :class:`Guard`.

Placed here — a sibling of :mod:`oumigo.protocol` — so the client agent layer and the
server/router can share one set of guardrail types without an ``api`` ↔ ``service`` import
cycle. V1 wires this into the agent tier (``OumigoChat``); the router may reuse the same types
later.
"""

from __future__ import annotations

from oumigo.guard.chain import ChainOutcome, GuardChain, GuardLike
from oumigo.guard.guard import Guard, GuardContext
from oumigo.guard.points import OUTPUT_POINTS, InterceptPoint
from oumigo.guard.profile import GuardProfile
from oumigo.guard.verdict import Decision, Verdict

__all__ = [
    "OUTPUT_POINTS",
    "ChainOutcome",
    "Decision",
    "Guard",
    "GuardChain",
    "GuardContext",
    "GuardLike",
    "GuardProfile",
    "InterceptPoint",
    "Verdict",
]
