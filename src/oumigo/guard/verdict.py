"""A guard's decision at one intercept point — the narrow-waist return type.

``intercept`` returns a :class:`Verdict`, one of six decisions. Chain semantics
(:mod:`oumigo.guard.chain`) hinge on the decision:

* ``ALLOW``     — nothing to change; the chain moves on.
* ``TRANSFORM`` — replace the content with ``content``; the chain **continues** with the
  rewritten value, so a later guard sees it (transforms *accumulate*).
* ``FLAG``      — allow the content through unchanged but record an audit event; continues.
* ``SKIP``      — drop this unit of work and keep going where that makes sense (a skipped
  *tool call* feeds ``reason`` back to the model as the result; the loop continues).
* ``BLOCK``     — end this turn; ``reason`` becomes the surfaced text.
* ``STOP``      — abort the whole request.

``SKIP`` / ``BLOCK`` / ``STOP`` **short-circuit** the chain (the split of "drop one call" vs
"end the turn" vs "abort the run" follows Cline's `skip`-vs-`stop` distinction). Build a
verdict with the classmethods (``Verdict.block("...")``) rather than the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Decision(str, Enum):
    """The six guard decisions. ``str`` value = the audit/log label."""

    ALLOW = "allow"
    TRANSFORM = "transform"
    FLAG = "flag"
    SKIP = "skip"
    BLOCK = "block"
    STOP = "stop"


# Decisions that stop the chain from consulting further guards at this point.
_HALTING = frozenset({Decision.SKIP, Decision.BLOCK, Decision.STOP})


@dataclass(frozen=True)
class Verdict:
    """One guard's decision. Immutable; created via the classmethods below.

    ``content`` is meaningful only for ``TRANSFORM`` (the replacement). ``reason`` is the
    human-readable explanation carried by ``FLAG`` / ``SKIP`` / ``BLOCK`` / ``STOP`` — logged
    for a flag, and surfaced as the response text for a block/stop.
    """

    decision: Decision
    content: Any = None
    reason: str | None = None

    # -- constructors ------------------------------------------------------- #

    @classmethod
    def allow(cls) -> Verdict:
        """Let the content through unchanged."""
        return _ALLOW

    @classmethod
    def transform(cls, content: Any) -> Verdict:
        """Replace the content with ``content`` and continue the chain."""
        return cls(Decision.TRANSFORM, content=content)

    @classmethod
    def flag(cls, reason: str = "") -> Verdict:
        """Allow the content but record an audit event with ``reason``."""
        return cls(Decision.FLAG, reason=reason)

    @classmethod
    def skip(cls, reason: str = "") -> Verdict:
        """Drop this unit of work (e.g. a single tool call) and continue."""
        return cls(Decision.SKIP, reason=reason)

    @classmethod
    def block(cls, reason: str = "") -> Verdict:
        """End this turn; ``reason`` becomes the surfaced text."""
        return cls(Decision.BLOCK, reason=reason)

    @classmethod
    def stop(cls, reason: str = "") -> Verdict:
        """Abort the whole request; ``reason`` becomes the surfaced text."""
        return cls(Decision.STOP, reason=reason)

    # -- predicates --------------------------------------------------------- #

    @property
    def is_allow(self) -> bool:
        return self.decision is Decision.ALLOW

    @property
    def is_transform(self) -> bool:
        return self.decision is Decision.TRANSFORM

    @property
    def is_flag(self) -> bool:
        return self.decision is Decision.FLAG

    @property
    def skips(self) -> bool:
        return self.decision is Decision.SKIP

    @property
    def blocks(self) -> bool:
        return self.decision is Decision.BLOCK

    @property
    def stops(self) -> bool:
        return self.decision is Decision.STOP

    @property
    def halts(self) -> bool:
        """True for ``SKIP`` / ``BLOCK`` / ``STOP`` — the chain short-circuits here."""
        return self.decision in _HALTING


_ALLOW = Verdict(Decision.ALLOW)  # the sole ALLOW instance (Verdict.allow() returns it)
