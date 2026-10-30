"""StaticProvider — the non-provisioning providers: LAN and CUSTOM.

In both modes the manager does NOT provision instances. Workers are hand-started
(or created by tooling outside oumigo) and self-register with the manager, so the
fleet is simply whatever registers. The two differ only in the network posture the
manager assumes, not in provisioning:

- ``LAN``    — workers on our own trusted local network (mDNS on, auth may be off).
- ``CUSTOM`` — workers anywhere: Vast.ai, RunPod, ConoHa, other clouds, other sites.
  No vendor-specific processing; the manager assumes an untrusted wide-area network
  (mDNS off, TLS required, auth fail-closed — see `oumigo.service.manager.posture`).

Cloud backends that provision through a vendor API (e.g. ConoHa) are future
implementations of the same Provider protocol.
"""

from __future__ import annotations

from typing import Any

LAN = "LAN"
CUSTOM = "CUSTOM"


class StaticProvider:
    """Non-provisioning provider: workers self-register with the manager."""

    def __init__(self, name: str = LAN) -> None:
        self.name = name

    @property
    def is_wan(self) -> bool:
        """True when workers may sit on an untrusted wide-area network (CUSTOM)."""
        return self.name == CUSTOM

    def provision(self, spec: Any) -> Any:
        raise NotImplementedError(
            f"{self.name} provider does not provision workers; start them manually "
            "(they self-register with the manager)."
        )

    def get_status(self, handle: Any) -> Any:
        raise NotImplementedError(f"{self.name} provider does not track instance status.")

    def get_address(self, handle: Any) -> str:
        raise NotImplementedError(f"{self.name} provider does not resolve instance addresses.")

    def terminate(self, handle: Any) -> None:
        raise NotImplementedError(f"{self.name} provider does not terminate instances.")
