"""StaticProvider — the LAN provider (v1).

In LAN mode the manager does NOT provision instances. Workers are hand-started
(or externally managed) and self-register with the manager, so the fleet is
simply whatever registers. This is the real first implementation of the Provider
protocol; cloud backends (e.g. ConoHa) are future implementations of the same shape.
"""

from __future__ import annotations

from typing import Any


class StaticProvider:
    """LAN provider: no provisioning; workers self-register with the manager."""

    name = "LAN"

    def provision(self, spec: Any) -> Any:
        raise NotImplementedError(
            "LAN provider does not provision workers; start them manually "
            "(they self-register with the manager)."
        )

    def get_status(self, handle: Any) -> Any:
        raise NotImplementedError("LAN provider does not track instance status.")

    def get_address(self, handle: Any) -> str:
        raise NotImplementedError("LAN provider does not resolve instance addresses.")

    def terminate(self, handle: Any) -> None:
        # Nothing to tear down; LAN workers are externally managed.
        return None

    def list(self, cluster_id: str) -> list[Any]:
        # No provider-side inventory in LAN mode; the manager's registry is the
        # source of truth for which workers exist.
        return []
