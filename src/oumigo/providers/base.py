"""The Provider protocol: the minimal lifecycle primitives the manager needs.

Keep this small — every method must be something the manager actually calls.
Do not model all of the cloud/OpenStack API here.
"""

from __future__ import annotations

from typing import Any, Protocol


class Provider(Protocol):
    """Minimal provisioning surface for a fleet of worker nodes."""

    def provision(self, spec: Any) -> Any:
        """Create a node from a NodeSpec; return an opaque instance handle."""
        ...

    def get_status(self, handle: Any) -> Any:
        """Provisioning/liveness status of the instance."""
        ...

    def get_address(self, handle: Any) -> str:
        """Reachable address (LAN or public) the manager/agent should use."""
        ...

    def terminate(self, handle: Any) -> None:
        """Tear the instance down. Must be idempotent."""
        ...

    def list(self, cluster_id: str) -> list[Any]:
        """All instances tagged with cluster_id — for reconciliation / reaping."""
        ...
