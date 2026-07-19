"""In-memory node registry (v1).

Tracks which workers have registered and when they were last seen. This is the
manager's source of truth for the fleet's *actual* state. Thread-safe because the
HTTP server may touch it from multiple worker threads. State is not persisted —
the manager rebuilds it from re-registrations after a restart.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass

from oumigo.protocol.states import NodeState


@dataclass
class NodeRecord:
    node_id: str
    address: str
    state: str
    incarnation: int
    capabilities: dict
    registered_at: float
    last_seen: float
    run_state: str | None = None  # IDLE/EXECUTING while serving; None otherwise

    def as_dict(self) -> dict:
        return asdict(self)


class Registry:
    """Thread-safe map of node_id -> NodeRecord."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeRecord] = {}
        self._lock = threading.Lock()

    def register(
        self,
        node_id: str,
        address: str,
        state: str,
        incarnation: int,
        capabilities: dict,
    ) -> NodeRecord:
        """Idempotent: keyed on node_id, so a re-register updates in place."""
        now = time.time()
        with self._lock:
            existing = self._nodes.get(node_id)
            record = NodeRecord(
                node_id=node_id,
                address=address,
                state=state,
                incarnation=incarnation,
                capabilities=capabilities,
                registered_at=existing.registered_at if existing else now,
                last_seen=now,
            )
            self._nodes[node_id] = record
            return record

    def heartbeat(self, node_id: str, state: str, run_state: str | None = None) -> bool:
        """Update last_seen + state. Returns False if the node is unknown (should re-register)."""
        with self._lock:
            record = self._nodes.get(node_id)
            if record is None:
                return False
            record.last_seen = time.time()
            record.state = state
            record.run_state = run_state
            return True

    def list(self) -> list[NodeRecord]:
        with self._lock:
            return list(self._nodes.values())

    def count(self) -> int:
        with self._lock:
            return len(self._nodes)

    def reap(self, timeout_s: float, forget_after_s: float) -> tuple[list[str], list[str]]:
        """Mark silent nodes LOST and forget long-absent ones.

        A node silent longer than `timeout_s` is marked LOST (it drops out of the
        healthy set; a returning worker's heartbeat restores it). A node silent
        longer than `forget_after_s` is removed entirely. Returns
        (newly_lost_ids, removed_ids) for logging.
        """
        now = time.time()
        newly_lost: list[str] = []
        removed: list[str] = []
        with self._lock:
            for node_id, record in list(self._nodes.items()):
                silent = now - record.last_seen
                if silent > forget_after_s:
                    del self._nodes[node_id]
                    removed.append(node_id)
                elif silent > timeout_s and record.state != NodeState.LOST.value:
                    record.state = NodeState.LOST.value
                    newly_lost.append(node_id)
        return newly_lost, removed
