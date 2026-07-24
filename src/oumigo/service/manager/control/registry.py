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
    seq: int = 0  # 1-based order of first registration; drives the friendly name
    port: int | None = None  # worker's actual vLLM port (negotiated); None until reported
    model: str | None = None  # effective model the worker serves (env-negotiable); None until reported
    backend: str | None = None  # inference backend in use ("vllm" | "transformer"); None until reported
    # Worker's negotiated in-flight cap; None until reported (router falls back to the
    # fleet default from node_spec.max_concurrent_requests).
    max_concurrent_requests: int | None = None

    @property
    def name(self) -> str:
        """Human-facing label — the manager owns the seq -> UUID mapping."""
        return f"Worker#{self.seq}"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["name"] = self.name  # a property, so not captured by asdict
        return d


class Registry:
    """Thread-safe map of node_id -> NodeRecord."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeRecord] = {}
        self._lock = threading.Lock()
        self._seq = 0  # monotonic; assigns Worker#N on first registration

    def register(
        self,
        node_id: str,
        address: str,
        state: str,
        incarnation: int,
        capabilities: dict,
        port: int | None = None,
        model: str | None = None,
        backend: str | None = None,
    ) -> NodeRecord:
        """Idempotent: keyed on node_id, so a re-register updates in place.

        `port` is the worker's actual vLLM port (part of its node_id) and `model` its
        effective (env-negotiated) model; both are known at registration, but a
        re-register that omits either keeps the last value.
        """
        now = time.time()
        with self._lock:
            existing = self._nodes.get(node_id)
            if existing is not None:
                seq = existing.seq  # keep the same name across re-registrations
            else:
                self._seq += 1
                seq = self._seq
            record = NodeRecord(
                node_id=node_id,
                address=address,
                state=state,
                incarnation=incarnation,
                capabilities=capabilities,
                registered_at=existing.registered_at if existing else now,
                last_seen=now,
                seq=seq,
                # worker reports its actual port at registration; keep last known if omitted
                port=port if port is not None else (existing.port if existing else None),
                # worker reports its effective model at registration; keep last if omitted
                model=model if model is not None else (existing.model if existing else None),
                # worker reports its backend at registration; keep last if omitted
                backend=backend if backend is not None else (existing.backend if existing else None),
                # keep negotiated in-flight cap across re-register
                max_concurrent_requests=existing.max_concurrent_requests if existing else None,
            )
            self._nodes[node_id] = record
            return record

    def heartbeat(
        self,
        node_id: str,
        state: str,
        run_state: str | None = None,
        port: int | None = None,
        max_concurrent_requests: int | None = None,
    ) -> bool:
        """Update last_seen + state (+ port / in-flight cap when reported). False if unknown."""
        with self._lock:
            record = self._nodes.get(node_id)
            if record is None:
                return False
            record.last_seen = time.time()
            record.state = state
            record.run_state = run_state
            if port is not None:  # worker's negotiated vLLM port; keep last known if omitted
                record.port = port
            if max_concurrent_requests is not None:  # negotiated in-flight cap; keep last if omitted
                record.max_concurrent_requests = max_concurrent_requests
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
