"""The worker *handle* — a client-side view of a worker child this process spawned.

This is the API-layer handle returned by :func:`oumigo.api.api.oumigo_create_worker`, not
the worker *service* (that lives under ``oumigo.service.worker``). It supervises one
vLLM/HF replica child and reports its state by polling the manager's ``/workers``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import httpx

from oumigo.common.proc import terminate
from oumigo.protocol.states import NodeState

# Grace given to a worker child on teardown before escalating to SIGKILL. On SIGTERM the
# coordinator drains and runs the backend's process-group shutdown (which reaps EngineCore
# and frees VRAM); this must outlast that clean path so we don't SIGKILL the coordinator
# mid-shutdown and re-leak the very orphan the group teardown exists to prevent. Sized for
# the common idle stop (quick drain + the backend's ~30s SIGTERM->SIGKILL grace).
_WORKER_STOP_GRACE_S = 35.0


@dataclass
class OumigoWorker:
    """A worker child this process spawned, supervising one vLLM/HF replica."""

    manager_url: str
    address: str
    port: int
    model: str
    backend: str = "vllm"
    node_id: str | None = None
    _child: subprocess.Popen | None = field(default=None, repr=False)

    def _record(self, timeout_s: float = 5.0) -> dict | None:
        """This worker's registry record from the manager, matched by address:port."""
        try:
            resp = httpx.get(f"{self.manager_url}/workers", timeout=timeout_s)
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        for rec in resp.json().get("workers", []):
            if rec.get("address") == self.address and rec.get("port") == self.port:
                return rec
        return None

    def state(self) -> str | None:
        """The node state the manager last saw (e.g. ``SERVING``), or None if unknown."""
        rec = self._record()
        return rec.get("state") if rec else None

    def is_serving(self) -> bool:
        # Registry serializes state as the lowercase NodeState value ("serving"); normalize
        # so a casing mismatch can't silently make this always-False.
        return str(self.state() or "").lower() == NodeState.SERVING.value

    def is_alive(self) -> bool:
        """True while the worker child process is still running."""
        return self._child is not None and self._child.poll() is None

    def stop(self) -> None:
        """Stop the worker child; the coordinator drains and shuts down the replica."""
        terminate(self._child, grace_s=_WORKER_STOP_GRACE_S)
        self._child = None

    def __enter__(self) -> OumigoWorker:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
