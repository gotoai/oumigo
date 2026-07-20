"""Coordinator — the long-lived worker-node process.

Owns the node state machine (see docs/worker-node-states.md) and drives it from a
single loop whose tick is the heartbeat:

    REGISTERING -> INITIALIZING -> SERVING -> DRAINING -> STOPPED
                        ^  (restart)             |
                        +---- crash -------------+  (policy exhausted -> FAILED)

Each tick reconciles the vLLM child's real status into the node state, then
heartbeats that state to the manager and acts on any command it returns (STOP).
Keeping reconcile and heartbeat on one loop means the long INITIALIZING window
(model load — minutes) is *visible* to the manager rather than looking like a
silent, LOST node.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path

import httpx

from oumigo import discovery
from oumigo.config.spec import NodeSpec
from oumigo.protocol.messages import (
    HeartbeatRequest,
    HeartbeatResponse,
    NodeCapabilities,
    RegisterRequest,
    RegisterResponse,
    WorkerCommand,
)
from oumigo.protocol.states import NodeState, RunState
from oumigo.worker import client
from oumigo.worker.identity import resolve_node_identity
from oumigo.worker.metrics import MetricsCollector
from oumigo.worker.supervisor import VLLMProcess

log = logging.getLogger("oumigo.worker")


class WorkerCoordinator:
    """Registers with the manager, supervises vLLM, and drives the state machine."""

    def __init__(
        self,
        manager_url: str,
        token: str | None,
        node_id: str,
        register_req: RegisterRequest,
        *,
        max_restarts: int = 3,
        restart_backoff_s: float = 5.0,
        stop_grace_s: float = 30.0,
        drain_timeout_s: float = 120.0,
        metrics_enabled: bool = True,
        metrics_grid_s: float = 5.0,
        metrics_report_s: float = 30.0,
        metrics_capacity_s: float = 1800.0,
        metrics_evict_chunk_s: float = 300.0,
    ) -> None:
        self.manager_url = manager_url
        self.token = token
        self.node_id = node_id
        self.register_req = register_req
        self.max_restarts = max_restarts
        self.restart_backoff_s = restart_backoff_s
        self.stop_grace_s = stop_grace_s
        self.drain_timeout_s = drain_timeout_s
        self.metrics_enabled = metrics_enabled
        self.metrics_grid_s = metrics_grid_s
        self.metrics_report_s = metrics_report_s
        self.metrics_capacity_s = metrics_capacity_s
        self.metrics_evict_chunk_s = metrics_evict_chunk_s

        self._started_at = time.time()  # worker:start_timestamp (float UTC epoch)
        self.node_state: NodeState = NodeState.REGISTERING
        self.run_state: RunState | None = None
        self.interval: float = 10.0
        self.spec: NodeSpec | None = None
        self.supervisor: VLLMProcess | None = None
        self.metrics: MetricsCollector | None = None
        self._restarts = 0

        self._stop = threading.Event()   # graceful stop requested (signal or STOP command)
        self._force = threading.Event()  # second signal: stop draining, kill now

    # --- entrypoint ----------------------------------------------------------

    def run(self) -> None:
        self._install_signals()
        resp = self._register()
        spec = resp.node_spec
        if spec is None:
            raise SystemExit(
                "manager accepted registration but returned no vLLM config; "
                "set model.name in the manager's manager.yaml — nothing to serve"
            )
        self._start_vllm(spec)
        self._start_metrics()
        self._loop()
        # Loop returns on a graceful stop (drain + shut down) or on FAILED (already dead).
        if self.node_state != NodeState.FAILED:
            self._drain_and_stop()
        self._stop_metrics()
        log.info("coordinator exiting (final state=%s)", self.node_state.value)

    # --- main loop -----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._reconcile()
            if self.node_state == NodeState.FAILED:
                self._send_heartbeat()  # tell the manager promptly, then give up
                return
            resp = self._send_heartbeat()
            if resp is None:
                continue  # transient blip; stay alive and retry next tick
            if not resp.known:
                log.warning("manager does not know us; re-registering")
                self._reregister()
            elif resp.command == WorkerCommand.STOP:
                log.info("received STOP from manager")
                self._stop.set()
                return

    def _reconcile(self) -> None:
        """Fold the vLLM child's real status into node_state / run_state."""
        sup = self.supervisor
        if sup is None:
            return
        if sup.running:
            if self.node_state == NodeState.INITIALIZING and sup.is_healthy():
                log.info("vLLM healthy -> SERVING")
                self.node_state = NodeState.SERVING
                if self.metrics is not None:
                    self.metrics.mark_serving()  # stamp vllm:start_timestamp on this edge
            if self.node_state == NodeState.SERVING:
                self.run_state = sup.run_state()
            return
        # Not running and we didn't ask it to stop -> unexpected exit.
        log.warning("vLLM exited unexpectedly (code=%s)", sup.poll())
        self._handle_crash()

    def _handle_crash(self) -> None:
        self.run_state = None
        if self.metrics is not None:
            self.metrics.clear_serving()  # left SERVING; re-stamped when it serves again
        if self._restarts >= self.max_restarts:
            log.error("vLLM restart policy exhausted (%d attempts) -> FAILED", self._restarts)
            self.node_state = NodeState.FAILED
            return
        self._restarts += 1
        backoff = min(self.restart_backoff_s * self._restarts, 30.0)
        log.info(
            "restarting vLLM (attempt %d/%d) after %.0fs backoff",
            self._restarts,
            self.max_restarts,
            backoff,
        )
        self.node_state = NodeState.INITIALIZING
        if self._stop.wait(backoff):  # interruptible: a stop during backoff aborts the restart
            return
        assert self.spec is not None
        self.supervisor = VLLMProcess(self.spec)
        self.supervisor.start()

    # --- steps ---------------------------------------------------------------

    def _register(self) -> RegisterResponse:
        try:
            resp = client.register(self.manager_url, self.register_req, self.token)
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 401:
                raise SystemExit("registration rejected (401): check the shared bearer token")
            raise SystemExit(f"registration failed: HTTP {code}")
        except httpx.RequestError as exc:
            raise SystemExit(f"could not reach manager at {self.manager_url}: {exc}")
        self.interval = resp.heartbeat_interval_s or 10
        log.info(
            "registered with manager %s (accepted=%s, heartbeat=%.0fs)",
            self.manager_url,
            resp.accepted,
            self.interval,
        )
        return resp

    def _reregister(self) -> None:
        try:
            client.register(self.manager_url, self.register_req, self.token)
        except Exception as exc:  # noqa: BLE001 - best-effort; next tick retries
            log.warning("re-registration failed: %s", exc)

    def _start_vllm(self, spec: NodeSpec) -> None:
        self.spec = spec
        self.node_state = NodeState.INITIALIZING
        self.supervisor = VLLMProcess(spec)
        self.supervisor.start()
        log.info("loading model %s on port %d (this can take minutes)", spec.model, spec.port)

    def _start_metrics(self) -> None:
        """Start grid-aligned metrics sampling + reporting (background threads)."""
        if not self.metrics_enabled:
            return
        # vLLM exposes /metrics on 127.0.0.1:<port>; the scraper no-ops until it serves.
        vllm_url = f"http://127.0.0.1:{self.spec.port}" if self.spec else None
        self.metrics = MetricsCollector(
            self.manager_url,
            self.token,
            self.node_id,
            grid_s=self.metrics_grid_s,
            report_s=self.metrics_report_s,
            capacity_s=self.metrics_capacity_s,
            evict_chunk_s=self.metrics_evict_chunk_s,
            vllm_url=vllm_url,
            worker_start=self._started_at,
        )
        self.metrics.start()
        log.info(
            "metrics collector started (grid=%.0fs, report=%.0fs, buffer=%.0fs)",
            self.metrics_grid_s,
            self.metrics_report_s,
            self.metrics_capacity_s,
        )

    def _stop_metrics(self) -> None:
        if self.metrics is not None:
            self.metrics.stop()
            self.metrics = None

    def _drain_and_stop(self) -> None:
        """DRAINING: refuse new work, let in-flight finish, then stop vLLM -> STOPPED."""
        sup = self.supervisor
        if sup is None or not sup.running:
            self.node_state = NodeState.STOPPED
            self.run_state = None
            self._send_heartbeat()
            return

        self.node_state = NodeState.DRAINING
        deadline = time.monotonic() + self.drain_timeout_s
        while (
            not self._force.is_set()
            and time.monotonic() < deadline
            and sup.run_state() == RunState.EXECUTING
        ):
            self.run_state = RunState.EXECUTING
            log.info("draining: waiting for in-flight requests to finish")
            self._send_heartbeat()
            time.sleep(min(self.interval, max(0.0, deadline - time.monotonic())))

        sup.stop(self.stop_grace_s)
        self.node_state = NodeState.STOPPED
        self.run_state = None
        self._send_heartbeat()
        log.info("vLLM stopped cleanly -> STOPPED")

    # --- helpers -------------------------------------------------------------

    def _send_heartbeat(self) -> HeartbeatResponse | None:
        try:
            return client.heartbeat(
                self.manager_url,
                HeartbeatRequest(
                    node_id=self.node_id,
                    node_state=self.node_state,
                    run_state=self.run_state,
                ),
                self.token,
            )
        except Exception as exc:  # noqa: BLE001 - keep the worker alive across blips
            log.warning("heartbeat failed: %s", exc)
            return None

    def _install_signals(self) -> None:
        def handler(signum: int, _frame: object) -> None:
            if self._stop.is_set():
                log.warning("second signal (%d); forcing shutdown", signum)
                self._force.set()
            else:
                log.info("signal %d received; draining and shutting down", signum)
                self._stop.set()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


def run_worker(
    manager_url: str | None,
    token: str | None,
    state_dir: Path | None = None,
    discover_timeout: float = discovery.DEFAULT_DISCOVER_TIMEOUT,
) -> None:
    """Resolve identity, find the manager, then hand off to the coordinator loop."""
    node_id, incarnation, path = resolve_node_identity(state_dir)
    log.info("node identity %s (incarnation %d) [%s]", node_id, incarnation, path)

    if not manager_url:
        log.info("no manager URL provided; discovering via mDNS (up to %.0fs) ...", discover_timeout)
        manager_url = discovery.discover_manager(discover_timeout)
        if not manager_url:
            raise SystemExit(
                f"could not discover a manager on the LAN within {discover_timeout:.0f}s; "
                "set --manager-url / $OUMIGO_MANAGER_URL"
            )
        log.info("discovered manager at %s", manager_url)

    register_req = RegisterRequest(
        node_id=node_id,
        address=discovery.get_lan_ip(),
        incarnation=incarnation,
        state=NodeState.REGISTERING,
        capabilities=NodeCapabilities(),
    )
    WorkerCoordinator(manager_url, token, node_id, register_req).run()
