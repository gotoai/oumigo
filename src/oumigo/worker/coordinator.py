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
import os
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
from oumigo.worker.identity import resolve_worker_identity
from oumigo.worker.metrics import MetricsCollector
from oumigo.worker.supervisor import (
    HFProcess,
    PortUnavailable,
    VLLMProcess,
    _ServerProcess,
    find_free_port,
)

log = logging.getLogger("oumigo.worker")

# Worker backends: which inference server the coordinator supervises.
BACKEND_VLLM = "vllm"
BACKEND_TRANSFORMER = "transformer"
BACKENDS = (BACKEND_VLLM, BACKEND_TRANSFORMER)


class WorkerCoordinator:
    """Registers with the manager, supervises the backend, and drives the state machine."""

    def __init__(
        self,
        manager_url: str,
        token: str | None,
        node_id: str,
        register_req: RegisterRequest,
        *,
        spec: NodeSpec,
        vllm_port: int,
        backend: str = BACKEND_VLLM,
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
        self.backend = backend
        # The transformer backend generates one request at a time (no batching), so it
        # tells the router to admit only one in-flight request to this worker. vLLM
        # batches — leave None so the router keeps the fleet-default capacity.
        self.report_max_concurrent: int | None = 1 if backend == BACKEND_TRANSFORMER else None
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
        # Spec + actual port are resolved before construction (the port is folded into
        # node_id), so vLLM launches on this exact port and the id stays stable.
        self.spec: NodeSpec | None = spec
        self.vllm_port: int | None = vllm_port  # actual (preflight-selected) port; part of node_id
        self.supervisor: _ServerProcess | None = None
        self.metrics: MetricsCollector | None = None
        self._restarts = 0

        self._stop = threading.Event()   # graceful stop requested (signal or STOP command)
        self._force = threading.Event()  # second signal: stop draining, kill now

    # --- entrypoint ----------------------------------------------------------

    def run(self) -> None:
        self._install_signals()
        assert self.spec is not None  # resolved pre-construction via /spec
        self._register()  # identity + port already chosen; response only carries cadence
        self._start_vllm(self.spec)
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
        port = self._select_port(self.spec)  # re-preflight: the port may have freed or moved
        if port is None:
            return  # FAILED set by _select_port
        self.supervisor = self._make_process(self.spec, port)
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

    def _make_process(self, spec: NodeSpec, port: int) -> _ServerProcess:
        """Build the supervised backend child for this worker's `--backend`."""
        if self.backend == BACKEND_TRANSFORMER:
            return HFProcess(spec, port=port)
        return VLLMProcess(spec, port=port)

    def _start_vllm(self, spec: NodeSpec) -> None:
        self.spec = spec
        port = self.vllm_port  # preflighted before construction (it is part of node_id)
        if port is None:  # defensive; run_worker always selects a port first
            log.error("no backend port selected -> FAILED")
            self.node_state = NodeState.FAILED
            return
        self.node_state = NodeState.INITIALIZING
        self.supervisor = self._make_process(spec, port)
        self.supervisor.start()
        log.info(
            "loading model %s on port %d via %s (this can take minutes)",
            spec.model, port, self.backend,
        )

    def _select_port(self, spec: NodeSpec) -> int | None:
        """Re-preflight a free port for a *restart*, preferring the one already folded
        into our node_id so identity stays put. None (and FAILED) if none is free.

        Done before launch because vLLM binds only after the model loads — scanning
        here costs a socket bind, not a model load. Updates `self.vllm_port` (the
        heartbeat reports it) if a fallover was unavoidable.
        """
        preferred = self.vllm_port if self.vllm_port is not None else spec.port
        try:
            port = find_free_port(spec.host, preferred)
        except PortUnavailable as exc:
            log.error("cannot restart vLLM: %s -> FAILED", exc)
            self.node_state = NodeState.FAILED
            return None
        if port != preferred:
            log.warning("preferred vLLM port %d is in use; falling over to %d", preferred, port)
        self.vllm_port = port
        return port

    def _start_metrics(self) -> None:
        """Start grid-aligned metrics sampling + reporting (background threads)."""
        if not self.metrics_enabled:
            return
        # vLLM exposes /metrics on 127.0.0.1:<port>; the scraper no-ops until it serves.
        # Use the actual negotiated port (may differ from spec.port on a fallover).
        vllm_url = f"http://127.0.0.1:{self.vllm_port}" if self.vllm_port else None
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
                    vllm_port=self.vllm_port,
                    max_concurrent_requests=self.report_max_concurrent,
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


def _apply_env_overrides(spec: NodeSpec | None) -> NodeSpec:
    """Overlay the negotiable env/.env settings onto the manager's spec (env wins).

    The model the manager hands out is negotiable per worker: `MODEL_NAME` and
    `MAX_MODEL_LEN` override `model` / `max_model_len`, and `HF_HOME` (with `~`/`$VARS`
    expanded) + `HF_TOKEN` (blank -> anonymous) are normalized in `os.environ` so the
    spawned backend child inherits them. When the manager configured no model at all,
    `MODEL_NAME` alone is enough to serve.
    """
    # HF cache/token: normalize in place so the backend child inherits them unchanged.
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = os.path.expanduser(os.path.expandvars(hf_home))
    if not os.environ.get("HF_TOKEN", "").strip():
        os.environ.pop("HF_TOKEN", None)  # empty token -> anonymous, not a blank credential

    model = os.environ.get("MODEL_NAME") or (spec.model if spec else None)
    if not model:
        raise SystemExit(
            "no model to serve: set MODEL_NAME in the environment/.env, or configure "
            "model.name on the manager"
        )
    updates: dict = {"model": model}
    max_len = os.environ.get("MAX_MODEL_LEN")
    if max_len:
        try:
            updates["max_model_len"] = int(max_len)
        except ValueError:
            raise SystemExit(f"MAX_MODEL_LEN must be an integer, got {max_len!r}")
    base = spec or NodeSpec(model=model)  # manager has no model -> defaults + env model
    return base.model_copy(update=updates)


def run_worker(
    manager_url: str | None,
    token: str | None,
    state_dir: Path | None = None,
    discover_timeout: float = discovery.DEFAULT_DISCOVER_TIMEOUT,
    backend: str = BACKEND_VLLM,
) -> None:
    """Find the manager, fetch + env-negotiate the spec, preflight a port, derive
    identity, then hand off to the coordinator loop. The node_id is a hash of address +
    the *actual* backend port, so the port must be chosen before we register."""
    if backend not in BACKENDS:
        raise SystemExit(f"unknown --backend {backend!r}; choose one of {', '.join(BACKENDS)}")

    if not manager_url:
        log.info("no manager URL provided; discovering via mDNS (up to %.0fs) ...", discover_timeout)
        manager_url = discovery.discover_manager(discover_timeout)
        if not manager_url:
            raise SystemExit(
                f"could not discover a manager on the LAN within {discover_timeout:.0f}s; "
                "set --manager-url / $OUMIGO_MANAGER_URL"
            )
        log.info("discovered manager at %s", manager_url)

    # 1) Fetch the fleet spec (for the preferred port + model), then overlay env overrides.
    #    A manager without a model is fine as long as MODEL_NAME is set in the environment.
    try:
        spec = client.fetch_node_spec(manager_url, token)
    except Exception as exc:  # noqa: BLE001 - surface any client/server error as a clean exit
        raise SystemExit(f"could not fetch spec from manager {manager_url}: {exc}")
    spec = _apply_env_overrides(spec)

    # 2) Preflight the actual port, then derive identity from address:port.
    address = discovery.get_lan_ip()
    try:
        port = find_free_port(spec.host, spec.port)
    except PortUnavailable as exc:
        raise SystemExit(f"cannot start backend: {exc}")
    node_id, incarnation, path = resolve_worker_identity(address, port, state_dir)
    log.info(
        "worker identity %s (incarnation %d) at %s:%d, backend=%s, model=%s [%s]",
        node_id, incarnation, address, port, backend, spec.model, path,
    )

    register_req = RegisterRequest(
        node_id=node_id,
        address=address,
        vllm_port=port,
        model=spec.model,  # effective model (after env negotiation) for the manager to display
        incarnation=incarnation,
        state=NodeState.REGISTERING,
        capabilities=NodeCapabilities(),
    )
    WorkerCoordinator(
        manager_url, token, node_id, register_req, spec=spec, vllm_port=port, backend=backend
    ).run()
