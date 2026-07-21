"""Manager control-plane HTTP server (v1).

Accepts worker registrations and heartbeats, tracks them in an in-memory
`Registry`, and (unless disabled) advertises the manager on the LAN via mDNS.
`run_server` also starts the data-plane router (`manager.router`) on the same
event loop, sharing this `Registry` so routing follows live heartbeat state.

Concurrency model: parallelism is by **process**, not threads. This runs in its
own process (spawned by `manager run`, or launched directly / by systemd), and
uvicorn's event loop owns the **main thread**. Route handlers are `async` so they
run on that loop rather than in FastAPI's sync threadpool, and mDNS is handled
via the async lifespan — so there are no worker threads to contend under load.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from zeroconf.asyncio import AsyncZeroconf

from oumigo import discovery
from oumigo.common.logging import configure_logging, set_verbosity
from oumigo.common.proc import die_with_parent_preexec, terminate
from oumigo.config.spec import NodeSpec
from oumigo.manager.control.registry import Registry
from oumigo.manager.control.store import MetricStore
from oumigo.manager.router.server import create_router_app
from oumigo.manager.settings import (
    build_node_spec,
    get_dashboard,
    get_data_plane,
    load_manager_yaml,
)
from oumigo.protocol.messages import (
    HeartbeatRequest,
    HeartbeatResponse,
    MetricsReport,
    RegisterRequest,
    RegisterResponse,
)

log = logging.getLogger("oumigo.manager")


def create_app(
    registry: Registry,
    token: str | None,
    heartbeat_interval: int = 10,
    status_info: dict | None = None,
    advertise_port: int | None = None,
    reaper_timeout: float | None = None,
    reaper_forget_after: float | None = None,
    node_spec: NodeSpec | None = None,
    store: MetricStore | None = None,
) -> FastAPI:
    """Build the control-plane FastAPI app.

    If `advertise_port` is set, the app advertises via mDNS from its lifespan, in a
    background task so registration happens after the socket is serving (no race)
    and without delaying startup or blocking the loop.
    """

    store = store or MetricStore()

    async def check_auth(authorization: str | None = Header(default=None)) -> None:
        if token is None:
            return  # auth disabled (v1 fail-open)
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        background: list[asyncio.Task] = []
        aiozc = service = None
        if advertise_port is not None:
            aiozc = AsyncZeroconf()
            service = discovery.build_service_info(advertise_port)

            async def _advertise() -> None:
                try:
                    await aiozc.async_register_service(service)
                    log.info("advertising via mDNS as %s (port %d)", discovery.SERVICE_TYPE, advertise_port)
                except Exception as exc:  # noqa: BLE001 - mDNS failure must not block the server
                    log.warning("mDNS advertise failed (%s); workers must use an explicit URL", exc)

            background.append(asyncio.create_task(_advertise()))

        if reaper_timeout:
            forget_after = reaper_forget_after if reaper_forget_after is not None else float("inf")
            background.append(
                asyncio.create_task(_reaper_loop(registry, reaper_timeout, forget_after))
            )

        try:
            yield
        finally:
            for bg in background:
                bg.cancel()
            if aiozc is not None:
                if service is not None:
                    try:
                        await aiozc.async_unregister_service(service)
                    except Exception:  # noqa: BLE001
                        pass
                await aiozc.async_close()

    app = FastAPI(title="oumigo manager control plane", lifespan=lifespan)

    @app.get("/spec")
    async def spec() -> dict:
        """Hand the fleet's vLLM spec to a worker so it can preflight its port and
        derive its node_id *before* registering. None until a model is configured."""
        return {"node_spec": node_spec.model_dump() if node_spec is not None else None}

    @app.post("/register", response_model=RegisterResponse)
    async def register(req: RegisterRequest, _: None = Depends(check_auth)) -> RegisterResponse:
        registry.register(
            node_id=req.node_id,
            address=req.address,
            state=req.state.value,
            incarnation=req.incarnation,
            capabilities=req.capabilities.model_dump(),
            port=req.vllm_port,
            model=req.model,
            backend=req.backend,
        )
        log.info(
            "registered worker %s at %s:%s (incarnation=%d)",
            req.node_id, req.address, req.vllm_port, req.incarnation,
        )
        if node_spec is None:
            log.warning("no model configured; node %s gets no vLLM config", req.node_id)
        return RegisterResponse(
            accepted=True,
            node_id=req.node_id,
            heartbeat_interval_s=heartbeat_interval,
            node_spec=node_spec,
        )

    @app.post("/heartbeat", response_model=HeartbeatResponse)
    async def heartbeat(req: HeartbeatRequest, _: None = Depends(check_auth)) -> HeartbeatResponse:
        run_state = req.run_state.value if req.run_state is not None else None
        known = registry.heartbeat(
            req.node_id, req.node_state.value, run_state, req.vllm_port,
            req.max_concurrent_requests,
        )
        if not known:
            log.warning("heartbeat from unknown node %s (asking it to re-register)", req.node_id)
        # STOP delivery hook: a future console/router command sets a per-node command
        # here (HeartbeatResponse.command). None for now — the pull channel is wired,
        # the trigger is not.
        return HeartbeatResponse(ok=True, known=known)

    @app.post("/metrics")
    async def metrics(report: MetricsReport, _: None = Depends(check_auth)) -> dict:
        written = store.ingest(report.node_id, report.points)
        log.debug("metrics: ingested %d points from node %s", written, report.node_id)
        return {"ok": True, "accepted": written}

    @app.get("/metrics/latest")
    async def metrics_latest() -> dict:
        # The last grid slot received per node — powers the console `metrics` command.
        return {"nodes": store.latest_per_node()}

    @app.get("/metrics/since")
    async def metrics_since(after: str = "", prefix: str = "") -> dict:
        # Watermark read seam for the reporting plane's incremental pull. Open (no
        # auth), like the other read endpoints. `prefix` is a comma list of metric
        # domains to keep (e.g. "worker:,gpu:"); empty means all.
        prefixes = [p for p in prefix.split(",") if p]
        return {"points": store.since(after, prefixes or None)}

    @app.get("/workers")
    async def workers() -> dict:
        return {"workers": [r.as_dict() for r in registry.list()]}

    @app.get("/status")
    async def status() -> dict:
        data = {"workers": registry.count(), "auth": token is not None}
        if status_info:
            data.update(status_info)
        return data

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "nodes": registry.count()}

    return app


async def _reaper_loop(registry: Registry, timeout_s: float, forget_after_s: float) -> None:
    """Periodically mark silent nodes LOST and forget long-absent ones."""
    interval = max(1.0, timeout_s / 5)
    try:
        while True:
            await asyncio.sleep(interval)
            lost, removed = registry.reap(timeout_s, forget_after_s)
            for node_id in lost:
                log.info("node %s went silent (> %.0fs) -> LOST", node_id, timeout_s)
            for node_id in removed:
                log.info("node %s forgotten after prolonged silence", node_id)
    except asyncio.CancelledError:
        pass


def run_server(
    host: str,
    port: int,
    token: str | None,
    provider_name: str,
    heartbeat_interval: int = 10,
    heartbeat_timeout: float = 30,
    forget_after: float = 14 * 86400,
    advertise: bool = True,
    verbose: bool = False,
    config_file: Path | None = None,
    node_spec: NodeSpec | None = None,
) -> None:
    """Run the manager's control plane **and** data-plane router together.

    Both FastAPI apps run on one event loop in this process and share the same
    `Registry`, so the router always routes against live heartbeat state. Blocks
    until SIGINT/SIGTERM. SIGUSR1/SIGUSR2 toggle log verbosity at runtime — used by
    an attached console in the parent process.
    """
    configure_logging(verbose)

    # Load the config once for both the vLLM spec and the data-plane bind. A child
    # server spawned by the launcher only receives --config-file, so re-read here.
    manager_config = load_manager_yaml(config_file) if config_file is not None else {}
    if node_spec is None:
        node_spec = build_node_spec(manager_config)
    data_host, data_port = get_data_plane(manager_config)
    dash_enabled, dash_host, dash_port = get_dashboard(manager_config)

    registry = Registry()
    control_app = create_app(
        registry,
        token,
        heartbeat_interval,
        status_info={"provider": provider_name, "address": f"{host}:{port}"},
        advertise_port=port if advertise else None,
        reaper_timeout=heartbeat_timeout,
        reaper_forget_after=forget_after,
        node_spec=node_spec,
    )
    router_app = create_router_app(registry, node_spec)

    log.info(
        "manager control plane on %s:%d | data plane on %s:%d "
        "(provider=%s, auth=%s, model=%s)",
        host,
        port,
        data_host,
        data_port,
        provider_name,
        "enabled" if token else "disabled",
        node_spec.model if node_spec else "unset",
    )

    signal.signal(signal.SIGUSR1, lambda *_: set_verbosity(True))   # console: verbose on
    signal.signal(signal.SIGUSR2, lambda *_: set_verbosity(False))  # console: verbose off

    # Reporting plane: bundled with the manager, but its own process for fault
    # isolation (a render hang there must never touch inference or heartbeats). It
    # pulls from this control plane over loopback, so point it at 127.0.0.1:<port>.
    dashboard = None
    if dash_enabled:
        dashboard = _spawn_dashboard(port, dash_host, dash_port, verbose)

    log_level = "info" if verbose else "warning"
    try:
        asyncio.run(
            _serve_both(
                (control_app, host, port),
                (router_app, data_host, data_port),
                log_level,
            )
        )
    finally:
        # Clean up every child this process spawned. SIGINT/SIGTERM/normal exit reach
        # here; a hard kill (SIGKILL) or crash does not — the child's PR_SET_PDEATHSIG
        # (armed at spawn) covers that case instead.
        terminate(dashboard)


def _spawn_dashboard(
    control_port: int, host: str, port: int, verbose: bool
) -> subprocess.Popen | None:
    """Start the reporting-plane dashboard as a child process, or None if it fails.

    Runs in a new session so a terminal Ctrl-C reaches only the manager, which owns
    this child's lifecycle (terminated in `run_server`'s finally). `PR_SET_PDEATHSIG`
    additionally makes the kernel SIGTERM it if the manager is hard-killed, so it can't
    orphan. A dashboard that fails to launch must never take the control plane down —
    hence the best-effort guard.
    """
    cmd = [
        sys.executable, "-m", "oumigo.manager.dashboard",
        "--host", host,
        "--port", str(port),
        "--control-url", f"http://127.0.0.1:{control_port}",
    ]
    if verbose:
        cmd.append("--verbose")
    try:
        child = subprocess.Popen(  # noqa: S603 - fixed argv
            cmd, start_new_session=True, preexec_fn=die_with_parent_preexec()
        )
    except OSError as exc:
        log.warning("reporting-plane dashboard failed to start (%s); continuing without it", exc)
        return None
    log.info("reporting-plane dashboard on %s:%d (child pid %d)", host, port, child.pid)
    return child


async def _serve_both(
    control: tuple[FastAPI, str, int],
    router: tuple[FastAPI, str, int],
    log_level: str,
) -> None:
    """Serve the control and router apps concurrently on one event loop.

    uvicorn's per-server signal handlers would clobber each other (last one wins),
    so we disable them and install a single handler that asks both to drain.
    """
    servers: list[uvicorn.Server] = []
    for app, bind_host, bind_port in (control, router):
        config = uvicorn.Config(
            app,
            host=bind_host,
            port=bind_port,
            log_config=None,
            access_log=False,
            log_level=log_level,
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # we manage shutdown centrally
        servers.append(server)

    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        for server in servers:
            server.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:  # pragma: no cover - non-Unix
            pass

    await asyncio.gather(*(server.serve() for server in servers))
