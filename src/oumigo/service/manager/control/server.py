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
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from zeroconf.asyncio import AsyncZeroconf

from oumigo import discovery
from oumigo.common.http import local_control_url
from oumigo.common.logging import configure_logging, set_verbosity
from oumigo.common.proc import die_with_parent_preexec, terminate
from oumigo.common.state import default_state_dir
from oumigo.config.spec import NodeSpec
from oumigo.providers.static import CUSTOM, LAN
from oumigo.service.manager.access import FailureLimiter, WorkerAllowlist
from oumigo.service.manager.control.registry import Registry
from oumigo.service.manager.control.store import MetricStore
from oumigo.service.manager.credentials import Authenticator, AuthStore
from oumigo.service.manager.tunnel import TunnelConn, TunnelRegistry
from oumigo.service.manager.router.server import create_router_app
from oumigo.service.manager.settings import (
    build_node_spec,
    get_agent_defaults,
    get_auth_state_dir,
    get_client_auth,
    get_control_tls,
    get_dashboard,
    get_data_plane,
    get_trusted_proxies,
    get_tunnel,
    get_worker_allowlist,
    load_manager_yaml,
)
from oumigo.protocol.messages import (
    HeartbeatRequest,
    HeartbeatResponse,
    MetricsReport,
    RegisterRequest,
    RegisterResponse,
    Transport,
)

log = logging.getLogger("oumigo.service.manager")


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
    auth: Authenticator | None = None,
    provider_name: str = LAN,
    tunnels: TunnelRegistry | None = None,
    tunnel_enabled: bool = True,
    tunnel_pool_spare: int = 1,
    worker_allowlist: list[str] | None = None,
) -> FastAPI:
    """Build the control-plane FastAPI app.

    If `advertise_port` is set, the app advertises via mDNS from its lifespan, in a
    background task so registration happens after the socket is serving (no race)
    and without delaying startup or blocking the loop.

    `auth` decides who may call the worker-facing endpoints (see
    `oumigo.service.manager.credentials`). Without one, an in-memory authenticator is
    built around `token` (the reusable fleet token), enforced only when it is set.
    `worker_allowlist` (addresses / CIDR ranges, None = off) and a failed-auth limiter
    guard the same endpoints by source address (`oumigo.service.manager.access`).
    """

    store = store or MetricStore()
    if auth is None:
        auth = Authenticator(AuthStore(), token, wan=provider_name == CUSTOM)
    tunnels = tunnels if tunnels is not None else TunnelRegistry()
    default_capacity = node_spec.max_concurrent_requests if node_spec else 4

    def _pool_size(capacity: int | None) -> int:
        """Tunnel connections a worker keeps open: its capacity plus spares, so a parked
        connection is always waiting even when every slot is busy."""
        return (capacity if capacity is not None else default_capacity) + tunnel_pool_spare

    allowlist = WorkerAllowlist(worker_allowlist)
    limiter = FailureLimiter()

    async def worker_source(request: Request) -> None:
        """Gate for worker-facing endpoints: allowlist, then failed-auth throttling."""
        host = request.client.host if request.client else None
        if not allowlist.allows(host):
            log.warning("worker request from %s refused: not in worker_allowlist", host)
            raise HTTPException(status_code=403, detail="source address not allowed")
        if limiter.blocked(host):
            raise HTTPException(status_code=429, detail="too many failed auth attempts")
        request.state.source = host

    def _deny(request: Request, node_id: str | None = None) -> HTTPException:
        limiter.record(request.state.source)
        if node_id is not None and auth.store.is_node_revoked(node_id):
            return HTTPException(status_code=403, detail=f"node {node_id} is revoked")
        return HTTPException(status_code=401, detail="invalid or missing bearer token")

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
        background.append(asyncio.create_task(_revocation_loop(registry, tunnels, auth.store)))

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

    app = FastAPI(title="OumiGo manager control plane", lifespan=lifespan)

    @app.get("/spec", dependencies=[Depends(worker_source)])
    async def spec(request: Request, authorization: str | None = Header(default=None)) -> dict:
        """Hand the fleet's vLLM spec to a worker so it can preflight its port and
        derive its node_id *before* registering. None until a model is configured.
        `provider` tells the worker which posture to enforce (CUSTOM: TLS only)."""
        if not auth.may_read_spec(authorization):
            raise _deny(request)
        return {
            "node_spec": node_spec.model_dump() if node_spec is not None else None,
            "provider": provider_name,
        }

    @app.post("/register", response_model=RegisterResponse, dependencies=[Depends(worker_source)])
    async def register(
        req: RegisterRequest, request: Request, authorization: str | None = Header(default=None)
    ) -> RegisterResponse:
        tunneled = req.transport == Transport.TUNNEL
        if tunneled and not tunnel_enabled:  # checked first: don't spend a join token on it
            raise HTTPException(400, "this manager does not accept tunnel workers (tunnel.enabled: false)")
        if not auth.may_register(authorization, req.node_id):
            raise _deny(request, req.node_id)
        registry.register(
            node_id=req.node_id,
            address=req.address,
            state=req.state.value,
            incarnation=req.incarnation,
            capabilities=req.capabilities.model_dump(),
            port=req.vllm_port,
            model=req.model,
            backend=req.backend,
            transport=req.transport.value,
        )
        log.info(
            "registered worker %s at %s:%s via %s (incarnation=%d)",
            req.node_id, req.address, req.vllm_port, req.transport.value, req.incarnation,
        )
        if node_spec is None:
            log.warning("no model configured; node %s gets no vLLM config", req.node_id)
        return RegisterResponse(
            accepted=True,
            node_id=req.node_id,
            heartbeat_interval_s=heartbeat_interval,
            node_spec=node_spec,
            node_token=auth.store.node_token(req.node_id),
            # Tunnel workers: the key their backend must require (the router presents it),
            # and how many connections to keep open. The transformer backend serves one
            # request at a time, so it needs only one slot.
            vllm_api_key=auth.store.vllm_api_key(req.node_id) if tunneled else None,
            tunnel_pool_size=(
                _pool_size(1 if req.backend == "transformer" else None) if tunneled else None
            ),
        )

    @app.post("/heartbeat", response_model=HeartbeatResponse, dependencies=[Depends(worker_source)])
    async def heartbeat(
        req: HeartbeatRequest, request: Request, authorization: str | None = Header(default=None)
    ) -> HeartbeatResponse:
        if not auth.may_act_as(authorization, req.node_id):
            raise _deny(request, req.node_id)
        run_state = req.run_state.value if req.run_state is not None else None
        known = registry.heartbeat(
            req.node_id, req.node_state.value, run_state, req.vllm_port,
            req.max_concurrent_requests, req.tunnel_rtt_ms,
        )
        if not known:
            log.warning("heartbeat from unknown node %s (asking it to re-register)", req.node_id)
        # STOP delivery hook: a future console/router command sets a per-node command
        # here (HeartbeatResponse.command). None for now — the pull channel is wired,
        # the trigger is not.
        record = registry.get(req.node_id) if known else None
        pool_size = None
        if record is not None and record.transport == Transport.TUNNEL.value:
            pool_size = _pool_size(record.max_concurrent_requests)
        return HeartbeatResponse(ok=True, known=known, tunnel_pool_size=pool_size)

    @app.websocket("/tunnel")
    async def tunnel(ws: WebSocket) -> None:
        """A tunnel worker's outbound connection: parked until the router borrows it to
        carry one HTTP/1.1 connection to the worker's backend. Refused before the upgrade
        (HTTP 403) unless it carries a valid node token of a registered tunnel worker."""
        host = ws.client.host if ws.client else None
        if not allowlist.allows(host) or limiter.blocked(host):
            log.warning("tunnel from %s refused: not allowed or throttled", host)
            await ws.close(code=1008)
            return
        node_id = auth.tunnel_node(ws.headers.get("authorization"))
        record = registry.get(node_id) if node_id is not None else None
        if (
            not tunnel_enabled
            or record is None
            or record.transport != Transport.TUNNEL.value
        ):
            if node_id is None:
                limiter.record(host)  # a bad credential, not just "not registered yet"
            log.debug("tunnel refused (node=%s)", node_id)
            await ws.close(code=1008)
            return
        await ws.accept()
        conn = TunnelConn(ws, node_id)
        tunnels.add(conn)
        try:
            await conn.pump()  # returns when either side closes the connection
        finally:
            tunnels.remove(conn)

    @app.post("/metrics", dependencies=[Depends(worker_source)])
    async def metrics(
        report: MetricsReport, request: Request, authorization: str | None = Header(default=None)
    ) -> dict:
        if not auth.may_act_as(authorization, report.node_id):
            raise _deny(request, report.node_id)
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
        rows = []
        for record in registry.list():
            row = record.as_dict()
            if record.transport == Transport.TUNNEL.value:
                row.update(tunnels.stats(record.node_id))
            rows.append(row)
        return {"workers": rows}

    @app.get("/status")
    async def status() -> dict:
        data = {"workers": registry.count(), "auth": auth.enforced}
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


async def _revocation_loop(
    registry: Registry, tunnels: TunnelRegistry, auth_store: AuthStore, interval: float = 2.0
) -> None:
    """Enforce `oumigo manager node revoke` (written by the CLI into the auth store) on
    a running manager: forget the node and close its open tunnel connections. Its node
    token is already refused on the next call; this cuts the connections it holds."""
    try:
        while True:
            await asyncio.sleep(interval)
            revoked = auth_store.revoked_nodes()
            for record in registry.list():
                if record.node_id in revoked:
                    registry.remove(record.node_id)
                    closed = await tunnels.close_node(record.node_id)
                    log.warning("node %s revoked: dropped (closed %d tunnel connections)",
                                record.node_id, closed)
    except asyncio.CancelledError:
        pass


def run_server(
    host: str,
    port: int,
    token: str | None,
    provider_name: str,
    heartbeat_interval: int = 10,
    heartbeat_timeout: float = 30,
    forget_after: float = 3600,  # 1 hour
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

    # Auth state (signing secret, minted join tokens, revoked nodes) persists in the state
    # dir so node tokens survive a manager restart and the CLI can mint/revoke while we run.
    state_dir = get_auth_state_dir(manager_config) or default_state_dir()
    auth_store = AuthStore.at(state_dir)
    auth = Authenticator(auth_store, token, wan=provider_name == CUSTOM)
    log.info("auth state in %s (worker auth %s)", auth_store.path,
             "enforced" if auth.enforced else "DISABLED")
    client_auth = get_client_auth(manager_config)
    log.info("router client auth %s", "REQUIRED (client API keys)" if client_auth else "off")
    allowlist = get_worker_allowlist(manager_config)
    if allowlist is not None:
        log.info("worker_allowlist ON: %s", ", ".join(allowlist) or "(empty: no worker allowed)")

    tunnel_enabled, pool_spare = get_tunnel(manager_config)
    registry = Registry()
    tunnels = TunnelRegistry()  # shared with the router on this one event loop
    control_app = create_app(
        registry,
        token,
        heartbeat_interval,
        status_info={"provider": provider_name, "address": f"{host}:{port}"},
        advertise_port=port if advertise else None,
        reaper_timeout=heartbeat_timeout,
        reaper_forget_after=forget_after,
        node_spec=node_spec,
        auth=auth,
        provider_name=provider_name,
        tunnels=tunnels,
        tunnel_enabled=tunnel_enabled,
        tunnel_pool_spare=pool_spare,
        worker_allowlist=allowlist,
    )
    router_app = create_router_app(registry, node_spec,
                                   agent_defaults=get_agent_defaults(manager_config),
                                   tunnels=tunnels, auth_store=auth_store,
                                   client_auth=client_auth)

    log.info(
        "manager control plane on %s:%d | data plane on %s:%d "
        "(provider=%s, auth=%s, model=%s)",
        host,
        port,
        data_host,
        data_port,
        provider_name,
        "enforced" if auth.enforced else "disabled",
        node_spec.model if node_spec else "unset",
    )

    signal.signal(signal.SIGUSR1, lambda *_: set_verbosity(True))   # console: verbose on
    signal.signal(signal.SIGUSR2, lambda *_: set_verbosity(False))  # console: verbose off

    # Control-plane transport: optional built-in TLS (normally nginx terminates it), and
    # X-Forwarded-For believed only from the trusted proxies (nginx), so the client IP
    # the allowlist sees can't be spoofed by a worker sending its own header.
    control_opts: dict = {
        "proxy_headers": True,
        "forwarded_allow_ips": get_trusted_proxies(manager_config),
    }
    tls = get_control_tls(manager_config)
    if tls is not None:
        control_opts.update(ssl_certfile=tls[0], ssl_keyfile=tls[1])
        log.info("control plane serves TLS itself (%s)", tls[0])

    # Reporting plane: bundled with the manager, but its own process for fault
    # isolation (a render hang there must never touch inference or heartbeats). It
    # pulls from this control plane over loopback, so point it at 127.0.0.1:<port>.
    dashboard = None
    if dash_enabled:
        dashboard = _spawn_dashboard(port, dash_host, dash_port, verbose, tls=tls is not None)

    log_level = "info" if verbose else "warning"
    try:
        asyncio.run(
            _serve_both(
                (control_app, host, port, control_opts),
                (router_app, data_host, data_port, {}),
                log_level,
            )
        )
    finally:
        # Clean up every child this process spawned. SIGINT/SIGTERM/normal exit reach
        # here; a hard kill (SIGKILL) or crash does not — the child's PR_SET_PDEATHSIG
        # (armed at spawn) covers that case instead.
        terminate(dashboard)


def _spawn_dashboard(
    control_port: int, host: str, port: int, verbose: bool, tls: bool = False
) -> subprocess.Popen | None:
    """Start the reporting-plane dashboard as a child process, or None if it fails.

    Runs in a new session so a terminal Ctrl-C reaches only the manager, which owns
    this child's lifecycle (terminated in `run_server`'s finally). `PR_SET_PDEATHSIG`
    additionally makes the kernel SIGTERM it if the manager is hard-killed, so it can't
    orphan. A dashboard that fails to launch must never take the control plane down —
    hence the best-effort guard.
    """
    cmd = [
        sys.executable, "-m", "oumigo.service.manager.dashboard",
        "--host", host,
        "--port", str(port),
        "--control-url", local_control_url(control_port, tls),
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
    control: tuple[FastAPI, str, int, dict],
    router: tuple[FastAPI, str, int, dict],
    log_level: str,
) -> None:
    """Serve the control and router apps concurrently on one event loop.

    uvicorn's per-server signal handlers would clobber each other (last one wins),
    so we disable them and install a single handler that asks both to drain.
    """
    servers: list[uvicorn.Server] = []
    for app, bind_host, bind_port, extra in (control, router):
        config = uvicorn.Config(
            app,
            host=bind_host,
            port=bind_port,
            log_config=None,
            access_log=False,
            log_level=log_level,
            **extra,
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
