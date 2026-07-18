"""Manager control-plane HTTP server (v1).

Accepts worker registrations and heartbeats, tracks them in an in-memory
`Registry`, and (unless disabled) advertises the manager on the LAN via mDNS.
No vLLM / routing yet.

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
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from zeroconf.asyncio import AsyncZeroconf

from oumigo import discovery
from oumigo.common.logging import configure_logging, set_verbosity
from oumigo.manager.control.registry import Registry
from oumigo.protocol.messages import (
    HeartbeatRequest,
    HeartbeatResponse,
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
) -> FastAPI:
    """Build the control-plane FastAPI app.

    If `advertise_port` is set, the app advertises via mDNS from its lifespan, in a
    background task so registration happens after the socket is serving (no race)
    and without delaying startup or blocking the loop.
    """

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

    @app.post("/register", response_model=RegisterResponse)
    async def register(req: RegisterRequest, _: None = Depends(check_auth)) -> RegisterResponse:
        registry.register(
            node_id=req.node_id,
            address=req.address,
            state=req.state.value,
            incarnation=req.incarnation,
            capabilities=req.capabilities.model_dump(),
        )
        log.info("registered node %s at %s (incarnation=%d)", req.node_id, req.address, req.incarnation)
        return RegisterResponse(
            accepted=True, node_id=req.node_id, heartbeat_interval_s=heartbeat_interval
        )

    @app.post("/heartbeat", response_model=HeartbeatResponse)
    async def heartbeat(req: HeartbeatRequest, _: None = Depends(check_auth)) -> HeartbeatResponse:
        known = registry.heartbeat(req.node_id, req.state.value)
        if not known:
            log.warning("heartbeat from unknown node %s (asking it to re-register)", req.node_id)
        return HeartbeatResponse(ok=True, known=known)

    @app.get("/nodes")
    async def nodes() -> dict:
        return {"nodes": [r.as_dict() for r in registry.list()]}

    @app.get("/status")
    async def status() -> dict:
        data = {"nodes": registry.count(), "auth": token is not None}
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
) -> None:
    """Run the control-plane server in the foreground (event loop on the main thread).

    Blocks until SIGINT/SIGTERM (uvicorn's own handling). SIGUSR1/SIGUSR2 toggle log
    verbosity at runtime — used by an attached console in the parent process.
    """
    configure_logging(verbose)

    registry = Registry()
    app = create_app(
        registry,
        token,
        heartbeat_interval,
        status_info={"provider": provider_name, "address": f"{host}:{port}"},
        advertise_port=port if advertise else None,
        reaper_timeout=heartbeat_timeout,
        reaper_forget_after=forget_after,
    )

    log.info(
        "manager control plane on %s:%d (provider=%s, auth=%s)",
        host,
        port,
        provider_name,
        "enabled" if token else "disabled",
    )

    signal.signal(signal.SIGUSR1, lambda *_: set_verbosity(True))   # console: verbose on
    signal.signal(signal.SIGUSR2, lambda *_: set_verbosity(False))  # console: verbose off

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        log_level="info" if verbose else "warning",
    )
