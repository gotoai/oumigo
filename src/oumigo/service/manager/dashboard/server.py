"""Reporting-plane web server (V1.0).

Its own FastAPI app / uvicorn process — fault-isolated from the control plane and
router by design (this is the riskiest, least-critical code; a render hang here
must never touch inference or heartbeats). A background task polls the control
plane into a `MetricMirror`; request handlers read that mirror and roll it up
on the fly. Strictly read-only: no control actions live here.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from oumigo.service.manager.dashboard.aggregate import gpu_util_series
from oumigo.service.manager.dashboard.source import MetricMirror
from oumigo.service.manager.dashboard.web import INDEX_HTML

log = logging.getLogger("oumigo.service.manager.dashboard")


def create_dashboard_app(control_url: str, *, poll_interval_s: float = 5.0) -> FastAPI:
    """Build the dashboard app, wiring a background pull loop into its lifespan."""
    mirror = MetricMirror(control_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # No keep-alive reuse: the poll interval (~10s) outlives uvicorn's idle
        # keep-alive timeout (~5s), so a pooled connection is reliably closed
        # server-side by the next pull. Reusing it races the server's close and
        # surfaces spurious `RemoteProtocolError: Server disconnected` warnings.
        # A fresh connection per pull costs nothing at this cadence and removes it.
        async with httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=0)
        ) as client:
            await mirror.refresh(client)  # seed once so the first page isn't empty
            stop = asyncio.Event()

            async def poll_loop() -> None:
                while not stop.is_set():
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=poll_interval_s)
                    except asyncio.TimeoutError:
                        await mirror.refresh(client)  # interval elapsed -> pull

            task = asyncio.create_task(poll_loop())
            try:
                yield
            finally:
                stop.set()
                task.cancel()

    app = FastAPI(title="oumigo reporting plane (dashboard)", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/api/sheets/gpu_util")
    async def sheet_gpu_util() -> dict:
        return gpu_util_series(
            mirror.snapshot(), datetime.now(timezone.utc), node_info=mirror.node_info
        )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "control_url": control_url,
            "buffered_points": len(mirror),
            "last_pull_ok": mirror.last_ok.strftime("%Y-%m-%d %H:%M:%S") if mirror.last_ok else None,
            "last_error": mirror.last_error,
        }

    return app


def run_dashboard(
    host: str,
    port: int,
    control_url: str,
    *,
    poll_interval_s: float = 10.0,
    verbose: bool = False,
) -> None:
    """Run the reporting-plane dashboard in the foreground (blocks until stopped)."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("reporting-plane dashboard on %s:%d (control=%s)", host, port, control_url)
    app = create_dashboard_app(control_url, poll_interval_s=poll_interval_s)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        log_level="info" if verbose else "warning",
    )
