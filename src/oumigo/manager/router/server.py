"""Data-plane router — forwards OpenAI-compatible client calls to worker vLLMs.

The router is health-aware: it only ever forwards to workers the control-plane
`Registry` reports as SERVING (vLLM up and accepting requests). It shares that
`Registry` object in-process with the control plane, so selection always sees the
latest heartbeat-driven state without a network hop.

Homogeneous *model*, per-worker *port*: every worker serves the same model, but
each may run vLLM on a different port — it preflight-negotiates a free one at
startup (preferred = `node_spec.port`) and reports it on its heartbeat. So a target
is `http://{worker.address}:{worker.port}`, falling back to the fleet-default
`node_spec.port` for a worker that hasn't reported its port yet.

Endpoints are a thin passthrough of the vLLM OpenAI API (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`), including **SSE streaming** — the request body is
forwarded verbatim and the upstream response is relayed chunk-by-chunk.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from oumigo.config.spec import NodeSpec
from oumigo.manager.control.registry import NodeRecord, Registry
from oumigo.protocol.states import NodeState, RunState

log = logging.getLogger("oumigo.manager.router")

# Hop-by-hop / length headers we must not blindly copy across the proxy boundary.
_DROP_REQUEST = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "upgrade"}
_DROP_RESPONSE = {
    "content-length",
    "content-encoding",   # httpx decodes the body, so the original encoding no longer applies
    "transfer-encoding",
    "connection",
    "keep-alive",
    "content-type",       # carried separately as the response media_type
}


class WorkerSelector:
    """Picks a healthy worker per request. Round-robin, or a simple least-loaded.

    ``round_robin`` cycles SERVING workers evenly. ``least_loaded`` prefers workers
    whose last heartbeat reported ``IDLE`` (no in-flight work), cycling among those;
    it falls back to round-robin over all SERVING workers when none are idle.
    """

    def __init__(self, registry: Registry, strategy: str = "round_robin") -> None:
        self.registry = registry
        self.strategy = strategy
        self._rr = 0
        self._lock = threading.Lock()

    def _healthy(self) -> list[NodeRecord]:
        return [r for r in self.registry.list() if r.state == NodeState.SERVING.value]

    def healthy_count(self) -> int:
        return len(self._healthy())

    def pick(self) -> NodeRecord | None:
        healthy = self._healthy()
        if not healthy:
            return None
        pool = healthy
        if self.strategy == "least_loaded":
            idle = [r for r in healthy if r.run_state == RunState.IDLE.value]
            pool = idle or healthy
        with self._lock:
            record = pool[self._rr % len(pool)]
            self._rr += 1
        return record


def _filter_request_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST}


def _filter_response_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESPONSE}


async def _relay(upstream: httpx.Response):
    """Yield the upstream body chunk-by-chunk, then close the connection."""
    try:
        async for chunk in upstream.aiter_bytes():
            yield chunk
    finally:
        await upstream.aclose()


def create_router_app(
    registry: Registry,
    node_spec: NodeSpec | None = None,
    strategy: str = "round_robin",
) -> FastAPI:
    """Build the data-plane FastAPI app that proxies to healthy worker vLLMs."""
    selector = WorkerSelector(registry, strategy)
    vllm_port = node_spec.port if node_spec else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One shared client; no total read timeout so long generations aren't cut off.
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            app.state.client = client
            yield

    app = FastAPI(title="oumigo data plane (router)", lifespan=lifespan)

    def _upstream_base() -> str:
        record = selector.pick()
        if record is None:
            raise HTTPException(503, "no healthy (SERVING) workers available")
        # Each worker reports its actual (preflight-negotiated) port; fall back to the
        # fleet default from node_spec for workers that haven't reported one yet.
        port = record.port if record.port is not None else vllm_port
        if port is None:
            raise HTTPException(503, "no model configured; workers have no vLLM to route to")
        return f"http://{record.address}:{port}"

    async def _forward(request: Request, method: str, path: str) -> Response:
        base = _upstream_base()
        url = base + path
        body = await request.body()
        headers = _filter_request_headers(request.headers)
        client: httpx.AsyncClient = request.app.state.client

        streaming = False
        if body:
            try:
                streaming = bool(json.loads(body).get("stream"))
            except (ValueError, TypeError):
                streaming = False

        try:
            if streaming:
                up_req = client.build_request(method, url, content=body, headers=headers)
                upstream = await client.send(up_req, stream=True)
                return StreamingResponse(
                    _relay(upstream),
                    status_code=upstream.status_code,
                    headers=_filter_response_headers(upstream),
                    media_type=upstream.headers.get("content-type"),
                )
            upstream = await client.request(
                method, url, content=body or None, headers=headers
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=_filter_response_headers(upstream),
                media_type=upstream.headers.get("content-type"),
            )
        except httpx.HTTPError as exc:
            log.warning("upstream error routing to %s: %s", base, exc)
            raise HTTPException(502, f"upstream error talking to worker at {base}: {exc}") from exc

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _forward(request, "POST", "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _forward(request, "POST", "/v1/completions")

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        return await _forward(request, "GET", "/v1/models")

    @app.get("/healthz")
    async def healthz() -> dict:
        n = selector.healthy_count()
        return {"status": "ok" if n else "no_workers", "healthy_workers": n}

    return app
