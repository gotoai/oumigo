"""Data-plane router — forwards OpenAI-compatible client calls to worker vLLMs.

The router is health-aware: it only ever forwards to workers the control-plane
`Registry` reports as SERVING (vLLM up and accepting requests). It shares that
`Registry` object in-process with the control plane, so selection always sees the
latest heartbeat-driven state without a network hop.

Dispatch is a **FIFO admission queue** (`WorkerPool`), not round-robin. Each request
is handed to a random worker that has a free slot; when every worker is at capacity
the request waits in a single FIFO queue until a slot frees (or a new worker joins).
Capacity is per worker and **negotiated** — a worker's reported
`max_concurrent_requests` (defaulting to the fleet's `node_spec.max_concurrent_requests`).
This gives backpressure and fault isolation: a stuck worker simply stops freeing its
slots, so the queue flows to the others instead of piling onto it — one bad node can
no longer stall the whole fleet.

Homogeneous *model*, per-worker *port*: every worker serves the same model, but each
may run vLLM on a different port — it preflight-negotiates a free one at startup
(preferred = `node_spec.port`) and reports it on its heartbeat. So a target is
`http://{worker.address}:{worker.port}`, falling back to the fleet-default
`node_spec.port` for a worker that hasn't reported its port yet.

Endpoints are a thin passthrough of the vLLM OpenAI API (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`), including **SSE streaming** — the upstream response
is relayed chunk-by-chunk.

Model field: a client **may** send a `model` in the request body, but the router
**always ignores and overwrites it** with the fleet's real model name (the model the
manager configured, handed to every worker at registration). The homogeneous fleet
serves exactly one model, so the manager is the source of truth — a client can't 404
by naming a model the worker's vLLM doesn't serve.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute

from oumigo import __version__
from oumigo.config.spec import NodeSpec
from oumigo.service.manager.control.registry import NodeRecord, Registry
from oumigo.protocol.states import NodeState

log = logging.getLogger("oumigo.service.manager.router")

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

DEFAULT_QUEUE_TIMEOUT_S = 300.0  # give up (503) if all workers stay saturated this long
_PROMOTE_INTERVAL_S = 0.2        # re-check the queue this often (catches newly-joined workers)


class NoWorkersAvailable(RuntimeError):
    """Raised by `WorkerPool.acquire` when the fleet has no SERVING worker at all."""


class WorkerPool:
    """FIFO admission control over the SERVING workers.

    A request acquires a *slot* on a worker before it's forwarded and releases it when
    the response finishes. Selection: any SERVING worker whose in-flight count is below
    its capacity is eligible; one is chosen at random. When none are eligible the caller
    joins a single FIFO queue and is woken — in arrival order — as slots free up. A
    worker that never returns a response never frees its slots, so it drops out of the
    eligible set and the queue routes around it (fault isolation).

    Capacity is per worker: the registry record's negotiated `max_concurrent_requests`,
    falling back to `default_capacity` when the worker hasn't reported one.

    Runs entirely on the manager's single event loop, so the in-flight map and the
    waiter queue need no locks — no `await` happens inside a bookkeeping critical section.
    """

    def __init__(
        self,
        registry: Registry,
        default_capacity: int,
        queue_timeout: float = DEFAULT_QUEUE_TIMEOUT_S,
        rng: random.Random | None = None,
    ) -> None:
        self.registry = registry
        self.default_capacity = max(1, default_capacity)
        self.queue_timeout = queue_timeout
        self._rng = rng or random
        self._inflight: dict[str, int] = defaultdict(int)
        self._waiters: deque[asyncio.Future[NodeRecord]] = deque()

    def _capacity(self, record: NodeRecord) -> int:
        cap = record.max_concurrent_requests
        return cap if cap is not None else self.default_capacity

    def _serving(self) -> list[NodeRecord]:
        return [r for r in self.registry.list() if r.state == NodeState.SERVING.value]

    def _available(self) -> list[NodeRecord]:
        return [r for r in self._serving() if self._inflight[r.node_id] < self._capacity(r)]

    def healthy_count(self) -> int:
        return len(self._serving())

    async def acquire(self) -> NodeRecord:
        """Return a worker with a free slot, waiting FIFO if all are saturated.

        Raises `NoWorkersAvailable` when the fleet is empty (nothing to wait for) and
        `asyncio.TimeoutError` if the queue wait exceeds `queue_timeout`.
        """
        # Fast path: nobody queued ahead of us and a slot is free -> take it now.
        if not self._waiters:
            avail = self._available()
            if avail:
                return self._take(self._rng.choice(avail))

        if self.healthy_count() == 0:  # nothing serving -> fail fast instead of hanging
            raise NoWorkersAvailable

        fut: asyncio.Future[NodeRecord] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            return await asyncio.wait_for(fut, self.queue_timeout)
        except BaseException:
            # Timed out, or the client disconnected while queued. Drop our waiter; if a
            # slot was handed to us in the race window, give it back.
            if fut in self._waiters:
                self._waiters.remove(fut)
            elif fut.done() and not fut.cancelled() and fut.exception() is None:
                self.release(fut.result().node_id)
            raise

    def _take(self, record: NodeRecord) -> NodeRecord:
        self._inflight[record.node_id] += 1
        return record

    def release(self, node_id: str) -> None:
        """Free one slot on `node_id` and hand any freed capacity to waiting requests."""
        if self._inflight.get(node_id, 0) > 0:
            self._inflight[node_id] -= 1
        self._promote()

    def _promote(self) -> None:
        """Give available slots to the oldest waiters, in FIFO order."""
        while self._waiters:
            avail = self._available()
            if not avail:
                return
            fut = self._waiters[0]
            if fut.done():  # cancelled/timed-out (or already resolved) -> discard
                self._waiters.popleft()
                continue
            record = self._take(self._rng.choice(avail))
            self._waiters.popleft()
            fut.set_result(record)

    async def run_promoter(self, interval: float = _PROMOTE_INTERVAL_S) -> None:
        """Periodically retry the queue so capacity that appears via a newly-SERVING
        worker (not a `release`) still wakes waiters. Cheap: only acts while queued."""
        try:
            while True:
                await asyncio.sleep(interval)
                if self._waiters:
                    self._promote()
        except asyncio.CancelledError:
            pass


def _filter_request_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST}


def _filter_response_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESPONSE}


async def _relay(upstream: httpx.Response, on_done: Callable[[], None]):
    """Yield the upstream body chunk-by-chunk, then close and release the slot.

    `on_done` runs in the generator's finally, so the worker's slot is freed when the
    stream completes *or* the client disconnects mid-stream.
    """
    try:
        async for chunk in upstream.aiter_bytes():
            yield chunk
    finally:
        await upstream.aclose()
        on_done()


def create_router_app(
    registry: Registry,
    node_spec: NodeSpec | None = None,
    queue_timeout: float = DEFAULT_QUEUE_TIMEOUT_S,
) -> FastAPI:
    """Build the data-plane FastAPI app that proxies to healthy worker vLLMs."""
    default_capacity = node_spec.max_concurrent_requests if node_spec else 4
    vllm_port = node_spec.port if node_spec else None
    fleet_model = node_spec.model if node_spec else None  # authoritative model name
    pool = WorkerPool(registry, default_capacity, queue_timeout)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One shared client; no total read timeout so long generations aren't cut off.
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            app.state.client = client
            promoter = asyncio.create_task(pool.run_promoter())
            try:
                yield
            finally:
                promoter.cancel()

    app = FastAPI(title="oumigo data plane (router)", lifespan=lifespan)

    async def _forward(request: Request, method: str, path: str) -> Response:
        try:
            record = await pool.acquire()
        except NoWorkersAvailable:
            raise HTTPException(503, "no healthy (SERVING) workers available") from None
        except asyncio.TimeoutError:
            raise HTTPException(503, "all workers saturated; request queue timed out") from None

        released = False

        def _release() -> None:
            nonlocal released
            if not released:
                released = True
                pool.release(record.node_id)

        try:
            # Each worker reports its actual (preflight-negotiated) port; fall back to the
            # fleet default from node_spec for workers that haven't reported one yet.
            port = record.port if record.port is not None else vllm_port
            if port is None:
                raise HTTPException(503, "no model configured; workers have no vLLM to route to")
            url = f"http://{record.address}:{port}" + path
            body = await request.body()
            headers = _filter_request_headers(request.headers)  # drops content-length; httpx recomputes
            client: httpx.AsyncClient = request.app.state.client

            # Parse the body once: detect streaming AND overwrite the client's `model`
            # with the fleet's real model name (spec: the client's value is ignored, the
            # manager is authoritative). Non-JSON / non-dict bodies pass through untouched.
            streaming = False
            if body:
                try:
                    payload = json.loads(body)
                except (ValueError, TypeError):
                    payload = None
                if isinstance(payload, dict):
                    streaming = bool(payload.get("stream"))
                    if fleet_model is not None:
                        payload["model"] = fleet_model
                        body = json.dumps(payload).encode()

            if streaming:
                up_req = client.build_request(method, url, content=body, headers=headers)
                upstream = await client.send(up_req, stream=True)
                # Slot is released by _relay's finally (stream end or client disconnect).
                return StreamingResponse(
                    _relay(upstream, _release),
                    status_code=upstream.status_code,
                    headers=_filter_response_headers(upstream),
                    media_type=upstream.headers.get("content-type"),
                )
            upstream = await client.request(method, url, content=body or None, headers=headers)
            response = Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=_filter_response_headers(upstream),
                media_type=upstream.headers.get("content-type"),
            )
            _release()
            return response
        except HTTPException:
            _release()
            raise
        except httpx.HTTPError as exc:
            _release()
            log.warning("upstream error routing to %s: %s", record.address, exc)
            raise HTTPException(502, f"upstream error talking to worker {record.address}: {exc}") from exc
        except BaseException:  # client disconnect / cancellation: never leak the slot
            _release()
            raise

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        """proxy an OpenAI chat completion to a healthy worker vLLM"""
        return await _forward(request, "POST", "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        """proxy an OpenAI text completion to a healthy worker vLLM"""
        return await _forward(request, "POST", "/v1/completions")

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        """list the model(s) served by the fleet"""
        return await _forward(request, "GET", "/v1/models")

    @app.get("/healthz")
    async def healthz() -> dict:
        """report data-plane liveness and the healthy-worker count"""
        n = pool.healthy_count()
        return {"status": "ok" if n else "no_workers", "healthy_workers": n}

    @app.get("/version")
    async def version() -> dict:
        """show the package version"""
        return {"version": __version__}

    @app.get("/list")
    async def list_commands() -> dict:
        """show the supported API commands"""
        paths: dict[str, dict] = {}
        for route in app.routes:
            if not isinstance(route, APIRoute):
                continue  # skip the auto docs/openapi routes
            summary = (route.endpoint.__doc__ or "").strip().splitlines()[0] if route.endpoint.__doc__ else ""
            for method in sorted(m for m in route.methods if m not in ("HEAD", "OPTIONS")):
                paths.setdefault(route.path, {})[method.lower()] = {
                    "summary": summary,
                    "operationId": route.unique_id,
                }
        return {"paths": paths}

    return app
