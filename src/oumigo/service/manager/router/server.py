"""Data-plane router — forwards OpenAI-compatible client calls to worker vLLMs.

The router is health-aware: it only ever forwards to workers the control-plane
`Registry` reports as SERVING (vLLM up and accepting requests). It shares that
`Registry` object in-process with the control plane, so selection always sees the
latest heartbeat-driven state without a network hop.

Dispatch is a **FIFO admission queue** (`WorkerPool`), not round-robin. Each request
is handed to a random eligible worker (one serving the model it names) that has a free
slot; when every eligible worker is at capacity the request waits in a single FIFO
queue until a slot it can use frees (or a new worker joins).
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

Transport, per worker: a `direct` worker is dialed at that address. A `tunnel` worker
dialed out to the manager instead (`oumigo.service.manager.tunnel`); its requests go
through a second httpx client whose connections are borrowed tunnel connections, with
the node's backend API key in place of the client's `Authorization` header. A tunnel
worker is routable only while it has at least one live tunnel connection — a good
heartbeat alone doesn't prove the data path works.

Endpoints are a thin passthrough of the vLLM OpenAI API (`/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/v1/models`), including **SSE streaming** — the
upstream response is relayed chunk-by-chunk.

Client auth: with `client_auth` on (always under provider CUSTOM, opt-in under LAN),
`/v1/*`, `/agent-defaults` and `/list` require `Authorization: Bearer <client API key>`
(`oumigo manager apikey create`); `/healthz` and `/version` stay open. The client's key
is checked here and never forwarded: it is dropped before the request goes to any
worker, `direct` or `tunnel`.

Model field: routing is **by model id**, and a model id is a catalog profile id
(`gemma-4-12B-it-qat-w4a16-ct@128k`): every start-time configuration is its own model.
A request naming an id goes only to workers running that profile; `any-model` (or no
`model` at all) goes to any SERVING worker, and the router then names the chosen
worker's id in the request, so the response says which model answered. An id the fleet
doesn't know is `404 model_not_found`. With no SERVING worker for the request, it is
`503 model_loading` (with `Retry-After`) while a matching worker is still starting, else
`503 model_unavailable`; a request that queued too long behind busy workers is
`503 model_busy`. All are in OpenAI's error shape, and their messages are written for
the end user (the details go to the log). `/v1/models` is answered here, from the
registry: every id with a SERVING worker, each with its task, and `any-model` (a chat
entry) while at least one of them is a chat model.

Tasks: a model is for `chat` or `embed` (its profile's `task`, as the worker reports
it). The chat paths, `any-model` included, only reach chat workers; naming an embedding
model there is `400 wrong_task`. `/v1/embeddings` only reaches embed workers and needs
an explicit model (`400 model_required`): vectors of two embedding models never mix, so
a random pick would quietly corrupt a caller's index. An embedding request may carry
`input_type` (`query`, `document`, ...), an oumigo extension: the router puts the
model's prompt for it (profile `embed.prompts`) before each input and drops the field,
so clients need not know each model's instructions. An unknown type is
`400 invalid_input_type`.

Capability caps: on `/v1/chat/completions` the router may also trim over-long audio
and clamp `max_tokens`, both **off unless the chosen worker's profile sets
`caps.max_audio_seconds` / `caps.max_output_tokens`**. These are limits of the *model* rather
than business policy — the router is the only component that knows which model the
fleet serves — so they live here even though request shaping otherwise belongs to a
gateway in front. See `oumigo.service.manager.router.shaping` for what was measured
and why. A trimmed request comes back with `x-oumigo-audio-*` response headers.
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
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute

from oumigo import __version__
from oumigo.config.catalog import ANY_MODEL, FleetCatalog, Profile
from oumigo.config.spec import TASK_CHAT, TASK_EMBED, NodeSpec, Task
from oumigo.service.manager.control.registry import NodeRecord, Registry
from oumigo.service.manager.credentials import AuthStore, bearer
from oumigo.service.manager.router.shaping import (
    cap_audio,
    clamp_output_tokens,
    shaping_headers,
)
from oumigo.service.manager.tunnel import TunnelRegistry, TunnelTransport, tunnel_host
from oumigo.protocol.messages import Transport
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

CHAT_PATH = "/v1/chat/completions"  # the only path carrying messages worth shaping
EMBEDDINGS_PATH = "/v1/embeddings"

DEFAULT_QUEUE_TIMEOUT_S = 300.0  # give up (503) if all workers stay saturated this long
# Upstream connection reuse. Idle connections are kept this long — below the worker
# backend's own idle timeout (`VLLM_HTTP_TIMEOUT_KEEP_ALIVE`, which the worker raises to
# 120s) so the router never reuses a connection the backend is closing. httpx's 5s
# default meant most requests after a short pause paid a fresh connect (+1-2 RTT on WAN).
# No connection cap: `WorkerPool` already bounds in-flight requests per worker, and
# httpx's default of 100 across *all* workers would queue a large fleet inside httpx.
UPSTREAM_LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=None, keepalive_expiry=60.0)
_PROMOTE_INTERVAL_S = 0.2        # re-check the queue this often (catches newly-joined workers)


RETRY_AFTER_S = 30  # advised to clients while a model is starting or busy
_STARTING = {NodeState.REGISTERING.value, NodeState.INITIALIZING.value}


class NoWorkersAvailable(RuntimeError):
    """Raised by `WorkerPool.acquire` when no eligible worker is SERVING at all."""


Eligible = Callable[[NodeRecord], bool]  # which workers a request accepts


class WorkerPool:
    """FIFO admission control over the SERVING workers.

    A request acquires a *slot* on a worker before it's forwarded and releases it when
    the response finishes. Selection: any SERVING worker whose in-flight count is below
    its capacity is eligible; one is chosen at random. A request may narrow that set
    (`eligible`, e.g. "serves model X"). When none are free the caller joins a single
    FIFO queue and is woken as slots free up: a freed slot goes to the **oldest waiter
    that accepts that worker**, so a busy model never holds up requests for another one,
    and order is kept within each model. A worker that never returns a response never
    frees its slots, so it drops out of the eligible set and the queue routes around it
    (fault isolation).

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
        tunnels: TunnelRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.tunnels = tunnels
        self.default_capacity = max(1, default_capacity)
        self.queue_timeout = queue_timeout
        self._rng = rng or random
        self._inflight: dict[str, int] = defaultdict(int)
        self._waiters: deque[tuple[asyncio.Future[NodeRecord], Eligible | None]] = deque()

    def _capacity(self, record: NodeRecord) -> int:
        cap = record.max_concurrent_requests
        return cap if cap is not None else self.default_capacity

    def routable(self, record: NodeRecord) -> bool:
        """SERVING, and (for a tunnel worker) with a tunnel connection up."""
        return self._routable(record)

    def _routable(self, record: NodeRecord) -> bool:
        if record.state != NodeState.SERVING.value:
            return False
        if record.transport == Transport.TUNNEL.value:
            return self.tunnels is not None and self.tunnels.connected(record.node_id) > 0
        return True

    def serving(self, eligible: Eligible | None = None) -> list[NodeRecord]:
        return [r for r in self.registry.list()
                if self._routable(r) and (eligible is None or eligible(r))]

    def _available(self, eligible: Eligible | None = None) -> list[NodeRecord]:
        return [r for r in self.serving(eligible)
                if self._inflight[r.node_id] < self._capacity(r)]

    def healthy_count(self, eligible: Eligible | None = None) -> int:
        return len(self.serving(eligible))

    async def acquire(self, eligible: Eligible | None = None,
                      timeout: float | None = None) -> NodeRecord:
        """Return a worker (among `eligible`) with a free slot, waiting FIFO if all are
        saturated.

        Raises `NoWorkersAvailable` when no eligible worker is serving (nothing to wait
        for) and `asyncio.TimeoutError` if the queue wait exceeds `timeout` (default:
        the pool's `queue_timeout`).
        """
        # Earlier waiters first: whatever they can use goes to them. A free slot that is
        # left over is ours if we accept that worker.
        self._promote()
        avail = self._available(eligible)
        if avail:
            return self._take(self._rng.choice(avail))

        if self.healthy_count(eligible) == 0:  # nothing serving -> fail fast, don't hang
            raise NoWorkersAvailable

        fut: asyncio.Future[NodeRecord] = asyncio.get_running_loop().create_future()
        self._waiters.append((fut, eligible))
        try:
            return await asyncio.wait_for(fut, timeout or self.queue_timeout)
        except BaseException:
            # Timed out, or the client disconnected while queued. Drop our waiter; if a
            # slot was handed to us in the race window, give it back.
            if any(waiter is fut for waiter, _ in self._waiters):
                self._waiters = deque(w for w in self._waiters if w[0] is not fut)
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
        """Give available slots to waiters, oldest first, each on a worker it accepts."""
        if not self._waiters:
            return
        kept: deque[tuple[asyncio.Future[NodeRecord], Eligible | None]] = deque()
        for fut, eligible in self._waiters:
            if fut.done():  # cancelled/timed-out (or already resolved) -> discard
                continue
            avail = self._available(eligible)
            if avail:
                fut.set_result(self._take(self._rng.choice(avail)))
            else:
                kept.append((fut, eligible))
        self._waiters = kept

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


def _model_error(status: int, code: str, message: str,
                 retry_after: int | None = None, param: str = "model") -> JSONResponse:
    """An OpenAI-shaped error about the request's `model` (or another `param`), so SDKs
    surface it properly.

    `message` is shown to end users by client apps: say what they need, no internals.
    """
    kind = "invalid_request_error" if status < 500 else "server_error"
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse(status_code=status, headers=headers, content={
        "error": {"message": message, "type": kind, "param": param, "code": code}})


class _InputTypeError(ValueError):
    """`input_type` can't be applied; the message is for the end user."""


def apply_input_type(payload: dict, prompts: dict[str, str]) -> None:
    """Replace an embedding request's `input_type` with the model's prompt for it.

    The named prompt is put before every input string and `input_type` is removed, so
    the backend sees plain OpenAI input. Without `input_type` the payload is untouched.
    Raises `_InputTypeError` for a type the model has no prompt for, or for token-id
    input (a text prefix can't be put before token ids).
    """
    if "input_type" not in payload:
        return
    input_type = payload.pop("input_type")
    if input_type not in prompts:
        valid = ", ".join(sorted(prompts))
        raise _InputTypeError(
            f"{payload.get('model')} has no input_type {input_type!r}; valid: {valid}."
            if valid else f"{payload.get('model')} takes no input_type.")
    prefix = prompts[input_type]
    inputs = payload.get("input")
    if isinstance(inputs, str):
        payload["input"] = prefix + inputs
    elif isinstance(inputs, list) and all(isinstance(item, str) for item in inputs):
        payload["input"] = [prefix + item for item in inputs]
    else:
        raise _InputTypeError("input_type needs text input, not token ids.")


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


_RETRYABLE = (httpx.ConnectError, httpx.WriteError, httpx.RemoteProtocolError)


async def _send_with_retry(
    client: httpx.AsyncClient, request: httpx.Request, attempts: int
) -> httpx.Response:
    """Send (response body not yet read), retrying a failure that happened before any
    response arrived — so a retry can never re-run a generation that already started."""
    for attempt in range(attempts):
        try:
            return await client.send(request, stream=True)
        except _RETRYABLE as exc:
            if attempt + 1 >= attempts:
                raise
            log.info("upstream connection failed before a response (%s); retrying", exc)
    raise AssertionError("unreachable")


def create_router_app(
    registry: Registry,
    node_spec: NodeSpec | None = None,
    queue_timeout: float = DEFAULT_QUEUE_TIMEOUT_S,
    agent_defaults: dict[str, float | None] | None = None,
    tunnels: TunnelRegistry | None = None,
    auth_store: AuthStore | None = None,
    client_auth: bool = False,
    catalog: FleetCatalog | None = None,
    fleet_agent_defaults: dict[str, float | None] | None = None,
) -> FastAPI:
    """Build the data-plane FastAPI app that proxies to healthy worker vLLMs.

    `tunnels` (shared with the control plane) and `auth_store` (to derive each tunnel
    worker's backend API key) enable routing to `tunnel` workers; without them only
    `direct` workers are routable. `client_auth` requires a client API key from
    `auth_store` on the inference endpoints.

    Routing is by model id (= catalog profile id; see the module docstring). `catalog`
    supplies the allowed ids and each profile's caps and client timeouts; `node_spec`
    (the fleet default) covers workers registered without a profile.
    `agent_defaults` are the default profile's timeouts, `fleet_agent_defaults` the
    manager's own `agent:` block (for profiles without one).
    """
    if client_auth and auth_store is None:
        raise ValueError("client_auth needs an auth_store holding the client API keys")
    default_capacity = node_spec.max_concurrent_requests if node_spec else 4
    vllm_port = node_spec.port if node_spec else None
    default_served = node_spec.served_name if node_spec else None
    no_limits = {"turn_timeout": None, "stall_timeout": None}

    def _served(record: NodeRecord) -> str | None:
        """The model id a worker answers to (a pre-0.5 worker: the fleet default's)."""
        return record.served_name or default_served

    def _profile(model_id: str | None) -> Profile | None:
        return catalog.get(model_id) if catalog is not None and model_id else None

    def _is_chat(record: NodeRecord) -> bool:
        """`any-model` and the chat paths only ever reach chat workers."""
        return record.task == TASK_CHAT

    def _task_of(model_id: str) -> str:
        """A model id's task: its profile's, else what its workers report (open fleet)."""
        profile = _profile(model_id)
        if profile is not None:
            return profile.task
        for record in registry.list():
            if _served(record) == model_id:
                return record.task
        if node_spec is not None and model_id == default_served:
            return node_spec.task
        return TASK_CHAT

    def _prompts(model_id: str | None) -> dict[str, str]:
        """An embedding model's `input_type` -> prefix map (empty without a profile)."""
        profile = _profile(model_id)
        return profile.prompts if profile is not None else {}

    def _caps(record: NodeRecord) -> tuple[float | None, int | None]:
        """(max_audio_seconds, max_output_tokens) for the worker's model; None = off."""
        profile = _profile(record.profile)
        if profile is not None:
            return profile.caps.max_audio_seconds, profile.caps.max_output_tokens
        if node_spec is not None:
            return node_spec.max_audio_seconds, node_spec.max_output_tokens
        return None, None

    def _queue_seconds(model_id: str | None) -> float:
        """How long a request for `model_id` may queue; `any-model` (None) may take any
        serving chat worker, so it gets the longest wait among the chat models served."""
        if model_id is None:
            served = {_served(r) for r in pool.serving(_is_chat)} - {None}
            return max((_queue_seconds(m) for m in served), default=queue_timeout)
        profile = _profile(model_id)
        return (profile.caps.max_queue_seconds if profile else None) or queue_timeout

    def _agent(model_id: str | None) -> dict[str, float | None]:
        """Client timeouts for one model id."""
        profile = _profile(model_id)
        if profile is None:
            return dict(agent_defaults or no_limits)
        if profile.agent is not None:
            return profile.agent.model_dump()
        return dict(fleet_agent_defaults or no_limits)

    def _not_ready(model_id: str | None, eligible: Eligible | None) -> JSONResponse:
        """No SERVING worker for the request: is one on its way, or is there none?"""
        records = [r for r in registry.list() if eligible is None or eligible(r)]
        starting = [r for r in records if r.state in _STARTING
                    or (r.state == NodeState.SERVING.value and not pool.routable(r))]
        log.info("no routable worker for %s: %s", model_id or ANY_MODEL,
                 ", ".join(f"{r.node_id}={r.state}" for r in records) or "none registered")
        if starting:
            message = (f"{model_id} is starting, retry shortly." if model_id
                       else "A model is starting, retry shortly.")
            return _model_error(503, "model_loading", message, retry_after=RETRY_AFTER_S)
        message = (f"{model_id} is not available right now." if model_id
                   else "No model is available right now.")
        return _model_error(503, "model_unavailable", message)

    def _known() -> set[str]:
        """Model ids this fleet answers for: allowed profiles plus whatever registered
        workers serve (an open fleet has no catalog)."""
        ids = set(catalog.profiles) if catalog is not None else set()
        ids.update(m for m in (_served(r) for r in registry.list()) if m)
        if default_served:
            ids.add(default_served)
        return ids

    pool = WorkerPool(registry, default_capacity, queue_timeout, tunnels=tunnels)
    timeout = httpx.Timeout(600.0, connect=10.0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One shared client; no total read timeout so long generations aren't cut off.
        async with (
            httpx.AsyncClient(timeout=timeout, limits=UPSTREAM_LIMITS) as client,
            httpx.AsyncClient(
                timeout=timeout,
                transport=TunnelTransport(tunnels or TunnelRegistry(), UPSTREAM_LIMITS),
            ) as tunnel_client,
        ):
            app.state.client = client
            app.state.tunnel_client = tunnel_client
            promoter = asyncio.create_task(pool.run_promoter())
            try:
                yield
            finally:
                promoter.cancel()

    app = FastAPI(title="OumiGo data plane (router)", lifespan=lifespan)

    def require_client(request: Request) -> None:
        """Refuse a request without a valid client API key (when client auth is on)."""
        if not client_auth:
            return
        if auth_store.check_api_key(bearer(request.headers.get("authorization"))) is None:
            raise HTTPException(
                401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"}
            )

    authed = [Depends(require_client)]

    async def _forward(request: Request, method: str, path: str,
                       task: Task = TASK_CHAT) -> Response:
        # The body names the model, which decides the eligible workers, so it is read
        # and parsed before a worker is chosen. Non-JSON / non-dict bodies pass through
        # untouched and may go to any worker of the path's task.
        body = await request.body()
        payload = None
        if body:
            try:
                payload = json.loads(body)
            except (ValueError, TypeError):
                payload = None
        requested = payload.get("model") if isinstance(payload, dict) else None
        named = requested if requested not in (None, "", ANY_MODEL) else None
        if named is None and task == TASK_EMBED:
            # Vectors of two embedding models never mix, so the caller must say which.
            available = sorted(m for m in _known() if _task_of(m) == TASK_EMBED)
            return _model_error(
                400, "model_required",
                "Name an embedding model"
                + (f"; available: {', '.join(available)}." if available else "."))
        if named is not None:
            if named not in _known():
                return _model_error(
                    404, "model_not_found",
                    f"Model {named!r} is not available. Available models: "
                    f"{', '.join(sorted(_known()) + ([ANY_MODEL] if task == TASK_CHAT else []))}.")
            if _task_of(named) != task:
                return _model_error(
                    400, "wrong_task",
                    f"{named} is an embedding model; use /v1/embeddings."
                    if task == TASK_CHAT else f"{named} is not an embedding model.")
        if task == TASK_EMBED and isinstance(payload, dict) and "input_type" in payload:
            try:
                apply_input_type(payload, _prompts(named))
            except _InputTypeError as exc:
                return _model_error(400, "invalid_input_type", str(exc), param="input_type")
            body = json.dumps(payload).encode()
        if named is not None:
            eligible: Eligible = lambda r: _served(r) == named and r.task == task  # noqa: E731
        else:
            eligible = _is_chat

        try:
            record = await pool.acquire(eligible, _queue_seconds(named))
        except NoWorkersAvailable:
            return _not_ready(named, eligible)
        except asyncio.TimeoutError:
            log.info("queue wait timed out for %s", named or ANY_MODEL)
            message = (f"{named} is busy, retry shortly." if named is not None
                       else "All models are busy, retry shortly.")
            return _model_error(503, "model_busy", message, retry_after=RETRY_AFTER_S)

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
            headers = _filter_request_headers(request.headers)  # drops content-length; httpx recomputes
            if client_auth:
                # The client's API key was checked above; it never leaves the manager.
                headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            tunneled = record.transport == Transport.TUNNEL.value
            if tunneled:
                if auth_store is None:
                    raise HTTPException(503, "tunnel workers are not routable (no auth store)")
                url = f"http://{tunnel_host(record.node_id)}:{port}" + path
                client: httpx.AsyncClient = request.app.state.tunnel_client
                # The backend requires the node's key; the client's own credential (if any)
                # never leaves the manager.
                headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
                headers["Authorization"] = f"Bearer {auth_store.vllm_api_key(record.node_id)}"
            else:
                url = f"http://{record.address}:{port}" + path
                client = request.app.state.client

            # Name the chosen worker's model (needed for `any-model`; a no-op for a
            # request that named it), then apply that model's caps.
            streaming = False
            extra_headers: dict[str, str] = {}
            if isinstance(payload, dict):
                streaming = bool(payload.get("stream"))
                mutated = False
                served = _served(record)
                if served is not None and payload.get("model") != served:
                    payload["model"] = served
                    mutated = True
                max_audio_seconds, max_output_tokens = _caps(record)
                if path == CHAT_PATH:
                    # Capability caps: trim audio the model cannot use, and bound the
                    # reply so a degenerate generation can't hold the slot forever.
                    shaping = cap_audio(payload, max_audio_seconds)
                    if shaping.trimmed:
                        extra_headers = shaping_headers(shaping)
                        mutated = True
                    mutated |= clamp_output_tokens(payload, max_output_tokens)
                if mutated:
                    body = json.dumps(payload).encode()

            # A tunnel connection can die while parked (worker restart, NAT drop). Nothing
            # reached the backend if sending failed, so one retry on a fresh connection is
            # safe; anything later (a partial response) is surfaced as 502 instead.
            attempts = 2 if tunneled else 1

            if streaming:
                up_req = client.build_request(method, url, content=body, headers=headers)
                upstream = await _send_with_retry(client, up_req, attempts)
                # Slot is released by _relay's finally (stream end or client disconnect).
                return StreamingResponse(
                    _relay(upstream, _release),
                    status_code=upstream.status_code,
                    # X-Accel-Buffering: an nginx in front must not buffer SSE tokens.
                    headers={**_filter_response_headers(upstream), **extra_headers,
                             "X-Accel-Buffering": "no"},
                    media_type=upstream.headers.get("content-type"),
                )
            up_req = client.build_request(method, url, content=body or None, headers=headers)
            upstream = await _send_with_retry(client, up_req, attempts)
            try:
                await upstream.aread()
            finally:
                await upstream.aclose()
            response = Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers={**_filter_response_headers(upstream), **extra_headers},
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

    @app.post(CHAT_PATH, dependencies=authed)
    async def chat_completions(request: Request) -> Response:
        """proxy an OpenAI chat completion to a healthy worker vLLM"""
        return await _forward(request, "POST", CHAT_PATH)

    @app.post("/v1/completions", dependencies=authed)
    async def completions(request: Request) -> Response:
        """proxy an OpenAI text completion to a healthy worker vLLM"""
        return await _forward(request, "POST", "/v1/completions")

    @app.post(EMBEDDINGS_PATH, dependencies=authed)
    async def embeddings(request: Request) -> Response:
        """proxy OpenAI embeddings to a worker of the named embedding model"""
        return await _forward(request, "POST", EMBEDDINGS_PATH, TASK_EMBED)

    @app.get("/v1/models", dependencies=authed)
    async def models() -> dict:
        """list the models with a serving worker, plus any-model while a chat model is up"""
        serving: dict[str, list[NodeRecord]] = defaultdict(list)
        for record in pool.serving():
            served = _served(record)
            if served:
                serving[served].append(record)
        data = []
        if any(_task_of(model_id) == TASK_CHAT for model_id in serving):
            # any-model only ever reaches chat workers, so it is listed (and labelled)
            # only while one of them can answer it.
            data.append({"id": ANY_MODEL, "object": "model", "owned_by": "oumigo",
                         "oumigo": {"task": TASK_CHAT}})
        for model_id in sorted(serving):
            records = serving[model_id]
            profile = _profile(model_id)
            weights = profile.model if profile else (records[0].model or model_id)
            if profile is not None:
                max_len = profile.engine.max_model_len
            elif node_spec is not None and model_id == default_served:
                max_len = node_spec.max_model_len
            else:
                max_len = None
            task = _task_of(model_id)
            info: dict = {"weights": weights, "task": task, "max_model_len": max_len}
            if task == TASK_EMBED:
                info["dimensions"] = profile.embed.dimensions if profile and profile.embed else None
                info["input_types"] = sorted(_prompts(model_id))
            info["serving"] = len(records)
            data.append({
                "id": model_id,
                "object": "model",
                "owned_by": weights.split("/", 1)[0] if "/" in weights else "oumigo",
                "oumigo": info,
            })
        return {"object": "list", "data": data}

    @app.get("/agent-defaults", dependencies=authed)
    async def agent_defaults_(model: str | None = None) -> dict:
        """client-side defaults for a model; for any-model, the longest of those served"""
        if model not in (None, "", ANY_MODEL):
            return _agent(model)
        # Any serving chat model may answer, so the client must be prepared to wait as long
        # as the slowest one needs. None (no limit) outranks every number.
        served = sorted({_served(r) for r in pool.serving(_is_chat)} - {None})
        if not served:
            return dict(agent_defaults or no_limits)
        merged: dict[str, float | None] = {}
        for limits in map(_agent, served):
            for key, value in limits.items():
                current = merged.get(key, 0.0)
                merged[key] = None if value is None or current is None else max(current, value)
        return merged

    @app.get("/healthz")
    async def healthz() -> dict:
        """report data-plane liveness and the healthy-worker count"""
        n = pool.healthy_count()
        return {"status": "ok" if n else "no_workers", "healthy_workers": n}

    @app.get("/version")
    async def version() -> dict:
        """show the package version"""
        return {"version": __version__}

    @app.get("/list", dependencies=authed)
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
