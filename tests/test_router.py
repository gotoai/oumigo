"""Tests for the data-plane router: FIFO worker admission + forwarding passthrough."""

from __future__ import annotations

import asyncio
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from starlette.testclient import TestClient

from oumigo.config.spec import NodeSpec
from oumigo.manager.control.registry import Registry
from oumigo.manager.router.server import NoWorkersAvailable, WorkerPool, create_router_app
from oumigo.protocol.states import NodeState, RunState


def _register(
    reg: Registry, node_id: str, address: str, state: NodeState,
    run_state: RunState | None = None, max_concurrent: int | None = None,
) -> None:
    reg.register(node_id, address, state.value, 0, {})
    reg.heartbeat(node_id, state.value, run_state.value if run_state else None,
                  max_concurrent_requests=max_concurrent)


def _pool(reg: Registry, default_capacity: int = 4, seed: int = 0) -> WorkerPool:
    return WorkerPool(reg, default_capacity, queue_timeout=2.0, rng=random.Random(seed))


# --- admission / selection ------------------------------------------------------


async def test_pool_only_admits_serving_workers() -> None:
    reg = Registry()
    _register(reg, "a", "10.0.0.1", NodeState.SERVING)
    _register(reg, "b", "10.0.0.2", NodeState.INITIALIZING)   # not ready
    _register(reg, "c", "10.0.0.3", NodeState.LOST)           # dead
    pool = _pool(reg)
    picked = set()
    for _ in range(6):
        rec = await pool.acquire()
        picked.add(rec.address)
        pool.release(rec.node_id)   # release so we exercise selection, not saturation
    assert picked == {"10.0.0.1"}
    assert pool.healthy_count() == 1


async def test_pool_spreads_across_available_workers() -> None:
    reg = Registry()
    for i in range(3):
        _register(reg, f"n{i}", f"10.0.0.{i}", NodeState.SERVING)
    pool = _pool(reg, default_capacity=1)
    picked = {(await pool.acquire()).address for _ in range(3)}  # each has 1 slot
    assert picked == {"10.0.0.0", "10.0.0.1", "10.0.0.2"}  # all three used, none blocked


async def test_pool_no_workers_raises() -> None:
    with pytest.raises(NoWorkersAvailable):
        await _pool(Registry()).acquire()


async def test_pool_fifo_queue_when_saturated() -> None:
    reg = Registry()
    _register(reg, "solo", "10.0.0.1", NodeState.SERVING)
    pool = _pool(reg, default_capacity=1)

    first = await pool.acquire()                 # takes the only slot
    t2 = asyncio.create_task(pool.acquire())     # must queue
    t3 = asyncio.create_task(pool.acquire())     # queues behind t2
    await asyncio.sleep(0.05)
    assert not t2.done() and not t3.done()       # both waiting; the worker is full

    pool.release(first.node_id)
    await asyncio.sleep(0.05)
    assert t2.done() and not t3.done()           # FIFO: t2 served first
    pool.release((await t2).node_id)
    assert (await t3).address == "10.0.0.1"      # then t3


async def test_pool_stuck_worker_is_isolated() -> None:
    """A worker that never releases its slot stops receiving new work."""
    reg = Registry()
    _register(reg, "stuck", "10.0.0.1", NodeState.SERVING)
    _register(reg, "good", "10.0.0.2", NodeState.SERVING)
    pool = _pool(reg, default_capacity=1)

    # Take one slot on each worker; hold the "stuck" one forever (never released).
    a = await pool.acquire()
    b = await pool.acquire()
    assert {a.address, b.address} == {"10.0.0.1", "10.0.0.2"}
    good = a if a.address == "10.0.0.2" else b

    # Releasing + re-acquiring the good worker always returns the good worker; the
    # stuck node is at capacity, so it's never eligible.
    for _ in range(5):
        pool.release(good.node_id)
        good = await pool.acquire()
        assert good.address == "10.0.0.2"


async def test_pool_honors_negotiated_capacity() -> None:
    reg = Registry()
    _register(reg, "big", "10.0.0.1", NodeState.SERVING, max_concurrent=2)
    pool = _pool(reg, default_capacity=1)  # default 1, but the worker negotiated 2
    a = await pool.acquire()
    b = await pool.acquire()               # second slot allowed by the negotiated cap
    assert a.address == b.address == "10.0.0.1"
    t3 = asyncio.create_task(pool.acquire())   # third must queue
    await asyncio.sleep(0.05)
    assert not t3.done()
    pool.release(a.node_id)
    assert (await t3).address == "10.0.0.1"


async def test_pool_queue_times_out_when_never_freed() -> None:
    reg = Registry()
    _register(reg, "solo", "10.0.0.1", NodeState.SERVING)
    pool = WorkerPool(reg, 1, queue_timeout=0.2, rng=random.Random(0))
    await pool.acquire()  # saturate
    with pytest.raises(asyncio.TimeoutError):
        await pool.acquire()  # nothing ever frees -> queue timeout


# --- forwarding (against a real upstream on localhost) --------------------------


class _MockVLLM(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - required handler name
        if self.path == "/v1/models":
            self._json({"data": [{"id": "acme/mock"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        self._json({"echo_model": body.get("model"), "path": self.path})

    def _json(self, obj):
        payload = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture
def upstream():
    srv = HTTPServer(("127.0.0.1", 0), _MockVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address  # (host, port)
    srv.shutdown()


def _router_app(upstream_addr):
    host, port = upstream_addr
    reg = Registry()
    _register(reg, "w1", host, NodeState.SERVING)
    return create_router_app(reg, NodeSpec(model="acme/mock", port=port))


def test_forward_chat_completion(upstream) -> None:
    # `with` triggers the lifespan that opens the shared httpx client.
    with TestClient(_router_app(upstream)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "acme/mock", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert resp.json() == {"echo_model": "acme/mock", "path": "/v1/chat/completions"}


def test_router_overwrites_client_model_with_fleet_model(upstream) -> None:
    # Client asks for a bogus model; the router must overwrite it with the fleet's
    # real model name (NodeSpec.model) before forwarding, so vLLM never 404s.
    with TestClient(_router_app(upstream)) as client:  # fleet model = "acme/mock"
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "client/whatever", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert resp.json()["echo_model"] == "acme/mock"  # upstream saw the fleet model, not the client's


def test_forward_models(upstream) -> None:
    with TestClient(_router_app(upstream)) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "acme/mock"


def test_503_when_no_healthy_workers() -> None:
    app = create_router_app(Registry(), NodeSpec(model="acme/mock", port=9))
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"model": "acme/mock", "messages": []})
    assert resp.status_code == 503


def test_503_when_no_model_configured(upstream) -> None:
    reg = Registry()
    _register(reg, "w1", upstream[0], NodeState.SERVING)
    app = create_router_app(reg, node_spec=None)  # no vLLM port to route to
    with TestClient(app) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 503


def test_healthz_reports_worker_count(upstream) -> None:
    with TestClient(_router_app(upstream)) as client:
        body = client.get("/healthz").json()
    assert body == {"status": "ok", "healthy_workers": 1}
