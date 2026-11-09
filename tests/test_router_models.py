"""Multi-model routing: requests go to workers of the model id they name (= profile id)."""

from __future__ import annotations

import asyncio
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from starlette.testclient import TestClient

from oumigo.config.catalog import Caps, FleetCatalog, Profile
from oumigo.protocol.states import NodeState
from oumigo.service.manager.control.registry import Registry
from oumigo.service.manager.router.server import WorkerPool, create_router_app

SMALL, BIG, IDLE = "small-model@32k", "big-model@128k", "idle-model@8k"


class _Echo(BaseHTTPRequestHandler):
    last_body: dict | None = None

    def do_POST(self):  # noqa: N802 - required handler name
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        _Echo.last_body = body
        payload = json.dumps({"echo_model": body.get("model")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def upstreams():
    servers = [HTTPServer(("127.0.0.1", 0), _Echo) for _ in range(2)]
    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield [srv.server_address for srv in servers]
    for srv in servers:
        srv.shutdown()


def _profile(pid: str, model: str, agent: dict | None = None, **caps) -> Profile:
    return Profile(id=pid, model=model, caps=Caps(**caps), agent=agent)


def _fleet(addrs, agents: dict | None = None, queue_caps: dict | None = None, **router_kwargs):
    """Two serving workers: SMALL (output cap 100) and BIG (no cap); IDLE has none."""
    agents, queue_caps = agents or {}, queue_caps or {}
    catalog = FleetCatalog({p.id: p for p in (
        _profile(pid, model, agents.get(pid), max_queue_seconds=queue_caps.get(pid), **caps)
        for pid, model, caps in ((SMALL, "acme/small-model", {"max_output_tokens": 100}),
                                 (BIG, "acme/big-model", {}),
                                 (IDLE, "acme/idle-model", {}))
    )}, SMALL)
    reg = Registry()
    for node_id, (host, port), pid in (("small", addrs[0], SMALL), ("big", addrs[1], BIG)):
        reg.register(node_id, host, NodeState.SERVING.value, 0, {}, port=port,
                     model=catalog.get(pid).model, profile=pid)
        reg.heartbeat(node_id, NodeState.SERVING.value)
    return create_router_app(reg, catalog, **router_kwargs)


def test_requests_go_to_workers_of_the_named_model(upstreams) -> None:
    with TestClient(_fleet(upstreams)) as client:
        for model in (SMALL, BIG):
            for _ in range(4):
                resp = client.post("/v1/chat/completions",
                                   json={"model": model, "messages": [], "max_tokens": 500})
                assert resp.status_code == 200
                assert resp.json()["echo_model"] == model
                # Caps are the chosen worker's profile's: only SMALL clamps.
                assert _Echo.last_body["max_tokens"] == (100 if model == SMALL else 500)


def test_any_model_reaches_either_and_names_the_choice(upstreams) -> None:
    seen = set()
    with TestClient(_fleet(upstreams)) as client:
        for _ in range(30):
            resp = client.post("/v1/chat/completions", json={"model": "any-model", "messages": []})
            assert resp.status_code == 200
            seen.add(resp.json()["echo_model"])
    assert seen == {SMALL, BIG}


def test_vendor_repo_name_is_not_a_model_id(upstreams) -> None:
    with TestClient(_fleet(upstreams)) as client:
        resp = client.post("/v1/chat/completions", json={"model": "acme/big-model", "messages": []})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


def test_allowed_model_without_a_ready_worker_is_503(upstreams) -> None:
    with TestClient(_fleet(upstreams)) as client:
        resp = client.post("/v1/chat/completions", json={"model": IDLE, "messages": []})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_unavailable"


def test_models_lists_only_models_with_a_serving_worker(upstreams) -> None:
    with TestClient(_fleet(upstreams)) as client:
        data = client.get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == ["any-model", BIG, SMALL]  # not IDLE
    assert data[0] == {"id": "any-model", "object": "model", "owned_by": "oumigo",
                       "oumigo": {"task": "chat"}}
    assert data[1]["owned_by"] == "acme"
    assert data[1]["oumigo"] == {"weights": "acme/big-model", "task": "chat",
                                 "max_model_len": None, "serving": 1}


def test_agent_defaults_per_model(upstreams) -> None:
    app = _fleet(upstreams, agent_defaults={"turn_timeout": 90.0, "stall_timeout": None},
                 fleet_agent_defaults={"turn_timeout": 60.0, "stall_timeout": 30.0})
    with TestClient(app) as client:
        # BIG has no `agent:` of its own -> the manager's block.
        assert client.get(f"/agent-defaults?model={BIG}").json() == {
            "turn_timeout": 60.0, "stall_timeout": 30.0}


def test_any_model_agent_defaults_are_the_longest_served(upstreams) -> None:
    """Any serving model may answer an any-model request, so the client gets the longest
    timeouts among them; None (no limit) beats any number. IDLE serves nothing."""
    app = _fleet(upstreams, fleet_agent_defaults={"turn_timeout": 60.0, "stall_timeout": 30.0},
                 agents={SMALL: {"turn_timeout": 1800.0, "stall_timeout": None},
                         IDLE: {"turn_timeout": 9999.0, "stall_timeout": 9999.0}})
    with TestClient(app) as client:
        for query in ("", "?model=any-model"):
            assert client.get(f"/agent-defaults{query}").json() == {
                "turn_timeout": 1800.0, "stall_timeout": None}
        assert client.get(f"/agent-defaults?model={IDLE}").json()["turn_timeout"] == 9999.0


async def test_queue_wait_follows_the_models_max_queue_seconds() -> None:
    reg = Registry()
    reg.register("a", "10.0.0.1", NodeState.SERVING.value, 0, {}, profile=SMALL)
    reg.heartbeat("a", NodeState.SERVING.value, max_concurrent_requests=1)
    pool = WorkerPool(reg, 1, queue_timeout=0.05)
    held = await pool.acquire()
    with pytest.raises(asyncio.TimeoutError):           # the pool's default
        await pool.acquire()
    waiting = asyncio.create_task(pool.acquire(timeout=5.0))  # a longer per-model limit
    await asyncio.sleep(0.2)
    pool.release(held.node_id)
    assert (await asyncio.wait_for(waiting, 1.0)).node_id == "a"


def test_router_uses_the_profiles_queue_limit(upstreams) -> None:
    seen = []
    app = _fleet(upstreams, queue_caps={BIG: 1800.0})
    original = WorkerPool.acquire

    async def spy(self, eligible=None, timeout=None):
        seen.append(timeout)
        return await original(self, eligible, timeout)

    WorkerPool.acquire = spy
    try:
        with TestClient(app) as client:
            for model in (BIG, SMALL, "any-model"):
                client.post("/v1/chat/completions", json={"model": model, "messages": []})
    finally:
        WorkerPool.acquire = original
    # BIG: its own; SMALL: the router default; any-model: the longest served.
    assert seen == [1800.0, 300.0, 1800.0]


async def test_a_busy_model_does_not_block_another() -> None:
    reg = Registry()
    for node_id, pid in (("a", SMALL), ("b", BIG)):
        reg.register(node_id, "10.0.0.1", NodeState.SERVING.value, 0, {}, profile=pid)
        reg.heartbeat(node_id, NodeState.SERVING.value, max_concurrent_requests=1)
    pool = WorkerPool(reg, 1, queue_timeout=2.0, rng=random.Random(0))
    wants = {pid: (lambda r, pid=pid: r.served_name == pid) for pid in (SMALL, BIG)}

    first = await pool.acquire(wants[SMALL])                    # SMALL's only slot
    queued = asyncio.create_task(pool.acquire(wants[SMALL]))    # waits for SMALL
    await asyncio.sleep(0.01)
    other = await asyncio.wait_for(pool.acquire(wants[BIG]), 1.0)  # not stuck behind it
    assert (first.node_id, other.node_id) == ("a", "b") and not queued.done()

    pool.release(first.node_id)
    assert (await asyncio.wait_for(queued, 1.0)).node_id == "a"


# --- not ready: errors an end user reads -------------------------------------------------


def _router(*workers):
    """A catalog fleet whose workers are (node_id, profile, state) with no backend."""
    catalog = FleetCatalog({p.id: p for p in (
        _profile(SMALL, "acme/small-model"), _profile(BIG, "acme/big-model"))}, SMALL)
    reg = Registry()
    for node_id, pid, state in workers:
        reg.register(node_id, "10.0.0.9", state, 0, {}, port=7001, profile=pid)
        reg.heartbeat(node_id, state)
    return create_router_app(reg, catalog, queue_timeout=0.05)


@pytest.mark.parametrize("workers,model,code,message", [
    ([], SMALL, "model_unavailable", f"{SMALL} is not available right now."),
    ([], "any-model", "model_unavailable", "No model is available right now."),
    ([("a", SMALL, NodeState.INITIALIZING.value)], SMALL, "model_loading",
     f"{SMALL} is starting, retry shortly."),
    ([("a", SMALL, NodeState.REGISTERING.value)], "any-model", "model_loading",
     "A model is starting, retry shortly."),
    ([("a", BIG, NodeState.INITIALIZING.value)], SMALL, "model_unavailable",   # another model
     f"{SMALL} is not available right now."),
    ([("a", SMALL, NodeState.LOST.value)], SMALL, "model_unavailable",
     f"{SMALL} is not available right now."),
])
def test_not_ready_errors_say_what_the_user_needs(workers, model, code, message) -> None:
    with TestClient(_router(*workers)) as client:
        resp = client.post("/v1/chat/completions", json={"model": model, "messages": []})
    assert resp.status_code == 503
    assert resp.json()["error"] == {"message": message, "type": "server_error",
                                    "param": "model", "code": code}
    assert resp.headers.get("retry-after") == ("30" if code == "model_loading" else None)


def test_a_serving_tunnel_worker_without_its_tunnel_yet_is_starting() -> None:
    reg = Registry()
    reg.register("t", "10.0.0.9", NodeState.SERVING.value, 0, {}, port=7001, profile=SMALL,
                 transport="tunnel")
    reg.heartbeat("t", NodeState.SERVING.value)
    catalog = FleetCatalog({SMALL: _profile(SMALL, "acme/small-model")}, SMALL)
    app = create_router_app(reg, catalog)
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"model": SMALL, "messages": []})
    assert resp.json()["error"]["code"] == "model_loading"


async def test_queue_timeout_is_model_busy(upstreams) -> None:
    import httpx

    app = _fleet(upstreams, queue_timeout=0.05)
    pool_holder = []
    original = WorkerPool.acquire

    async def hold_all(self, eligible=None, timeout=None):
        pool_holder.append(self)
        return await original(self, eligible, timeout)

    WorkerPool.acquire = hold_all
    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                await client.post("/v1/chat/completions", json={"model": SMALL, "messages": []})
                pool = pool_holder[0]
                pool._inflight["small"] = 99            # SMALL's only worker is saturated
                resp = await client.post("/v1/chat/completions",
                                         json={"model": SMALL, "messages": []})
    finally:
        WorkerPool.acquire = original
    assert resp.status_code == 503 and resp.headers["retry-after"] == "30"
    assert resp.json()["error"]["code"] == "model_busy"
    assert resp.json()["error"]["message"] == f"{SMALL} is busy, retry shortly."
