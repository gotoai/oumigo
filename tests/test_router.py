"""Tests for the data-plane router: FIFO worker admission + forwarding passthrough."""

from __future__ import annotations

import asyncio
import base64
import json
import random
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from starlette.testclient import TestClient

from oumigo.config.spec import NodeSpec
from oumigo.service.manager.control.registry import Registry
from oumigo.service.manager.router.server import NoWorkersAvailable, WorkerPool, create_router_app
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

    last_body: dict | None = None   # what the router actually forwarded, for cap tests

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _MockVLLM.last_body = body
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


def _router_app(upstream_addr, **spec_kwargs):
    host, port = upstream_addr
    reg = Registry()
    _register(reg, "w1", host, NodeState.SERVING)
    return create_router_app(reg, NodeSpec(model="acme/mock", port=port, **spec_kwargs))


def test_forward_chat_completion(upstream) -> None:
    # `with` triggers the lifespan that opens the shared httpx client.
    with TestClient(_router_app(upstream)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "acme/mock", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert resp.json() == {"echo_model": "acme/mock", "path": "/v1/chat/completions"}


def test_unknown_model_is_404_in_openai_shape(upstream) -> None:
    with TestClient(_router_app(upstream)) as client:  # serves "acme/mock"
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "client/whatever", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert (error["code"], error["param"]) == ("model_not_found", "model")
    assert "acme/mock" in error["message"] and "any-model" in error["message"]


@pytest.mark.parametrize("model", ["any-model", None])
def test_any_model_or_none_is_routed_and_named(upstream, model) -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    if model is not None:
        body["model"] = model
    with TestClient(_router_app(upstream)) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.json()["echo_model"] == "acme/mock"  # the chosen worker's model id


def test_models_are_listed_by_the_router(upstream) -> None:
    with TestClient(_router_app(upstream)) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert [m["id"] for m in resp.json()["data"]] == ["any-model", "acme/mock"]


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
        resp = client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 503


def test_healthz_reports_worker_count(upstream) -> None:
    with TestClient(_router_app(upstream)) as client:
        body = client.get("/healthz").json()
    assert body == {"status": "ok", "healthy_workers": 1}


# --- model-capability caps (end-to-end through the proxy) -----------------------


def _wav30() -> str:
    """30 seconds of 16 kHz mono PCM, base64 — one Kari audio item."""
    data = b"\0" * (30 * 32000)
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data" + struct.pack("<I", len(data))
    )
    return base64.b64encode(header + data).decode()


def _audio_request(clips: int, **extra) -> dict:
    part = {"type": "input_audio", "input_audio": {"data": _wav30(), "format": "wav"}}
    return {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "transcribe"}, *([part] * clips)
        ]}],
        **extra,
    }


def _forwarded_audio(body: dict) -> list:
    parts = body["messages"][0]["content"]
    return [p for p in parts if p.get("type") == "input_audio"]


def test_caps_are_off_unless_configured(upstream) -> None:
    _MockVLLM.last_body = None
    with TestClient(_router_app(upstream)) as client:      # no caps in the NodeSpec
        resp = client.post("/v1/chat/completions", json=_audio_request(6))
    assert resp.status_code == 200
    assert len(_forwarded_audio(_MockVLLM.last_body)) == 6   # nothing trimmed
    assert "max_tokens" not in _MockVLLM.last_body           # nothing clamped
    assert "x-oumigo-audio-trimmed" not in resp.headers


def test_audio_cap_trims_the_forwarded_body_and_reports_it(upstream) -> None:
    _MockVLLM.last_body = None
    app = _router_app(upstream, max_audio_seconds=60.0)     # 2 clips' worth
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=_audio_request(6))

    assert resp.status_code == 200
    assert len(_forwarded_audio(_MockVLLM.last_body)) == 2
    note = _MockVLLM.last_body["messages"][0]["content"][0]
    assert note["type"] == "text" and note["text"].startswith("[OumiGo]")

    assert resp.headers["x-oumigo-audio-trimmed"] == "true"
    assert resp.headers["x-oumigo-audio-limit-seconds"] == "60"
    assert resp.headers["x-oumigo-audio-clips-dropped"] == "4/6"


def test_output_clamp_bounds_the_forwarded_body(upstream) -> None:
    _MockVLLM.last_body = None
    app = _router_app(upstream, max_output_tokens=4096)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=_audio_request(1, max_tokens=100000))
    assert _MockVLLM.last_body["max_tokens"] == 4096


def test_output_clamp_applies_when_the_client_named_no_limit(upstream) -> None:
    _MockVLLM.last_body = None
    app = _router_app(upstream, max_output_tokens=4096)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=_audio_request(1))
    assert _MockVLLM.last_body["max_tokens"] == 4096


def test_caps_do_not_touch_the_completions_endpoint(upstream) -> None:
    _MockVLLM.last_body = None
    app = _router_app(upstream, max_output_tokens=4096)
    with TestClient(app) as client:
        client.post("/v1/completions", json={"prompt": "hi"})
    assert "max_tokens" not in _MockVLLM.last_body


def test_agent_defaults_are_served_to_clients(upstream) -> None:
    app = _router_app(upstream)   # no agent: block configured
    with TestClient(app) as client:
        assert client.get("/agent-defaults").json() == {
            "turn_timeout": None, "stall_timeout": None,
        }

    host, port = upstream
    reg = Registry()
    _register(reg, "w1", host, NodeState.SERVING)
    app = create_router_app(reg, NodeSpec(model="acme/mock", port=port),
                            agent_defaults={"turn_timeout": 90.0, "stall_timeout": 45.0})
    with TestClient(app) as client:
        assert client.get("/agent-defaults").json()["turn_timeout"] == 90.0


# --- client auth (API keys) ----------------------------------------------------


class _HeaderEcho(_MockVLLM):
    seen_authorization: str | None = "unset"

    def do_POST(self):  # noqa: N802
        _HeaderEcho.seen_authorization = self.headers.get("Authorization")
        super().do_POST()


@pytest.fixture
def header_upstream():
    srv = HTTPServer(("127.0.0.1", 0), _HeaderEcho)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address
    srv.shutdown()


def _authed_router(upstream_addr):
    from oumigo.service.manager.credentials import AuthStore

    host, port = upstream_addr
    reg = Registry()
    _register(reg, "w1", host, NodeState.SERVING)
    store = AuthStore()
    app = create_router_app(reg, NodeSpec(model="acme/mock", port=port),
                            auth_store=store, client_auth=True)
    return app, store


_CHAT = {"messages": [{"role": "user", "content": "hi"}]}


def test_client_auth_refuses_missing_wrong_and_revoked_keys(upstream) -> None:
    app, store = _authed_router(upstream)
    key_id, key = store.create_api_key("t")
    with TestClient(app) as client:
        for headers in ({}, {"Authorization": "Bearer oumigo-key-wrong"},
                        {"Authorization": key}):   # no Bearer scheme
            resp = client.post("/v1/chat/completions", json=_CHAT, headers=headers)
            assert resp.status_code == 401
            assert resp.headers["www-authenticate"] == "Bearer"
        good = {"Authorization": f"Bearer {key}"}
        assert client.post("/v1/chat/completions", json=_CHAT, headers=good).status_code == 200
        assert client.get("/v1/models", headers=good).status_code == 200
        store.revoke_api_key(key_id)
        assert client.post("/v1/chat/completions", json=_CHAT, headers=good).status_code == 401


def test_client_auth_covers_metadata_but_not_health(upstream) -> None:
    app, _ = _authed_router(upstream)
    with TestClient(app) as client:
        for path in ("/v1/models", "/agent-defaults", "/list"):
            assert client.get(path).status_code == 401, path
        assert client.get("/healthz").status_code == 200
        assert client.get("/version").status_code == 200


def test_client_key_is_not_forwarded_to_direct_workers(header_upstream) -> None:
    app, store = _authed_router(header_upstream)
    _, key = store.create_api_key()
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=_CHAT,
                           headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 200
    assert _HeaderEcho.seen_authorization is None


def test_client_auth_off_keeps_the_open_lan_router(upstream) -> None:
    with TestClient(_router_app(upstream)) as client:
        assert client.post("/v1/chat/completions", json=_CHAT).status_code == 200


def test_client_auth_needs_a_store() -> None:
    with pytest.raises(ValueError):
        create_router_app(Registry(), NodeSpec(model="m", port=1), client_auth=True)
