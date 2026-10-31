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


def _profile(pid: str, model: str, **caps) -> Profile:
    return Profile(id=pid, model=model, caps=Caps(**caps))


def _fleet(addrs, **router_kwargs):
    """Two serving workers: SMALL (output cap 100) and BIG (no cap); IDLE has none."""
    catalog = FleetCatalog({p.id: p for p in (
        _profile(SMALL, "acme/small-model", max_output_tokens=100),
        _profile(BIG, "acme/big-model"),
        _profile(IDLE, "acme/idle-model"),
    )}, SMALL)
    reg = Registry()
    for node_id, (host, port), pid in (("small", addrs[0], SMALL), ("big", addrs[1], BIG)):
        reg.register(node_id, host, NodeState.SERVING.value, 0, {}, port=port,
                     model=catalog.get(pid).model, profile=pid)
        reg.heartbeat(node_id, NodeState.SERVING.value)
    spec = catalog.default_profile.node_spec(7001)
    return create_router_app(reg, spec, catalog=catalog, **router_kwargs)


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
    assert data[1]["owned_by"] == "acme"
    assert data[1]["oumigo"] == {"weights": "acme/big-model", "max_model_len": None, "serving": 1}


def test_agent_defaults_per_model(upstreams) -> None:
    app = _fleet(upstreams, agent_defaults={"turn_timeout": 90.0, "stall_timeout": None},
                 fleet_agent_defaults={"turn_timeout": 60.0, "stall_timeout": 30.0})
    with TestClient(app) as client:
        assert client.get("/agent-defaults").json()["turn_timeout"] == 90.0  # the default's
        # BIG has no `agent:` of its own -> the manager's block.
        assert client.get(f"/agent-defaults?model={BIG}").json() == {
            "turn_timeout": 60.0, "stall_timeout": 30.0}


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
