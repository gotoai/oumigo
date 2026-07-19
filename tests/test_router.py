"""Tests for the data-plane router: worker selection + forwarding passthrough."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from starlette.testclient import TestClient

from oumigo.config.spec import NodeSpec
from oumigo.manager.control.registry import Registry
from oumigo.manager.router.server import WorkerSelector, create_router_app
from oumigo.protocol.states import NodeState, RunState


def _register(reg: Registry, node_id: str, address: str, state: NodeState,
              run_state: RunState | None = None) -> None:
    reg.register(node_id, address, state.value, 0, {})
    reg.heartbeat(node_id, state.value, run_state.value if run_state else None)


# --- selection ------------------------------------------------------------------


def test_selector_only_picks_serving_workers() -> None:
    reg = Registry()
    _register(reg, "a", "10.0.0.1", NodeState.SERVING)
    _register(reg, "b", "10.0.0.2", NodeState.INITIALIZING)   # not ready
    _register(reg, "c", "10.0.0.3", NodeState.LOST)           # dead
    sel = WorkerSelector(reg)
    picked = {sel.pick().address for _ in range(6)}
    assert picked == {"10.0.0.1"}  # only the SERVING node is ever chosen
    assert sel.healthy_count() == 1


def test_selector_round_robin_cycles_evenly() -> None:
    reg = Registry()
    for i in range(3):
        _register(reg, f"n{i}", f"10.0.0.{i}", NodeState.SERVING)
    sel = WorkerSelector(reg, strategy="round_robin")
    seq = [sel.pick().address for _ in range(6)]
    assert seq == ["10.0.0.0", "10.0.0.1", "10.0.0.2", "10.0.0.0", "10.0.0.1", "10.0.0.2"]


def test_selector_least_loaded_prefers_idle() -> None:
    reg = Registry()
    _register(reg, "busy", "10.0.0.1", NodeState.SERVING, RunState.EXECUTING)
    _register(reg, "idle", "10.0.0.2", NodeState.SERVING, RunState.IDLE)
    sel = WorkerSelector(reg, strategy="least_loaded")
    assert {sel.pick().address for _ in range(4)} == {"10.0.0.2"}  # idle only


def test_selector_returns_none_when_no_healthy() -> None:
    assert WorkerSelector(Registry()).pick() is None


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


def test_forward_models(upstream) -> None:
    with TestClient(_router_app(upstream)) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "acme/mock"


def test_503_when_no_healthy_workers() -> None:
    app = create_router_app(Registry(), NodeSpec(model="acme/mock", port=9))
    resp = TestClient(app).post("/v1/chat/completions", json={"model": "acme/mock", "messages": []})
    assert resp.status_code == 503


def test_503_when_no_model_configured(upstream) -> None:
    reg = Registry()
    _register(reg, "w1", upstream[0], NodeState.SERVING)
    app = create_router_app(reg, node_spec=None)  # no vLLM port to route to
    resp = TestClient(app).get("/v1/models")
    assert resp.status_code == 503


def test_healthz_reports_worker_count(upstream) -> None:
    body = TestClient(_router_app(upstream)).get("/healthz").json()
    assert body == {"status": "ok", "healthy_workers": 1}
