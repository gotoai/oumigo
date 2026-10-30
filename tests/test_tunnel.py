"""End-to-end tunnel transport on loopback: a real manager (control plane + router), a
fake backend that requires the node's API key, and a real TunnelClient dialing out."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from oumigo.config.spec import NodeSpec
from oumigo.service.manager.control.registry import Registry
from oumigo.service.manager.control.server import create_app
from oumigo.service.manager.credentials import Authenticator, AuthStore
from oumigo.service.manager.router.server import create_router_app
from oumigo.service.manager.tunnel import TunnelRegistry
from oumigo.service.worker.tunnel import TunnelClient, tunnel_url

JOIN = "fleet-secret"
MODEL = "test/model"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _backend_app(api_key: str, events: dict) -> FastAPI:
    """Stands in for vLLM: requires the key, streams 5 tokens on request."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if request.headers.get("authorization") != f"Bearer {api_key}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        body = await request.json()
        events["models"].append(body.get("model"))
        delay = float(body.get("delay", 0.05))
        if not body.get("stream"):
            return {"choices": [{"message": {"content": "hello"}}], "big": "x" * body.get("pad", 0)}

        async def gen():
            try:
                for i in range(int(body.get("tokens", 5))):
                    yield f"data: {json.dumps({'i': i})}\n\n".encode()
                    await asyncio.sleep(delay)
                yield b"data: [DONE]\n\n"
            except asyncio.CancelledError:
                events["cancelled"] += 1
                raise

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


class Servers:
    """Serve ASGI apps on loopback from one background event loop."""

    def __init__(self, apps) -> None:
        self._servers = []
        for app, port in apps:
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                                   log_level="warning", lifespan="on"))
            server.install_signal_handlers = lambda: None
            self._servers.append(server)
        servers = list(self._servers)

        async def run_all() -> None:
            await asyncio.gather(*(s.serve() for s in servers))

        threading.Thread(target=lambda: asyncio.run(run_all()), daemon=True).start()
        deadline = time.time() + 10
        while not all(s.started for s in self._servers):
            assert time.time() < deadline, "servers did not start"
            time.sleep(0.02)

    def stop(self) -> None:
        for s in self._servers:
            s.should_exit = True


class Fleet:
    """A manager (control plane + router) and a fake backend; the tunnel client runs in
    its own thread, as on a real worker."""

    def __init__(self) -> None:
        self.control_port, self.router_port, self.backend_port = (_free_port() for _ in range(3))
        self.store = AuthStore()
        self.registry = Registry()
        self.tunnels = TunnelRegistry()
        auth = Authenticator(self.store, JOIN, wan=True)
        spec = NodeSpec(model=MODEL, max_concurrent_requests=3)
        self.control = create_app(self.registry, JOIN, node_spec=spec, auth=auth,
                                  provider_name="CUSTOM", tunnels=self.tunnels)
        self.router = create_router_app(self.registry, spec, tunnels=self.tunnels,
                                        auth_store=self.store, queue_timeout=5.0)
        self.events: dict = {"models": [], "cancelled": 0}
        self.control_url = f"http://127.0.0.1:{self.control_port}"
        self.router_url = f"http://127.0.0.1:{self.router_port}"
        self.client: TunnelClient | None = None

    def start(self) -> None:
        self.manager = Servers([(self.control, self.control_port), (self.router, self.router_port)])
        self.reg = self.register()
        self.backend = Servers([(_backend_app(self.reg["vllm_api_key"], self.events),
                                 self.backend_port)])
        self.heartbeat(self.reg["node_token"])
        self.client = TunnelClient(self.control_url, token=lambda: self.reg["node_token"],
                                   backend_port=self.backend_port,
                                   pool_size=self.reg["tunnel_pool_size"])
        self.client.start()
        self.wait_conns(self.reg["tunnel_pool_size"])

    def register(self, node_id: str = "node1") -> dict:
        body = {"node_id": node_id, "address": "10.9.9.9", "vllm_port": self.backend_port,
                "transport": "tunnel"}
        resp = httpx.post(f"{self.control_url}/register", json=body,
                          headers={"Authorization": f"Bearer {JOIN}"})
        resp.raise_for_status()
        return resp.json()

    def heartbeat(self, token: str, node_id: str = "node1", state: str = "serving") -> dict:
        resp = httpx.post(f"{self.control_url}/heartbeat",
                          json={"node_id": node_id, "node_state": state},
                          headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        return resp.json()

    def workers(self) -> dict[str, dict]:
        rows = httpx.get(f"{self.control_url}/workers").json()["workers"]
        return {r["node_id"]: r for r in rows}

    def wait_conns(self, n: int, node_id: str = "node1", timeout: float = 10) -> None:
        deadline = time.time() + timeout
        while self.workers().get(node_id, {}).get("tunnel_conns") != n:
            assert time.time() < deadline, f"tunnel never reached {n}: {self.workers()}"
            time.sleep(0.05)

    def stop(self) -> None:
        if self.client is not None:
            self.client.stop()
        self.manager.stop()
        self.backend.stop()


@pytest.fixture
def fleet() -> Iterator[Fleet]:
    f = Fleet()
    f.start()
    yield f
    f.stop()


def test_tunnel_url() -> None:
    assert tunnel_url("https://oumigo.example.net/") == "wss://oumigo.example.net/tunnel"
    assert tunnel_url("http://127.0.0.1:7014") == "ws://127.0.0.1:7014/tunnel"


def test_registration_hands_out_tunnel_settings(fleet: Fleet) -> None:
    assert fleet.reg["tunnel_pool_size"] == 4  # capacity 3 + 1 spare
    assert fleet.reg["vllm_api_key"].startswith("oumigo-vllm-")
    row = fleet.workers()["node1"]
    assert row["transport"] == "tunnel" and row["tunnel_idle"] == 4


def test_non_streaming_request_through_the_tunnel(fleet: Fleet) -> None:
    resp = httpx.post(f"{fleet.router_url}/v1/chat/completions",
                      json={"model": "whatever", "messages": [], "pad": 300_000},
                      headers={"Authorization": "Bearer client-secret"}, timeout=10)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hello"
    assert len(resp.json()["big"]) == 300_000  # larger than one WebSocket message
    assert fleet.events["models"] == [MODEL]  # router still rewrites the model


def test_streaming_requests_run_concurrently(fleet: Fleet) -> None:
    def stream() -> tuple[list[str], dict]:
        with httpx.stream("POST", f"{fleet.router_url}/v1/chat/completions",
                          json={"stream": True, "tokens": 5, "delay": 0.1}, timeout=10) as r:
            return [line for line in r.iter_lines() if line], dict(r.headers)

    results: list = []
    threads = [threading.Thread(target=lambda: results.append(stream())) for _ in range(3)]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - started
    assert len(results) == 3
    for lines, headers in results:
        assert lines[-1] == "data: [DONE]" and len(lines) == 6
        assert headers["x-accel-buffering"] == "no"
    assert elapsed < 1.2  # 3 x 0.5s streams overlapped, not serialized


def test_client_disconnect_cancels_the_backend_request(fleet: Fleet) -> None:
    with httpx.stream("POST", f"{fleet.router_url}/v1/chat/completions",
                      json={"stream": True, "tokens": 50, "delay": 0.1}, timeout=10) as r:
        for line in r.iter_lines():
            if line:
                break  # read one token, then hang up
    deadline = time.time() + 5
    while fleet.events["cancelled"] == 0:
        assert time.time() < deadline, "backend never saw the cancellation"
        time.sleep(0.05)
    fleet.wait_conns(fleet.reg["tunnel_pool_size"])  # the slot dialed a fresh connection


def test_worker_without_tunnel_connections_is_not_routable(fleet: Fleet) -> None:
    fleet.client.stop()
    fleet.client = None
    fleet.wait_conns(0)
    resp = httpx.post(f"{fleet.router_url}/v1/chat/completions", json={}, timeout=10)
    assert resp.status_code == 503


def test_tunnel_refuses_bad_credentials(fleet: Fleet) -> None:
    from websockets.exceptions import InvalidStatus
    from websockets.sync.client import connect

    for headers in ({}, {"Authorization": f"Bearer {JOIN}"},
                    {"Authorization": "Bearer oumigo-node.node1.forged"}):
        with pytest.raises(InvalidStatus) as exc:
            connect(f"ws://127.0.0.1:{fleet.control_port}/tunnel", additional_headers=headers)
        assert exc.value.response.status_code == 403


def test_revoking_a_node_closes_its_tunnels(fleet: Fleet) -> None:
    fleet.store.revoke_node("node1")
    deadline = time.time() + 6
    while "node1" in fleet.workers():
        assert time.time() < deadline, "revoked node was not dropped"
        time.sleep(0.1)
    assert fleet.tunnels.connected("node1") == 0
