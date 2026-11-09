"""Embedding models in a fleet: the router keeps chat and embed workers apart, serves
`/v1/embeddings`, and turns `input_type` into the model's own prompt."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from starlette.testclient import TestClient

from oumigo.config.catalog import EmbedSettings, FleetCatalog, Profile
from oumigo.protocol.states import NodeState
from oumigo.service.manager.control.registry import Registry
from oumigo.service.manager.router.server import apply_input_type, create_router_app

CHAT, EMB, EMB2 = "chat-model@32k", "embedder@8k", "other-embedder@8k"
PROMPTS = {"query": "q: ", "document": ""}


class _Echo(BaseHTTPRequestHandler):
    """Answers with which worker got the request, on which path, with what body."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        payload = json.dumps({"worker": self.server.name, "path": self.path, "body": body}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def upstreams():
    servers = []
    for name in ("chat", "embed"):
        srv = HTTPServer(("127.0.0.1", 0), _Echo)
        srv.name = name
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
    yield [srv.server_address for srv in servers]
    for srv in servers:
        srv.shutdown()


def _fleet(addrs, catalog: bool = True):
    """A chat worker and an embed worker, both SERVING; EMB2 is allowed but not up."""
    profiles = FleetCatalog({p.id: p for p in (
        Profile(id=CHAT, model="acme/chat-model"),
        Profile(id=EMB, model="acme/embedder", task="embed",
                embed=EmbedSettings(dimensions=768, prompts=PROMPTS)),
        Profile(id=EMB2, model="acme/other-embedder", task="embed"),
    )}, CHAT)
    reg = Registry()
    for node_id, (host, port), pid in (("chat", addrs[0], CHAT), ("embed", addrs[1], EMB)):
        reg.register(node_id, host, NodeState.SERVING.value, 0, {}, port=port,
                     model=profiles.get(pid).model, profile=pid,
                     task=profiles.get(pid).task)
        reg.heartbeat(node_id, NodeState.SERVING.value)
    if not catalog:  # an open fleet: tasks only as the workers report them
        return create_router_app(reg, None)
    return create_router_app(reg, profiles.default_profile.node_spec(7001), catalog=profiles,
                             queue_timeout=0.05)


# --- routing by task ---------------------------------------------------------------------


def test_any_model_chat_never_reaches_an_embed_worker(upstreams) -> None:
    with TestClient(_fleet(upstreams)) as client:
        for path in ("/v1/chat/completions", "/v1/completions"):
            for model in ("any-model", None):
                for _ in range(10):
                    body = {"messages": []} if model is None else {"model": model, "messages": []}
                    resp = client.post(path, json=body)
                    assert resp.status_code == 200 and resp.json()["worker"] == "chat"


def test_embeddings_reach_the_named_embed_worker(upstreams) -> None:
    with TestClient(_fleet(upstreams)) as client:
        resp = client.post("/v1/embeddings", json={"model": EMB, "input": ["a", "b"]})
    assert resp.status_code == 200
    assert (resp.json()["worker"], resp.json()["path"]) == ("embed", "/v1/embeddings")
    assert resp.json()["body"] == {"model": EMB, "input": ["a", "b"]}


@pytest.mark.parametrize("path,body,code,message", [
    ("/v1/chat/completions", {"model": EMB, "messages": []}, "wrong_task",
     f"{EMB} is an embedding model; use /v1/embeddings."),
    ("/v1/completions", {"model": EMB, "prompt": "x"}, "wrong_task",
     f"{EMB} is an embedding model; use /v1/embeddings."),
    ("/v1/embeddings", {"model": CHAT, "input": "x"}, "wrong_task",
     f"{CHAT} is not an embedding model."),
    ("/v1/embeddings", {"model": "any-model", "input": "x"}, "model_required",
     f"Name an embedding model; available: {EMB}, {EMB2}."),
    ("/v1/embeddings", {"input": "x"}, "model_required",
     f"Name an embedding model; available: {EMB}, {EMB2}."),
])
def test_requests_for_the_wrong_task_are_refused(upstreams, path, body, code, message) -> None:
    with TestClient(_fleet(upstreams)) as client:
        resp = client.post(path, json=body)
    assert resp.status_code == 400
    assert resp.json()["error"] == {"message": message, "type": "invalid_request_error",
                                    "param": "model", "code": code}


def test_an_allowed_embed_model_without_a_worker_is_unavailable(upstreams) -> None:
    with TestClient(_fleet(upstreams)) as client:
        resp = client.post("/v1/embeddings", json={"model": EMB2, "input": "x"})
        unknown = client.post("/v1/embeddings", json={"model": "nope@8k", "input": "x"})
    assert resp.status_code == 503 and resp.json()["error"]["code"] == "model_unavailable"
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "model_not_found"


def test_an_open_fleet_routes_by_the_task_workers_report(upstreams) -> None:
    with TestClient(_fleet(upstreams, catalog=False)) as client:
        assert client.post("/v1/embeddings",
                           json={"model": EMB, "input": "x"}).json()["worker"] == "embed"
        assert client.post("/v1/chat/completions",
                           json={"model": EMB, "messages": []}).json()["error"]["code"] == "wrong_task"
        for _ in range(10):
            assert client.post("/v1/chat/completions",
                               json={"model": "any-model", "messages": []}).json()["worker"] == "chat"


def test_any_model_defaults_ignore_embed_models(upstreams) -> None:
    profiles = FleetCatalog({p.id: p for p in (
        Profile(id=CHAT, model="acme/chat-model", agent={"turn_timeout": 60.0}),
        Profile(id=EMB, model="acme/embedder", task="embed",
                agent={"turn_timeout": 9999.0}, caps={"max_queue_seconds": 9999.0}),
    )}, CHAT)
    reg = Registry()
    for node_id, (host, port), pid in (("chat", upstreams[0], CHAT), ("embed", upstreams[1], EMB)):
        reg.register(node_id, host, NodeState.SERVING.value, 0, {}, port=port, profile=pid,
                     task=profiles.get(pid).task)
        reg.heartbeat(node_id, NodeState.SERVING.value)
    app = create_router_app(reg, profiles.default_profile.node_spec(7001), catalog=profiles)
    with TestClient(app) as client:
        assert client.get("/agent-defaults").json()["turn_timeout"] == 60.0


# --- input_type --------------------------------------------------------------------------


def test_input_type_becomes_the_models_prompt(upstreams) -> None:
    with TestClient(_fleet(upstreams)) as client:
        query = client.post("/v1/embeddings",
                            json={"model": EMB, "input": "東京", "input_type": "query"}).json()
        docs = client.post("/v1/embeddings",
                           json={"model": EMB, "input": ["a", "b"], "input_type": "document",
                                 "encoding_format": "base64"}).json()
    assert query["body"] == {"model": EMB, "input": "q: 東京"}
    assert docs["body"] == {"model": EMB, "input": ["a", "b"], "encoding_format": "base64"}


@pytest.mark.parametrize("model,body,message", [
    (EMB, {"input": "x", "input_type": "passage"},
     f"{EMB} has no input_type 'passage'; valid: document, query."),
    (EMB, {"input": [1, 2, 3], "input_type": "query"}, "input_type needs text input, not token ids."),
    (EMB2, {"input": "x", "input_type": "query"}, f"{EMB2} takes no input_type."),
])
def test_unusable_input_types_are_refused(upstreams, model, body, message) -> None:
    with TestClient(_fleet(upstreams)) as client:
        resp = client.post("/v1/embeddings", json={"model": model, **body})
    assert resp.status_code == 400
    assert resp.json()["error"] == {"message": message, "type": "invalid_request_error",
                                    "param": "input_type", "code": "invalid_input_type"}


def test_apply_input_type_leaves_a_request_without_one_alone() -> None:
    payload = {"model": EMB, "input": [[1, 2]]}
    apply_input_type(payload, PROMPTS)
    assert payload == {"model": EMB, "input": [[1, 2]]}


# --- /v1/models ---------------------------------------------------------------------------


def test_models_say_what_each_model_is_for(upstreams) -> None:
    with TestClient(_fleet(upstreams)) as client:
        data = {m["id"]: m.get("oumigo") for m in client.get("/v1/models").json()["data"]}
    assert data[CHAT]["task"] == "chat" and "dimensions" not in data[CHAT]
    assert data[EMB] == {"weights": "acme/embedder", "task": "embed", "max_model_len": None,
                         "dimensions": 768, "input_types": ["document", "query"], "serving": 1}
