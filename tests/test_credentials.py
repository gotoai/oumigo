"""Worker credentials: minted join tokens, node tokens, revocation, and endpoint auth."""

from __future__ import annotations

import time

from starlette.testclient import TestClient

from oumigo.service.manager.control.registry import Registry
from oumigo.service.manager.control.server import create_app
from oumigo.service.manager.credentials import Authenticator, AuthStore, bearer


def _reg_body(node_id: str, transport: str = "direct") -> dict:
    return {"node_id": node_id, "address": "10.0.0.5", "vllm_port": 7001, "transport": transport}


def _hb_body(node_id: str) -> dict:
    return {"node_id": node_id, "node_state": "serving", "run_state": "idle"}


def _auth(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


# --- store ----------------------------------------------------------------------


def test_single_use_join_token_is_spent_once() -> None:
    store = AuthStore()
    _, token = store.create_join_token(uses=1)
    assert store.check_join_token(token, consume=False)  # peeking doesn't spend it
    assert store.check_join_token(token, consume=True)
    assert not store.check_join_token(token, consume=True)


def test_unlimited_expired_and_revoked_join_tokens() -> None:
    store = AuthStore()
    _, unlimited = store.create_join_token()
    assert all(store.check_join_token(unlimited, consume=True) for _ in range(3))
    _, expired = store.create_join_token(ttl_s=0.01)
    time.sleep(0.02)
    assert not store.check_join_token(expired, consume=True)
    token_id, revoked = store.create_join_token()
    assert store.revoke_join_token(token_id)
    assert not store.check_join_token(revoked, consume=False)
    assert not store.check_join_token("oumigo-join-made-up", consume=False)


def test_node_token_is_bound_and_tamper_evident() -> None:
    store = AuthStore()
    token = store.node_token("abc123")
    assert store.node_from_token(token) == "abc123"
    assert store.node_from_token(token[:-1] + ("0" if token[-1] != "0" else "1")) is None
    assert store.node_from_token(store.node_token("abc123").replace("abc123", "other1", 1)) is None
    assert store.vllm_api_key("abc123") != store.vllm_api_key("other1")


def test_secret_and_tokens_survive_reopen(tmp_path) -> None:
    first = AuthStore.at(tmp_path)
    node_token = first.node_token("n1")
    _, join = first.create_join_token(uses=2)
    second = AuthStore.at(tmp_path)  # e.g. a manager restart, or the CLI's own connection
    assert second.node_from_token(node_token) == "n1"
    assert second.check_join_token(join, consume=True)
    assert (tmp_path / "manager_auth.db").stat().st_mode & 0o777 == 0o600


def test_bearer_parsing() -> None:
    assert bearer("Bearer abc") == "abc"
    assert bearer("bearer  abc ") == "abc"
    assert bearer("Basic abc") is None and bearer(None) is None and bearer("Bearer ") is None


# --- endpoints ------------------------------------------------------------------


def _client(provider: str, reusable: str | None = None) -> tuple[TestClient, AuthStore]:
    store = AuthStore()
    auth = Authenticator(store, reusable, wan=provider == "CUSTOM")
    app = create_app(Registry(), reusable, auth=auth, provider_name=provider)
    return TestClient(app), store


def test_lan_without_token_stays_open_but_issues_node_tokens() -> None:
    client, store = _client("LAN")
    assert client.get("/spec").json()["provider"] == "LAN"
    resp = client.post("/register", json=_reg_body("n1"))
    assert resp.status_code == 200
    assert store.node_from_token(resp.json()["node_token"]) == "n1"
    assert client.post("/heartbeat", json=_hb_body("n1")).status_code == 200


def test_custom_is_enforced_even_without_a_reusable_token() -> None:
    client, store = _client("CUSTOM")
    assert client.get("/spec").status_code == 401
    assert client.post("/register", json=_reg_body("n1")).status_code == 401
    _, join = store.create_join_token(uses=1)
    assert client.get("/spec", headers=_auth(join)).status_code == 200  # not consumed
    resp = client.post("/register", json=_reg_body("n1"), headers=_auth(join))
    assert resp.status_code == 200
    node_token = resp.json()["node_token"]
    # The single-use join token is now spent: a second pod can't reuse it.
    assert client.post("/register", json=_reg_body("n2"), headers=_auth(join)).status_code == 401
    # The node token works for this node only.
    assert client.post("/heartbeat", json=_hb_body("n1"), headers=_auth(node_token)).status_code == 200
    assert client.post("/heartbeat", json=_hb_body("n2"), headers=_auth(node_token)).status_code == 401
    assert client.post("/heartbeat", json=_hb_body("n1"), headers=_auth(join)).status_code == 401
    # ...including re-registering itself (e.g. after a manager restart).
    assert client.post("/register", json=_reg_body("n1"), headers=_auth(node_token)).status_code == 200
    metrics = {"node_id": "n1", "points": []}
    assert client.post("/metrics", json=metrics, headers=_auth(node_token)).status_code == 200
    assert client.get("/status").json()["auth"] is True


def test_reusable_token_still_works_for_everything() -> None:
    client, _ = _client("CUSTOM", reusable="fleet-secret")
    assert client.post("/register", json=_reg_body("n1"), headers=_auth("fleet-secret")).status_code == 200
    assert client.post("/heartbeat", json=_hb_body("n1"), headers=_auth("fleet-secret")).status_code == 200
    assert client.post("/heartbeat", json=_hb_body("n1"), headers=_auth("wrong")).status_code == 401


def test_revoked_node_is_refused_with_403() -> None:
    client, store = _client("CUSTOM", reusable="fleet-secret")
    node_token = client.post(
        "/register", json=_reg_body("n1"), headers=_auth("fleet-secret")
    ).json()["node_token"]
    store.revoke_node("n1")
    assert client.post("/heartbeat", json=_hb_body("n1"), headers=_auth(node_token)).status_code == 403
    # Not even the reusable fleet token can bring a revoked node_id back.
    assert client.post("/register", json=_reg_body("n1"), headers=_auth("fleet-secret")).status_code == 403
    assert store.unrevoke_node("n1")
    assert client.post("/heartbeat", json=_hb_body("n1"), headers=_auth(node_token)).status_code == 200


def test_refused_tunnel_registration_does_not_spend_the_token() -> None:
    store = AuthStore()
    auth = Authenticator(store, None, wan=True)
    client = TestClient(create_app(Registry(), None, auth=auth, provider_name="CUSTOM",
                                   tunnel_enabled=False))
    _, join = store.create_join_token(uses=1)
    resp = client.post("/register", json=_reg_body("n1", "tunnel"), headers=_auth(join))
    assert resp.status_code == 400
    assert store.check_join_token(join, consume=False)  # still usable for a direct worker
