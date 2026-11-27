"""The web console: sign-in, Create Admin Account, browser protections and Close."""

from __future__ import annotations

import re
import stat
import time

import httpx
import pytest
from starlette.testclient import TestClient

from oumigo.common.env import read_env_file, set_env_values
from oumigo.service.manager.admin_account import (
    hash_password,
    is_password_hash,
    login_problems,
    password_problems,
    verify_password,
)
from oumigo.service.manager.credentials import AuthStore
from oumigo.service.manager.dashboard.security import (
    COOKIE_NAME,
    IDLE_LIMIT_S,
    Sessions,
    is_loopback,
)
from oumigo.service.manager.dashboard.server import create_dashboard_app

NODE = "0123456789abcdef"
PASSWORD = "Correct-horse9"
LOCAL = "http://127.0.0.1:7080"


def _workers(state: str = "serving") -> dict:
    return {"workers": [{
        "node_id": NODE, "name": "Worker#1", "seq": 1, "address": "10.0.0.5", "port": 7001,
        "capabilities": {"gpu": "NVIDIA L4"}, "backend": "vllm", "transport": "direct",
        "state": state, "profile": "mock-model@8k", "task": "chat",
        "last_seen": time.time() - 3, "closing": False}]}


def _control(state: str = "serving") -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/workers":
            return httpx.Response(200, json=_workers(state))
        if request.url.path == "/metrics/latest":
            return httpx.Response(200, json={"nodes": [
                {"node_id": NODE, "metrics": {"vllm:start_timestamp": 1.8e9}}]})
        return httpx.Response(200, json={"points": []})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def console(tmp_path, monkeypatch):
    """A console bound to `bind` with the admin account from `account` (login, password)."""
    monkeypatch.delenv("MG_ADMIN_USER", raising=False)
    monkeypatch.delenv("MG_ADMIN_PASSWORD_HASH", raising=False)
    clients = []

    def make(bind: str = "127.0.0.1", account: tuple[str, str] | None = None,
             state: str = "serving") -> tuple[TestClient, AuthStore]:
        if account is not None:
            monkeypatch.setenv("MG_ADMIN_USER", account[0])
            monkeypatch.setenv("MG_ADMIN_PASSWORD_HASH", hash_password(account[1]))
        app = create_dashboard_app("http://127.0.0.1:1", state_dir=tmp_path,
                                   env_file=tmp_path / ".env", bind_host=bind)
        client = TestClient(app, base_url=LOCAL)
        client.__enter__()
        clients.append(client)
        app.state.client = _control(state)
        return client, AuthStore.at(tmp_path)

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf-token" content="([^"]*)"', html) or \
        re.search(r'name="csrf_token" value="([^"]*)"', html)
    assert match, html
    return match.group(1)


def _sign_in(client: TestClient, login: str, password: str) -> httpx.Response:
    token = _csrf(client.get("/signin").text)
    return client.post("/signin", data={"csrf_token": token, "login": login,
                                        "password": password}, follow_redirects=False)


# --- account rules ----------------------------------------------------------------


def test_password_hash_roundtrip_and_format() -> None:
    encoded = hash_password(PASSWORD)
    assert is_password_hash(encoded) and re.fullmatch(r"scrypt:[0-9a-f:]+", encoded)
    assert verify_password(PASSWORD, encoded)
    assert not verify_password(PASSWORD + "x", encoded)
    assert not verify_password(PASSWORD, "not-a-hash")


def test_password_and_login_rules() -> None:
    assert password_problems(PASSWORD) == []
    assert len(password_problems("short")) >= 3
    assert any("space" in p for p in password_problems("Has space 1!xx"))
    assert any("printable ASCII" in p for p in password_problems("Pässword-123"))
    assert any("symbol" in p for p in password_problems("NoSymbol1234"))
    assert login_problems("yyw") == [] and login_problems("Admin User") != []


def test_set_env_values_keeps_other_lines(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("# fleet\nMG_JOIN_TOKEN=abc\nMG_ADMIN_USER=\n"
                   "MG_ADMIN_USER=dup\n")
    set_env_values(env, {"MG_ADMIN_USER": "yyw", "MG_ADMIN_PASSWORD_HASH": "scrypt:1"})
    assert env.read_text() == ("# fleet\nMG_JOIN_TOKEN=abc\nMG_ADMIN_USER=yyw\n"
                               "MG_ADMIN_PASSWORD_HASH=scrypt:1\n")
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    set_env_values(tmp_path / "new.env", {"A": "1"})
    assert read_env_file(tmp_path / "new.env") == {"A": "1"}


def test_sessions_are_signed_and_expire() -> None:
    sessions = Sessions.from_secret(b"s" * 32)
    session = sessions.new("yyw", "fp")
    cookie = sessions.encode(session)
    assert sessions.decode(cookie).login == "yyw"
    assert sessions.decode(cookie[:-1] + ("0" if cookie[-1] != "0" else "1")) is None
    assert Sessions.from_secret(b"t" * 32).decode(cookie) is None
    session.last -= IDLE_LIMIT_S + 1
    assert sessions.decode(sessions.encode(session)) is None
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1")
    assert not is_loopback("0.0.0.0")


# --- sign-in off ------------------------------------------------------------------


def test_without_an_admin_account_the_console_is_view_only(console) -> None:
    client, store = console()
    page = client.get("/workers")
    assert page.status_code == 200
    assert "Create Admin Account" in page.text and "Close needs a signed-in admin" in page.text
    data = client.get("/api/workers").json()
    assert not data["control"] and not data["workers"][0]["closable"]
    assert data["workers"][0]["start"] == "2027-01-15 08:00:00"
    resp = client.post(f"/api/workers/{NODE}/close", headers={"X-CSRF-Token": _csrf(page.text)})
    assert resp.status_code == 403
    assert store.close_requests() == {}


def test_close_needs_the_csrf_token_and_same_origin(console) -> None:
    client, store = console(account=("yyw", PASSWORD))
    _sign_in(client, "yyw", PASSWORD)
    token = _csrf(client.get("/workers").text)
    assert client.post(f"/api/workers/{NODE}/close").status_code == 403
    assert client.post(f"/api/workers/{NODE}/close",
                       headers={"X-CSRF-Token": "forged"}).status_code == 403
    assert client.post(f"/api/workers/{NODE}/close",
                       headers={"X-CSRF-Token": token,
                                "Origin": "https://evil.example"}).status_code == 403
    assert store.close_requests() == {}


def test_loopback_bind_answers_only_localhost(console) -> None:
    client, _ = console()
    assert client.get("/workers", headers={"Host": "evil.example:7080"}).status_code == 400
    assert client.get("/healthz", headers={"Host": "localhost:7080"}).status_code == 200


def test_the_console_refuses_a_non_loopback_bind(tmp_path) -> None:
    for host in ("0.0.0.0", "192.168.1.10"):
        with pytest.raises(ValueError, match="loopback only"):
            create_dashboard_app("http://127.0.0.1:1", state_dir=tmp_path, bind_host=host)


def test_close_refuses_a_stopped_or_unknown_worker(console) -> None:
    client, store = console(account=("yyw", PASSWORD), state="stopped")
    _sign_in(client, "yyw", PASSWORD)
    token = _csrf(client.get("/workers").text)
    assert client.post(f"/api/workers/{NODE}/close",
                       headers={"X-CSRF-Token": token}).status_code == 409
    assert client.post("/api/workers/ffffffffffffffff/close",
                       headers={"X-CSRF-Token": token}).status_code == 404
    assert store.close_requests() == {}


# --- sign-in on -------------------------------------------------------------------


def test_sign_in_guards_pages_and_names_the_closer(console) -> None:
    client, store = console(account=("yyw", PASSWORD))
    first = client.get("/workers", follow_redirects=False)
    assert first.status_code == 303 and first.headers["location"] == "/signin"
    assert client.get("/api/workers").status_code == 401
    assert client.get("/admin-account").status_code == 404

    assert _sign_in(client, "yyw", "wrong-Password1").status_code == 401
    assert _sign_in(client, "yyw", PASSWORD).status_code == 303
    page = client.get("/workers")
    assert "signed in as <b>yyw</b>" in page.text
    resp = client.post(f"/api/workers/{NODE}/close", headers={"X-CSRF-Token": _csrf(page.text)})
    assert resp.status_code == 200
    assert store.close_requests()[NODE].requested_by == "yyw"

    client.post("/signout", data={"csrf_token": _csrf(page.text)})
    assert client.get("/api/workers").status_code == 401


def test_sign_in_locks_after_five_wrong_passwords(console) -> None:
    client, _ = console(account=("yyw", PASSWORD))
    for _ in range(5):
        assert _sign_in(client, "yyw", "wrong-Password1").status_code == 401
    assert _sign_in(client, "yyw", PASSWORD).status_code == 429


# --- Create Admin Account ---------------------------------------------------------


def test_create_admin_account_writes_env_and_turns_sign_in_on(console, tmp_path) -> None:
    (tmp_path / ".env").write_text("MG_JOIN_TOKEN=abc\n")
    client, _ = console()
    form = client.get("/admin-account")
    assert 'name="password2"' in form.text
    token = _csrf(form.text)

    bad = client.post("/admin-account", data={"csrf_token": token, "login": "yyw",
                                              "password": PASSWORD, "password2": "other"})
    assert bad.status_code == 400 and "The two passwords differ" in bad.text
    weak = client.post("/admin-account", data={"csrf_token": token, "login": "Bad Name",
                                               "password": "weak", "password2": "weak"})
    assert weak.status_code == 400 and "login name" in weak.text

    ok = client.post("/admin-account", data={"csrf_token": token, "login": "yyw",
                                             "password": PASSWORD, "password2": PASSWORD},
                     follow_redirects=False)
    assert ok.status_code == 303 and ok.headers["location"] == "/signin"
    written = read_env_file(tmp_path / ".env")
    assert written["MG_JOIN_TOKEN"] == "abc"
    assert written["MG_ADMIN_USER"] == "yyw"
    assert verify_password(PASSWORD, written["MG_ADMIN_PASSWORD_HASH"])

    # Sign-in is on at once: the link is gone, pages need sign-in, the page is 404.
    assert client.get("/workers", follow_redirects=False).status_code == 303
    assert client.get("/admin-account").status_code == 404
    assert _sign_in(client, "yyw", PASSWORD).status_code == 303
    page = client.get("/workers")
    assert "Create Admin Account" not in page.text
    assert client.get("/api/workers").json()["workers"][0]["closable"]
    resp = client.post(f"/api/workers/{NODE}/close", headers={"X-CSRF-Token": _csrf(page.text)})
    assert resp.status_code == 200


def test_create_admin_account_refuses_a_key_from_the_environment(console, monkeypatch) -> None:
    monkeypatch.setenv("MG_ADMIN_USER", "")  # e.g. systemd Environment=, not the file
    client, _ = console()
    page = client.get("/admin-account")
    assert "is set in the manager" in page.text
    assert 'name="password2"' not in page.text


def test_session_cookie_is_http_only_and_strict(console) -> None:
    client, _ = console()
    resp = client.get("/workers")
    cookie = resp.headers["set-cookie"]
    assert cookie.startswith(COOKIE_NAME) and "HttpOnly" in cookie
    assert "SameSite=strict" in cookie or "samesite=strict" in cookie.lower()


def test_gpu_view_has_four_ranges(console) -> None:
    client, _ = console()
    page = client.get("/gpu").text
    for title in ("Past 60 Minutes", "Past 48 Hours", "Past 10 Days", "Past 45 Days"):
        assert title in page
    assert len(client.get("/api/sheets/gpu_util").json()["labels"]) == 720  # 60 min of 5 s
    for key, buckets in (("48h", 576), ("10d", 480), ("45d", 540)):
        assert len(client.get(f"/api/sheets/gpu_util?range={key}").json()["labels"]) == buckets
    assert client.get("/api/sheets/gpu_util?range=7d").status_code == 400


def test_token_views_and_api(console) -> None:
    client, _ = console()
    rate = client.get("/tokens").text
    assert "Token Generation" in rate and "Past 45 Days" in rate and "MT / min" in rate
    cumulative = client.get("/tokens/cumulative").text
    assert "Token Generation (Cumulative)" in cumulative and "running total" in cumulative
    for mode in ("rate", "cumulative"):
        for key, buckets in (("60m", 720), ("48h", 576), ("10d", 480), ("45d", 540)):
            out = client.get(f"/api/sheets/tokens?range={key}&mode={mode}").json()
            assert len(out["labels"]) == buckets
    assert client.get("/api/sheets/tokens?mode=total").status_code == 400
