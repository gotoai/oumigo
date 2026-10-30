"""Worker allowlist and failed-auth throttling on the worker-facing endpoints."""

from __future__ import annotations

from starlette.testclient import TestClient

from oumigo.service.manager.access import FailureLimiter, WorkerAllowlist
from oumigo.service.manager.control.registry import Registry
from oumigo.service.manager.control.server import create_app


def test_allowlist_off_allows_everyone() -> None:
    assert WorkerAllowlist(None).allows("203.0.113.7")


def test_allowlist_matches_addresses_and_ranges() -> None:
    wl = WorkerAllowlist(["203.0.113.7", "198.51.100.0/24", "2001:db8::/32"])
    assert wl.allows("203.0.113.7")
    assert wl.allows("198.51.100.200")
    assert wl.allows("::ffff:198.51.100.9")  # IPv4-mapped IPv6, as a dual-stack bind reports it
    assert wl.allows("2001:db8::1")
    assert not wl.allows("203.0.113.8")
    assert not wl.allows("not-an-ip") and not wl.allows(None)
    assert not WorkerAllowlist([]).allows("127.0.0.1")  # on with an empty list = nobody


def test_failure_limiter_blocks_after_threshold() -> None:
    limiter = FailureLimiter(max_failures=3, window_s=60)
    for _ in range(2):
        limiter.record("1.2.3.4")
    assert not limiter.blocked("1.2.3.4")
    limiter.record("1.2.3.4")
    assert limiter.blocked("1.2.3.4") and not limiter.blocked("5.6.7.8")


def _client(allowlist: list[str] | None, source: str) -> TestClient:
    app = create_app(Registry(), "fleet-secret", worker_allowlist=allowlist)
    return TestClient(app, client=(source, 50000))


def test_endpoints_enforce_the_allowlist() -> None:
    body = {"node_id": "n1", "address": "x", "vllm_port": 7001}
    auth = {"Authorization": "Bearer fleet-secret"}
    outside = _client(["10.0.0.0/8"], "203.0.113.7")
    assert outside.post("/register", json=body, headers=auth).status_code == 403
    assert outside.get("/spec", headers=auth).status_code == 403
    assert outside.get("/healthz").status_code == 200  # read endpoints aren't worker-facing
    inside = _client(["10.0.0.0/8"], "10.1.2.3")
    assert inside.post("/register", json=body, headers=auth).status_code == 200


def test_repeated_bad_tokens_get_throttled() -> None:
    client = _client(None, "203.0.113.7")
    bad = {"Authorization": "Bearer wrong"}
    codes = [client.get("/spec", headers=bad).status_code for _ in range(31)]
    assert codes[:30] == [401] * 30 and codes[30] == 429
    # Even the right token is turned away while throttled.
    assert client.get("/spec", headers={"Authorization": "Bearer fleet-secret"}).status_code == 429
