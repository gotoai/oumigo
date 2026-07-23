"""Unit tests for the programmatic API (oumigo.api).

These exercise the non-spawning seams: discovery reuse, config injection, and the
worker record-matching predicate. Paths that spawn real children (a manager server,
a vLLM worker) are integration-level and out of scope here.
"""

from __future__ import annotations

import types

import pytest
import yaml

from oumigo import api
from oumigo.api import OumigoWorker, oumigo_get_or_create_manager


def test_get_manager_reuses_discovered_lan_manager(monkeypatch):
    """When a manager is advertising, return a remote handle and spawn nothing."""
    monkeypatch.setattr(api.discovery, "discover_manager", lambda _t: "http://10.0.0.5:7014")

    def _no_spawn(*_a, **_k):  # fail loudly if the spawn path is taken
        raise AssertionError("should not spawn when a manager is discovered")

    monkeypatch.setattr(api, "_spawn_child", _no_spawn)

    mgr = oumigo_get_or_create_manager(data_port=7012)

    assert mgr.owned is False
    assert mgr.control_url == "http://10.0.0.5:7014"
    assert mgr.data_url == "http://10.0.0.5:7012"
    assert mgr.stop() is None  # no-op for a non-owned manager


def test_write_manager_config_round_trips(tmp_path, monkeypatch):
    """The throwaway manager.yaml carries provider, data-plane bind, and model block."""
    path = api._write_manager_config(
        provider="LAN",
        data_host="0.0.0.0",
        data_port=7012,
        model={"name": "google/gemma-4-E2B", "port": 7001, "max_concurrent_requests": 1},
    )
    config = yaml.safe_load(open(path, encoding="utf-8"))
    assert config["provider"] == "LAN"
    assert config["data_plane"] == {"host": "0.0.0.0", "port": 7012}
    assert config["model"]["name"] == "google/gemma-4-E2B"
    assert config["model"]["max_concurrent_requests"] == 1


def test_config_file_seeds_settings_and_call_args_override(tmp_path, monkeypatch):
    """config_file values seed the manager; an explicit call arg overrides that file."""
    cfg_path = tmp_path / "manager.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "provider": "LAN",
                "data_plane": {"host": "1.2.3.4", "port": 8012},
                "model": {"name": "yaml/model", "port": 9001},
                "bearer_token": "from-yaml",
                "dashboard": {"enabled": False},  # extra key must survive into the spawn config
            }
        )
    )
    monkeypatch.setattr(api.discovery, "discover_manager", lambda _t: None)
    monkeypatch.setattr(api, "_spawn_child", lambda argv, env: types.SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(api, "_wait_manager_healthy", lambda child, manager, timeout: True)
    monkeypatch.setattr(api.atexit, "register", lambda *_a, **_k: None)

    captured: dict = {}
    monkeypatch.setattr(api, "_write_manager_config", lambda **kw: captured.update(kw) or "/tmp/x.yaml")

    mgr = oumigo_get_or_create_manager(
        config_file=str(cfg_path),
        data_port=7099,  # explicit -> overrides the file's 8012
    )

    assert mgr.owned is True
    assert mgr.token == "from-yaml"          # from the file
    assert captured["provider"] == "LAN"     # from the file
    assert captured["data_host"] == "1.2.3.4"  # from the file
    assert captured["data_port"] == 7099     # call arg wins over the file
    assert captured["model"]["name"] == "yaml/model"  # from the file
    assert captured["base"]["dashboard"] == {"enabled": False}  # extra key carried through


def test_missing_config_file_is_ignored(tmp_path, monkeypatch):
    """A non-existent config_file is ignored; defaults apply and nothing raises."""
    monkeypatch.setattr(api.discovery, "discover_manager", lambda _t: None)
    monkeypatch.setattr(api, "_spawn_child", lambda argv, env: types.SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(api, "_wait_manager_healthy", lambda child, manager, timeout: True)
    monkeypatch.setattr(api.atexit, "register", lambda *_a, **_k: None)

    captured: dict = {}
    monkeypatch.setattr(api, "_write_manager_config", lambda **kw: captured.update(kw) or "/tmp/x.yaml")

    mgr = oumigo_get_or_create_manager(config_file=str(tmp_path / "nope.yaml"))

    assert mgr.owned is True
    assert captured["provider"] == "LAN"       # built-in default
    assert captured["data_port"] == 7012       # built-in default
    assert captured["model"]["name"] == api.DEFAULT_MODEL["name"]


def _no_mdns(*_a, **_k):
    raise AssertionError("mDNS discovery should not run when a manager is available")


def _stub_worker_spawn(monkeypatch, captured):
    """Stub out the spawn + serving-wait so create_worker resolves the manager only."""
    monkeypatch.setattr(api.discovery, "get_lan_ip", lambda: "10.0.0.9")
    monkeypatch.setattr(api, "_worker_incarnations", lambda _url, _addr: {})
    monkeypatch.setattr(api.atexit, "register", lambda *_a, **_k: None)
    monkeypatch.setattr(
        api, "_spawn_child",
        lambda argv, env: captured.update(argv=argv, env=env) or types.SimpleNamespace(poll=lambda: None),
    )
    sentinel = OumigoWorker(manager_url="x", address="10.0.0.9", port=7001, model="m")
    monkeypatch.setattr(api, "_wait_worker_serving", lambda *_a, **_k: sentinel)
    return sentinel


def test_create_worker_reuses_last_manager(monkeypatch):
    """A bare oumigo_create_worker() reuses the manager this process created (no mDNS)."""
    mgr = api.OumigoManager(
        control_url="http://10.0.0.1:7014", data_url="http://10.0.0.1:7012", token="tok"
    )
    monkeypatch.setattr(api, "_last_manager", mgr)
    monkeypatch.setattr(api.discovery, "discover_manager", _no_mdns)
    captured: dict = {}
    sentinel = _stub_worker_spawn(monkeypatch, captured)

    worker = api.oumigo_create_worker()

    assert worker is sentinel
    i = captured["argv"].index("--manager-url")
    assert captured["argv"][i + 1] == "http://10.0.0.1:7014"   # reused manager's URL
    assert captured["env"]["OUMIGO_MANAGER_TOKEN"] == "tok"     # token inherited


def test_create_worker_accepts_manager_handle(monkeypatch):
    """An explicit manager= handle supplies the URL and token; no mDNS is attempted."""
    mgr = api.OumigoManager(
        control_url="http://10.0.0.2:7014", data_url="http://10.0.0.2:7012", token="h-tok"
    )
    monkeypatch.setattr(api, "_last_manager", None)
    monkeypatch.setattr(api.discovery, "discover_manager", _no_mdns)
    captured: dict = {}
    _stub_worker_spawn(monkeypatch, captured)

    api.oumigo_create_worker(manager=mgr)

    i = captured["argv"].index("--manager-url")
    assert captured["argv"][i + 1] == "http://10.0.0.2:7014"
    assert captured["env"]["OUMIGO_MANAGER_TOKEN"] == "h-tok"


class _JsonResp:
    """Minimal httpx-like response returning a fixed JSON body."""

    def __init__(self, payload):
        self._payload = payload
        self.last_url = None
        self.last_params = None

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_metrics_latest_enriches_with_worker_name(monkeypatch):
    """metrics() returns the latest slot per node, labeled with the Worker#N name."""
    mgr = api.OumigoManager(control_url="http://m", data_url="http://m")

    def fake_get(url, **kwargs):
        if url.endswith("/metrics/latest"):
            return _JsonResp(
                {"nodes": [{"node_id": "abc", "timestamp": "2026-07-23 22:00:00",
                            "metrics": {"worker:cpu_util_pct": 12.5, "gpu:util_pct": 40.0}}]}
            )
        if url.endswith("/workers"):
            return _JsonResp({"workers": [{"node_id": "abc", "name": "Worker#1"}]})
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(api.httpx, "get", fake_get)
    result = mgr.metrics()

    assert result == [
        {
            "node_id": "abc",
            "name": "Worker#1",
            "timestamp": "2026-07-23 22:00:00",
            "metrics": {"worker:cpu_util_pct": 12.5, "gpu:util_pct": 40.0},
        }
    ]


def test_metrics_latest_filters_by_prefix(monkeypatch):
    """prefixes= filters the per-node metric dict client-side for the latest snapshot."""
    mgr = api.OumigoManager(control_url="http://m", data_url="http://m")

    def fake_get(url, **kwargs):
        if url.endswith("/metrics/latest"):
            return _JsonResp(
                {"nodes": [{"node_id": "abc", "timestamp": "t",
                            "metrics": {"worker:cpu_util_pct": 1.0, "vllm:start_timestamp": 2.0}}]}
            )
        return _JsonResp({"workers": []})

    monkeypatch.setattr(api.httpx, "get", fake_get)
    result = mgr.metrics(prefixes=("worker:",))

    assert result[0]["metrics"] == {"worker:cpu_util_pct": 1.0}


def test_metrics_since_returns_raw_points(monkeypatch):
    """metrics(since=...) hits /metrics/since with after+prefix and returns raw points."""
    mgr = api.OumigoManager(control_url="http://m", data_url="http://m")
    seen: dict = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen["params"] = kwargs.get("params")
        return _JsonResp({"points": [{"node_id": "abc", "metric": "gpu:util_pct",
                                      "timestamp": "t", "value": 40.0}]})

    monkeypatch.setattr(api.httpx, "get", fake_get)
    points = mgr.metrics(since="2026-07-23 21:00:00", prefixes=("gpu:", "worker:"))

    assert seen["url"].endswith("/metrics/since")
    assert seen["params"] == {"after": "2026-07-23 21:00:00", "prefix": "gpu:,worker:"}
    assert points == [{"node_id": "abc", "metric": "gpu:util_pct", "timestamp": "t", "value": 40.0}]


def _rec(node_id, incarnation, state, address="10.0.0.9", port=7001):
    return {"node_id": node_id, "incarnation": incarnation, "state": state,
            "address": address, "port": port, "model": "m", "backend": "vllm"}


def test_wait_worker_serving_detects_reregistered_node_by_incarnation(monkeypatch):
    """Regression: a worker re-using a remembered node_id is matched via its bumped
    incarnation — not masked as 'already known' (the false-timeout bug)."""
    before = {"n1": 5}  # the manager still remembers node n1 at this host, incarnation 5
    child = types.SimpleNamespace(poll=lambda: None, returncode=None)
    # Our fresh worker re-registers as n1 with incarnation 6: stopped(old) -> init -> serving.
    # States use the real lowercase NodeState values the registry serializes (regression:
    # matching against uppercase silently never fired, killing healthy workers at timeout).
    responses = iter([
        [_rec("n1", 5, "stopped")],       # stale snapshot value -> not ours yet
        [_rec("n1", 6, "initializing")],  # fresh incarnation, still loading
        [_rec("n1", 6, "serving")],       # fresh + serving -> ours
    ])
    monkeypatch.setattr(api, "_list_workers", lambda _url: next(responses))
    monkeypatch.setattr(api.time, "sleep", lambda _s: None)

    # timeout=None (the default): wait indefinitely, returning once it reaches serving.
    worker = api._wait_worker_serving(
        child, "http://m", "10.0.0.9", before, None, 0.01, "vllm", None
    )
    assert worker.node_id == "n1"
    assert worker.port == 7001


def test_wait_worker_serving_fails_fast_on_failed(monkeypatch):
    """A fresh worker reaching FAILED raises immediately — even with no timeout (None)."""
    child = types.SimpleNamespace(poll=lambda: None, returncode=None)
    monkeypatch.setattr(api, "_list_workers", lambda _url: [_rec("n2", 1, "failed")])
    monkeypatch.setattr(api, "terminate", lambda _c, **_k: None)
    monkeypatch.setattr(api.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="FAILED"):
        api._wait_worker_serving(child, "http://m", "10.0.0.9", {}, None, 0.01, "vllm", None)


def test_wait_worker_serving_finite_timeout_raises_and_tears_down(monkeypatch):
    """A finite timeout that elapses (worker never serves) terminates the child and raises."""
    child = types.SimpleNamespace(poll=lambda: None, returncode=None)
    monkeypatch.setattr(api, "_list_workers", lambda _url: [_rec("n3", 1, "initializing")])
    monkeypatch.setattr(api.time, "sleep", lambda _s: None)
    killed = {"n": 0}
    monkeypatch.setattr(api, "terminate", lambda _c, **_k: killed.__setitem__("n", killed["n"] + 1))

    with pytest.raises(RuntimeError, match="did not reach SERVING within"):
        api._wait_worker_serving(child, "http://m", "10.0.0.9", {}, 0.05, 0.01, "vllm", None)
    assert killed["n"] == 1  # child torn down on the finite-timeout give-up


def test_worker_record_matches_on_address_and_port(monkeypatch):
    """OumigoWorker.state() picks the record for this worker's address:port."""
    worker = OumigoWorker(
        manager_url="http://m", address="10.0.0.9", port=7001, model="m"
    )

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            # Real registry casing: NodeState serializes lowercase.
            return {
                "workers": [
                    {"address": "10.0.0.8", "port": 7001, "state": "serving"},
                    {"address": "10.0.0.9", "port": 7001, "state": "initializing"},
                ]
            }

    monkeypatch.setattr(api.httpx, "get", lambda *_a, **_k: _Resp())
    assert worker.state() == "initializing"
    assert worker.is_serving() is False
