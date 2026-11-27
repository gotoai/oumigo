"""Closing a worker: the close request, the manager's STOP, routing, and the drain limit."""

from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient

from oumigo.config.catalog import AgentDefaults
from oumigo.protocol.states import NodeState
from oumigo.service.manager.control.registry import Registry
from oumigo.service.manager.control.server import _close_loop, create_app
from oumigo.service.manager.credentials import Authenticator, AuthStore
from oumigo.service.manager.router.server import WorkerPool
from oumigo.service.worker import coordinator
from oumigo.service.worker.coordinator import (
    EXIT_FAILED,
    EXIT_REFUSED,
    EXIT_STOPPED,
    WorkerCoordinator,
    _drain_timeout_from_env,
)
from tests._fleet import one_profile_catalog, register_body

NODE = "0123456789abcdef"


def _client(store: AuthStore | None = None, registry: Registry | None = None,
            **app_options) -> tuple[TestClient, AuthStore, Registry]:
    store = store or AuthStore()
    registry = registry or Registry()
    options = {"catalog": one_profile_catalog(), **app_options}
    app = create_app(registry, None, auth=Authenticator(store, None, wan=False), **options)
    return TestClient(app), store, registry


def _beat(client: TestClient, state: str = "serving") -> dict:
    resp = client.post("/heartbeat", json={"node_id": NODE, "node_state": state})
    assert resp.status_code == 200
    return resp.json()


# --- store ------------------------------------------------------------------------


def test_close_requests_are_idempotent_and_survive_reopen(tmp_path) -> None:
    store = AuthStore.at(tmp_path)
    store.request_close(NODE, "yyw")
    first = store.close_requests()[NODE]
    store.request_close(NODE, "cli")  # asking again keeps the first request
    reopened = AuthStore.at(tmp_path)  # the CLI or console writes, the manager reads
    assert reopened.close_requests()[NODE] == first
    assert first.requested_by == "yyw"
    assert reopened.clear_close_request(NODE)
    assert not store.clear_close_request(NODE)
    assert store.close_requests() == {}


# --- manager ----------------------------------------------------------------------


def test_heartbeat_carries_stop_until_the_worker_stopped() -> None:
    client, store, registry = _client()
    client.post("/register", json=register_body(NODE))
    assert _beat(client)["command"] is None

    store.request_close(NODE, "yyw")
    assert _beat(client)["command"] == "stop"
    assert registry.get(NODE).closing
    assert _beat(client, "draining")["command"] == "stop"  # still applies while draining

    assert _beat(client, "stopped")["command"] is None
    assert store.close_requests() == {}  # done


def test_a_new_run_clears_a_close_left_from_the_previous_run() -> None:
    client, store, registry = _client()
    client.post("/register", json=register_body(NODE, incarnation=1))
    store.request_close(NODE, "yyw")
    _beat(client)
    assert registry.get(NODE).closing

    # The same run re-registers (e.g. after a manager restart): the close still applies.
    client.post("/register", json=register_body(NODE, incarnation=1))
    assert NODE in store.close_requests()
    assert _beat(client)["command"] == "stop"

    # A new run (its container restarted before it reported STOPPED): a fresh start.
    client.post("/register", json=register_body(NODE, incarnation=2))
    assert store.close_requests() == {}
    assert not registry.get(NODE).closing
    assert _beat(client)["command"] is None


def test_a_closed_worker_that_comes_back_is_routable_and_closable_again() -> None:
    """Close -> stopped -> the container restarts the worker (a new run): it must be
    routable and closable again, not stuck as closing."""
    client, store, registry = _client()
    pool = WorkerPool(registry, 1)
    client.post("/register", json=register_body(NODE, incarnation=6))
    store.request_close(NODE, "admin")
    assert _beat(client)["command"] == "stop"
    _beat(client, "stopped")
    assert store.close_requests() == {} and not registry.get(NODE).closing

    client.post("/register", json=register_body(NODE, incarnation=7))
    assert _beat(client)["command"] is None
    assert not registry.get(NODE).closing and pool.healthy_count() == 1
    store.request_close(NODE, "admin")  # and it can be closed again
    assert _beat(client)["command"] == "stop"


def test_a_new_run_clears_a_stale_closing_flag_even_without_a_request() -> None:
    client, _, registry = _client()
    client.post("/register", json=register_body(NODE, incarnation=1))
    registry.set_closing(NODE)  # e.g. left over by an older manager build
    client.post("/register", json=register_body(NODE, incarnation=2))
    assert not registry.get(NODE).closing


def test_close_loop_marks_closing_finishes_and_drops_unknown_nodes() -> None:
    registry = Registry()
    store = AuthStore()
    registry.register(NODE, "10.0.0.5", "serving", 0, {})
    store.request_close(NODE, "yyw")
    store.request_close("ffffffffffffffff", "cli")  # never registered

    async def run(seconds: float) -> None:
        task = asyncio.create_task(_close_loop(registry, store, stale_after=0.1,
                                               interval=0.05))
        await asyncio.sleep(seconds)
        task.cancel()

    asyncio.run(run(0.3))
    assert registry.get(NODE).closing
    assert set(store.close_requests()) == {NODE}  # the unknown one was dropped

    registry.heartbeat(NODE, NodeState.STOPPED.value)
    asyncio.run(run(0.2))
    assert store.close_requests() == {} and not registry.get(NODE).closing

    # A closing flag with no request behind it is cleared: the node is routable again.
    registry.heartbeat(NODE, NodeState.SERVING.value)
    registry.set_closing(NODE)
    asyncio.run(run(0.2))
    assert not registry.get(NODE).closing


def test_router_skips_a_closing_worker() -> None:
    registry = Registry()
    registry.register(NODE, "10.0.0.5", "serving", 0, {})
    pool = WorkerPool(registry, 1)
    assert pool.healthy_count() == 1
    registry.set_closing(NODE)
    assert pool.healthy_count() == 0
    # A re-registration of the same run keeps it closing.
    registry.register(NODE, "10.0.0.5", "serving", 0, {})
    assert pool.healthy_count() == 0


def test_spec_carries_the_drain_limit_from_the_profile_or_the_fleet() -> None:
    client, _, _ = _client(fleet_agent_defaults={"turn_timeout": 600.0})
    assert client.get("/spec").json()["node_spec"]["drain_timeout_s"] == 600.0

    catalog = one_profile_catalog(agent=AgentDefaults(turn_timeout=1800))
    client, _, _ = _client(catalog=catalog, fleet_agent_defaults={"turn_timeout": 600.0})
    assert client.get("/spec").json()["node_spec"]["drain_timeout_s"] == 1800.0

    client, _, _ = _client()
    assert client.get("/spec").json()["node_spec"]["drain_timeout_s"] is None


# --- worker -----------------------------------------------------------------------


def test_drain_timeout_from_env(monkeypatch) -> None:
    monkeypatch.delenv("WK_DRAIN_TIMEOUT", raising=False)
    assert _drain_timeout_from_env() is None
    monkeypatch.setenv("WK_DRAIN_TIMEOUT", " ")
    assert _drain_timeout_from_env() is None
    monkeypatch.setenv("WK_DRAIN_TIMEOUT", "900")
    assert _drain_timeout_from_env() == 900.0
    for bad in ("0", "-5", "ten", "nan"):
        monkeypatch.setenv("WK_DRAIN_TIMEOUT", bad)
        with pytest.raises(SystemExit, match="WK_DRAIN_TIMEOUT"):
            _drain_timeout_from_env()


class _Stopped:
    """A backend that is already down, so the drain is immediate."""

    running = False

    def stop(self, grace_s: float) -> None:
        pass


def _run(monkeypatch, final_state: NodeState | None = None, refused: bool = False) -> int:
    """Run a coordinator whose loop ends at once, in the given state."""
    worker = WorkerCoordinator("https://m", None, NODE, None, spec=object(),
                               vllm_port=7001, metrics_enabled=False)
    monkeypatch.setattr(worker, "_install_signals", lambda: None)
    monkeypatch.setattr(worker, "_register", lambda: None)
    monkeypatch.setattr(worker, "_send_heartbeat", lambda: None)

    def start(_spec) -> None:
        worker.supervisor = _Stopped()

    def loop() -> None:
        if final_state is not None:
            worker.node_state = final_state
        worker._refused = refused

    monkeypatch.setattr(worker, "_start_vllm", start)
    monkeypatch.setattr(worker, "_loop", loop)
    return worker.run()


def test_exit_code_tells_a_normal_stop_from_a_failure(monkeypatch) -> None:
    assert _run(monkeypatch) == EXIT_STOPPED == 0
    assert _run(monkeypatch, NodeState.FAILED) == EXIT_FAILED
    assert _run(monkeypatch, refused=True) == EXIT_REFUSED


def test_drain_waits_at_most_the_limit(monkeypatch) -> None:
    """The drain ends as soon as nothing is in flight, and never after the limit."""
    from oumigo.protocol.states import RunState

    class Busy:
        running = True
        stopped = False

        def __init__(self, busy_checks: int) -> None:
            self.busy_checks = busy_checks

        def run_state(self):
            self.busy_checks -= 1
            return RunState.EXECUTING if self.busy_checks >= 0 else RunState.IDLE

        def stop(self, grace_s: float) -> None:
            self.stopped = True

    def drain(busy_checks: int, limit: float) -> float:
        worker = WorkerCoordinator("https://m", None, NODE, None, spec=object(),
                                   vllm_port=7001, metrics_enabled=False,
                                   drain_timeout_s=limit)
        worker.interval = 0.05
        worker.supervisor = Busy(busy_checks)
        monkeypatch.setattr(worker, "_send_heartbeat", lambda: None)
        started = coordinator.time.monotonic()
        worker._drain_and_stop()
        assert worker.supervisor.stopped and worker.node_state == NodeState.STOPPED
        return coordinator.time.monotonic() - started

    assert drain(busy_checks=2, limit=10.0) < 1.0   # done once the requests finished
    assert 0.25 <= drain(busy_checks=10**6, limit=0.3) < 1.0  # cut off at the limit


# --- CLI --------------------------------------------------------------------------


def _manager_yaml(tmp_path) -> str:
    config = tmp_path / "manager.yaml"
    config.write_text(f"control_plane:\n  port: 1\nauth:\n  state_dir: {tmp_path}\n")
    return str(config)


def test_cli_node_close_writes_the_request(tmp_path) -> None:
    from typer.testing import CliRunner

    from oumigo.cli.main import app

    config = _manager_yaml(tmp_path)
    runner = CliRunner()
    # The manager isn't running: a node id is still accepted, a name is not.
    result = runner.invoke(app, ["manager", "node", "close", NODE, "-c", config])
    assert result.exit_code == 0, result.output
    assert AuthStore.at(tmp_path).close_requests()[NODE].requested_by == "cli"
    result = runner.invoke(app, ["manager", "node", "close", "Worker#1", "-c", config])
    assert result.exit_code == 1


def test_cli_admin_password_prints_the_hash_line(monkeypatch) -> None:
    from typer.testing import CliRunner

    from oumigo.cli.main import app
    from oumigo.service.manager.admin_account import verify_password

    answers = iter(["Correct-horse9", "Correct-horse9"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    result = CliRunner().invoke(app, ["manager", "admin-password"])
    assert result.exit_code == 0, result.output
    key, _, value = result.output.strip().partition("=")
    assert key == "MG_ADMIN_PASSWORD_HASH" and verify_password("Correct-horse9", value)

    monkeypatch.setattr("getpass.getpass", lambda prompt="": "weak")
    result = CliRunner().invoke(app, ["manager", "admin-password"])
    assert result.exit_code == 1 and "at least 10 characters" in result.output


def test_metrics_max_keeping_age_from_env(monkeypatch) -> None:
    from oumigo.service.worker.coordinator import _metrics_max_age_from_env

    monkeypatch.delenv("WK_METRICS_MAX_KEEPING_AGE", raising=False)
    assert _metrics_max_age_from_env() == 60.0
    monkeypatch.setenv("WK_METRICS_MAX_KEEPING_AGE", "120")
    assert _metrics_max_age_from_env() == 120.0
    for bad in ("10", "34.9", "soon", "nan"):  # below one report interval plus a sample
        monkeypatch.setenv("WK_METRICS_MAX_KEEPING_AGE", bad)
        with pytest.raises(SystemExit, match="at least 35"):
            _metrics_max_age_from_env()
