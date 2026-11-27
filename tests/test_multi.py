"""Several profiles on one node: start order, readiness, and stopping as one unit.

The children are a stand-in worker that speaks the `--ready-fd` protocol and logs what
it does, so no manager or backend is involved.
"""

from __future__ import annotations

import os
import signal
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from oumigo.service.worker import multi
from oumigo.service.worker.multi import parse_profiles, run_profiles

# Behaviour per profile, from $FAKE_<profile>: "serve" serves until stopped, "fail" exits
# without serving, "die" serves and then exits on its own, "close" serves and then exits
# 0 (closed by the manager), "quit" exits 0 before serving.
FAKE_WORKER = textwrap.dedent("""
    import os, signal, sys, time
    profile = os.environ["WK_MODEL_PROFILE"]
    mode = os.environ["FAKE_" + profile]
    log = open(os.environ["FAKE_LOG"], "a", buffering=1)
    log.write(f"start {profile} {' '.join(sys.argv[1:-2])}\\n")
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    time.sleep(0.3)
    if mode == "fail":
        sys.exit(3)
    if mode == "quit":
        log.write(f"exit {profile}\\n")
        sys.exit(0)
    fd = int(sys.argv[sys.argv.index("--ready-fd") + 1])
    os.write(fd, b"serving 900\\n")
    os.close(fd)
    log.write(f"ready {profile}\\n")
    if mode == "die":
        time.sleep(0.3)
        sys.exit(4)
    if mode == "close":
        time.sleep(0.3)
        log.write(f"exit {profile}\\n")
        sys.exit(0)
    while not stop:
        time.sleep(0.05)
    log.write(f"stop {profile}\\n")
""")


@pytest.fixture
def fake(tmp_path: Path, monkeypatch):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    log = tmp_path / "log"
    log.touch()
    monkeypatch.setattr(multi, "POLL_S", 0.05)

    def run(modes: dict[str, str], *, stop_when_ready: bool = False,
            stop_on: str | None = None) -> tuple[int, list[str]]:
        env = {**os.environ, "FAKE_LOG": str(log), **{f"FAKE_{p}": m for p, m in modes.items()}}
        if stop_when_ready or stop_on:  # a platform's SIGTERM, once every profile serves
            def stop() -> None:
                while log.read_text().count("ready ") < len(modes):
                    time.sleep(0.05)
                while stop_on and stop_on not in log.read_text():
                    time.sleep(0.05)
                time.sleep(0.3 if stop_on else 0)  # time to react, were it going to
                os.kill(os.getpid(), signal.SIGTERM)
            threading.Thread(target=stop, daemon=True).start()
        code = run_profiles(list(modes), ["--backend", "vllm"], env,
                            command=(sys.executable, str(script)))
        return code, log.read_text().splitlines()

    return run


def test_parse_profiles() -> None:
    assert parse_profiles(None) == []
    assert parse_profiles("ruri-v3-310m@8k") == ["ruri-v3-310m@8k"]
    assert parse_profiles(" harrier-oss-v1-0.6b@32k, ruri-v3-310m@8k ,") == [
        "harrier-oss-v1-0.6b@32k", "ruri-v3-310m@8k"]
    with pytest.raises(ValueError, match="more than once: a"):
        parse_profiles("a,b,a")


def test_starts_each_once_the_previous_serves_and_stops_all_on_signal(fake) -> None:
    code, log = fake({"big": "serve", "small": "serve"}, stop_when_ready=True)
    assert code == 0
    assert log[:4] == ["start big --backend vllm", "ready big",
                       "start small --backend vllm", "ready small"]
    assert sorted(log[4:]) == ["stop big", "stop small"]


def test_a_profile_that_cannot_start_stops_the_others(fake) -> None:
    code, log = fake({"big": "serve", "small": "fail"})
    assert code == 1
    assert log == ["start big --backend vllm", "ready big", "start small --backend vllm",
                   "stop big"]


def test_a_profile_that_exits_later_stops_the_others(fake) -> None:
    code, log = fake({"big": "serve", "small": "die"})
    assert code == 1
    assert log[-1] == "stop big"


def test_coordinator_reports_serving_once() -> None:
    """The ready hook fires on the first SERVING edge only, not after a backend restart."""
    from oumigo.protocol.states import NodeState
    from oumigo.service.worker.coordinator import WorkerCoordinator

    class Healthy:
        running = True

        def is_healthy(self) -> bool:
            return True

        def run_state(self):
            return None

    calls = []
    coordinator = WorkerCoordinator("https://m", None, "n1", None, spec=None, vllm_port=7001,
                                    metrics_enabled=False, on_serving=calls.append)
    coordinator.supervisor = Healthy()
    for _ in range(2):                     # first start, then a restart
        coordinator.node_state = NodeState.INITIALIZING
        coordinator._reconcile()
        assert coordinator.node_state == NodeState.SERVING
    assert calls == [120.0]  # the drain limit


def test_cli_hands_a_profile_list_to_the_supervisor(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from oumigo.cli.main import app

    seen = {}

    def fake_run(profiles, args):
        seen.update(profiles=profiles, args=args)
        return 7

    monkeypatch.setattr(multi, "run_profiles", fake_run)
    for var in ("WK_TRANSPORT", "WK_GPU_ID", "WK_DISCOVER_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WK_MODEL_PROFILE", "harrier-oss-v1-0.6b@32k,ruri-v3-310m@8k")
    monkeypatch.setenv("WK_MANAGER_URL", "https://oumigo.example.com")
    env_file = tmp_path / "worker.env"
    result = CliRunner().invoke(app, ["worker", "run", "--transport", "tunnel",
                                      "--state-dir", "/data/state", "--env-file", str(env_file)])
    assert result.exit_code == 7, result.output
    assert seen["profiles"] == ["harrier-oss-v1-0.6b@32k", "ruri-v3-310m@8k"]
    assert seen["args"] == ["--transport", "tunnel", "--discover-timeout", "60.0",
                            "--backend", "vllm", "--manager-url", "https://oumigo.example.com",
                            "--state-dir", "/data/state", "--env-file", str(env_file)]


def test_a_closed_profile_leaves_the_others_running(fake) -> None:
    code, log = fake({"big": "close", "small": "serve"}, stop_on="exit big")
    assert code == 0  # a failure would have stopped "small" and exited 1
    assert log[-1] == "stop small"  # stopped by the signal, after "big" had closed


def test_the_supervisor_exits_0_once_every_profile_closed(fake) -> None:
    code, log = fake({"big": "close", "small": "close"})
    assert code == 0
    assert sorted(line for line in log if line.startswith("exit")) == ["exit big", "exit small"]


def test_a_profile_closed_before_serving_does_not_stop_the_next(fake) -> None:
    code, log = fake({"big": "quit", "small": "close"})
    assert code == 0
    assert "ready small" in log


def test_ready_message_carries_the_drain_limit() -> None:
    def ready(message: bytes) -> tuple[bool, float | None]:
        read_fd, write_fd = os.pipe()
        os.write(write_fd, message)
        os.close(write_fd)
        return multi._wait_ready(None, read_fd, lambda: False)

    assert ready(b"serving 1800\n") == (True, 1800.0)
    assert ready(b"serving\n") == (True, None)  # a child that sends no limit
    assert ready(b"") == (False, None)           # exited before serving
    assert multi._stop_grace([]) == 120 + 30 + 30
    assert multi._stop_grace([900.0, 1800.0]) == 1800 + 30 + 30


def test_ready_callback_writes_the_drain_limit() -> None:
    read_fd, write_fd = os.pipe()
    multi.ready_callback(write_fd)(1800.0)
    assert os.read(read_fd, 64) == b"serving 1800\n"
    os.close(read_fd)
