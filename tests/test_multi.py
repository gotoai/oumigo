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
# without serving, "die" serves and then exits on its own.
FAKE_WORKER = textwrap.dedent("""
    import os, signal, sys, time
    profile = os.environ["OUMIGO_MODEL_PROFILE"]
    mode = os.environ["FAKE_" + profile]
    log = open(os.environ["FAKE_LOG"], "a", buffering=1)
    log.write(f"start {profile} {' '.join(sys.argv[1:-2])}\\n")
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    time.sleep(0.3)
    if mode == "fail":
        sys.exit(3)
    fd = int(sys.argv[sys.argv.index("--ready-fd") + 1])
    os.write(fd, b"serving\\n")
    os.close(fd)
    log.write(f"ready {profile}\\n")
    if mode == "die":
        time.sleep(0.3)
        sys.exit(4)
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

    def run(modes: dict[str, str], *, stop_when_ready: bool = False) -> tuple[int, list[str]]:
        env = {**os.environ, "FAKE_LOG": str(log), **{f"FAKE_{p}": m for p, m in modes.items()}}
        if stop_when_ready:  # a platform's SIGTERM, once every profile serves
            def stop() -> None:
                while log.read_text().count("ready ") < len(modes):
                    time.sleep(0.05)
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
                                    metrics_enabled=False, on_serving=lambda: calls.append(1))
    coordinator.supervisor = Healthy()
    for _ in range(2):                     # first start, then a restart
        coordinator.node_state = NodeState.INITIALIZING
        coordinator._reconcile()
        assert coordinator.node_state == NodeState.SERVING
    assert calls == [1]


def test_cli_hands_a_profile_list_to_the_supervisor(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from oumigo.cli.main import app

    seen = {}

    def fake_run(profiles, args):
        seen.update(profiles=profiles, args=args)
        return 7

    monkeypatch.setattr(multi, "run_profiles", fake_run)
    for var in ("OUMIGO_TRANSPORT", "OUMIGO_GPU_ID", "OUMIGO_DISCOVER_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OUMIGO_MODEL_PROFILE", "harrier-oss-v1-0.6b@32k,ruri-v3-310m@8k")
    monkeypatch.setenv("OUMIGO_MANAGER_URL", "https://oumigo.example.com")
    env_file = tmp_path / "worker.env"
    result = CliRunner().invoke(app, ["worker", "run", "--transport", "tunnel",
                                      "--state-dir", "/data/state", "--env-file", str(env_file)])
    assert result.exit_code == 7, result.output
    assert seen["profiles"] == ["harrier-oss-v1-0.6b@32k", "ruri-v3-310m@8k"]
    assert seen["args"] == ["--transport", "tunnel", "--discover-timeout", "60.0",
                            "--backend", "vllm", "--manager-url", "https://oumigo.example.com",
                            "--state-dir", "/data/state", "--env-file", str(env_file)]
