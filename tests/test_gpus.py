"""Choosing a worker's GPU (`--gpu-id` / WK_GPU_ID) and the profiles'
`gpu_memory_target_gib`."""

from __future__ import annotations

import subprocess
import sys

import pytest

from oumigo.config.spec import NodeSpec
from oumigo.service.worker import coordinator, gpus, metrics


@pytest.fixture(autouse=True)
def _no_gpu_env(monkeypatch):
    """Each test starts with no GPU chosen, and restores what the machine had."""
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", gpus.ENV_GPU_ID):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _spec(**kw) -> NodeSpec:
    return NodeSpec(model="acme/embed", profile="embed@8k", **kw)


# --- GPU ids --------------------------------------------------------------------------


def test_gpu_ids_are_nvidia_smi_indexes_or_uuids():
    assert gpus.parse_gpu_ids("1") == ["1"]
    assert gpus.parse_gpu_ids(" 0, 1 ") == ["0", "1"]
    assert gpus.parse_gpu_ids("GPU-3f2a9c1e-77aa-4b1e") == ["GPU-3f2a9c1e-77aa-4b1e"]
    for bad in ("", "gpu1", "-1", "0;1", "cuda:0"):
        with pytest.raises(ValueError):
            gpus.parse_gpu_ids(bad)


def test_choosing_a_gpu_pins_the_backend_in_nvidia_smi_order():
    assert gpus.select_gpus("1") == ["1"]
    assert gpus.os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert gpus.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert gpus.first_gpu() == "1"


def test_an_operator_set_cuda_visible_devices_is_honoured(monkeypatch):
    assert gpus.select_gpus(None) is None and gpus.first_gpu() == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert gpus.select_gpus(None) == ["2", "3"]
    assert gpus.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"  # so 2 means nvidia-smi's 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "nonsense")
    assert gpus.selected_gpus() is None


# --- the memory target ----------------------------------------------------------------


def test_the_target_becomes_this_gpus_share(monkeypatch):
    monkeypatch.setattr(coordinator, "probe_gpu_vram_gib", lambda gpu=None: 16.0)
    spec = coordinator._apply_memory_target(_spec(gpu_memory_target_gib=3.0,
                                                  gpu_memory_utilization=0.85))
    assert spec.gpu_memory_utilization == 0.1875
    monkeypatch.setattr(coordinator, "probe_gpu_vram_gib", lambda gpu=None: 48.0)
    big = coordinator._apply_memory_target(_spec(gpu_memory_target_gib=3.0))
    assert big.gpu_memory_utilization == 0.0625  # the same GiB, a smaller share


@pytest.mark.parametrize("target", [None, 0.0])
def test_without_a_target_the_profiles_share_applies(monkeypatch, target):
    monkeypatch.setattr(coordinator, "probe_gpu_vram_gib", lambda gpu=None: 16.0)
    spec = _spec(gpu_memory_target_gib=target, gpu_memory_utilization=0.85)
    assert coordinator._apply_memory_target(spec).gpu_memory_utilization == 0.85


def test_a_target_the_gpu_cannot_hold_or_read_stops_the_worker(monkeypatch):
    monkeypatch.setattr(coordinator, "probe_gpu_vram_gib", lambda gpu=None: 4.0)
    with pytest.raises(SystemExit, match="needs 6 GiB of GPU memory; GPU 0 has 4.0 GiB"):
        coordinator._apply_memory_target(_spec(gpu_memory_target_gib=6.0))
    monkeypatch.setattr(coordinator, "probe_gpu_vram_gib", lambda gpu=None: None)
    with pytest.raises(SystemExit, match="cannot be read"):
        coordinator._apply_memory_target(_spec(gpu_memory_target_gib=2.0))


def test_checks_read_the_chosen_gpus_and_the_smallest_decides(monkeypatch):
    sizes = {"0": 80.0, "1": 16.0, "2": 24.0}
    seen: list[str] = []

    def probe(gpu=None):
        seen.append(gpu)
        return sizes[gpu]

    monkeypatch.setattr(coordinator, "probe_gpu_vram_gib", probe)
    gpus.select_gpus("1")
    with pytest.raises(SystemExit, match="GPU 1 has 16.0 GiB"):
        coordinator._check_requirements(_spec(min_vram_gib=20), "vllm")
    gpus.select_gpus("2,1")
    spec = coordinator._apply_memory_target(_spec(gpu_memory_target_gib=4.0,
                                                  tensor_parallel_size=2))
    assert spec.gpu_memory_utilization == 0.25  # 4 of GPU 1's 16, the smaller card
    assert "0" not in seen


def test_gpu_ids_must_match_the_profiles_tensor_parallel_size(monkeypatch):
    monkeypatch.setattr(coordinator.client, "fetch_spec",
                        lambda *a, **kw: (_spec(tensor_parallel_size=1), "LAN"))
    with pytest.raises(SystemExit, match="names 2 GPU.s., but profile embed@8k runs on 1"):
        coordinator.run_worker("http://manager:7014", None, gpu_id="0,1")
    with pytest.raises(SystemExit, match="--gpu-id / WK_GPU_ID: not a GPU id: x"):
        coordinator.run_worker("http://manager:7014", None, gpu_id="x")


# --- where the setting comes from -----------------------------------------------------


def test_the_flag_beats_the_environment_which_beats_the_env_file(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from oumigo.cli.main import app

    envf = tmp_path / ".env"
    envf.write_text("WK_GPU_ID=2\n")
    called: dict = {}
    monkeypatch.setattr(coordinator, "run_worker", lambda **kw: called.update(kw))

    def run(*args: str) -> str | None:
        result = CliRunner().invoke(app, ["worker", "run", "--env-file", str(envf), *args])
        assert result.exit_code == 0, result.output
        return called["gpu_id"]

    assert run() == "2"
    monkeypatch.setenv(gpus.ENV_GPU_ID, "1")
    assert run() == "1"
    assert run("--gpu-id", "0") == "0"


# --- what the worker reads ------------------------------------------------------------


def test_probes_and_metrics_ask_nvidia_smi_about_the_chosen_gpus(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # no NVML: the nvidia-smi path
    monkeypatch.setattr(metrics.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    calls: list[list[str]] = []

    def run(argv, **kw):
        calls.append(argv)
        out = ("16384\n" if "--query-gpu=memory.total" in argv
               else "1, 37, 500, 16384, 53, 24.9\n")
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(metrics.subprocess, "run", run)
    gpus.select_gpus("1")
    assert metrics.probe_gpu_vram_gib() == 16.0
    assert calls[-1][-2:] == ["-i", "1"]
    sampler = metrics._GpuSampler()
    sampler.start()
    out = sampler.sample()
    assert calls[-1][-2:] == ["-i", "1"] and "gpu:#1_util_pct" in out
