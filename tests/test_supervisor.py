"""Unit tests for the pure argv builder — no subprocess spawned."""

from __future__ import annotations

from oumigo.config.spec import NodeSpec
from oumigo.worker.supervisor import build_argv


def test_build_argv_minimal() -> None:
    argv = build_argv(NodeSpec(model="acme/tiny"))
    assert argv[:3] == ["vllm", "serve", "acme/tiny"]
    assert "--host" in argv and "0.0.0.0" in argv
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    # optional flags absent when unset
    assert "--max-model-len" not in argv
    assert "--download-dir" not in argv


def test_build_argv_full() -> None:
    spec = NodeSpec(
        model="acme/big",
        port=9001,
        dtype="float16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        download_dir="/models",
        extra_args=["--enforce-eager"],
    )
    argv = build_argv(spec)
    assert argv[argv.index("--port") + 1] == "9001"
    assert argv[argv.index("--dtype") + 1] == "float16"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.85"
    assert argv[argv.index("--max-model-len") + 1] == "8192"
    assert argv[argv.index("--download-dir") + 1] == "/models"
    assert argv[-1] == "--enforce-eager"  # extra args appended verbatim, last


def test_build_node_spec_from_manager_config() -> None:
    from oumigo.manager.settings import build_node_spec

    assert build_node_spec({}) is None
    assert build_node_spec({"model": {"name": None}}) is None
    spec = build_node_spec(
        {"model": {"name": "acme/tiny", "dtype": "float16", "tensor_parallel_size": 2}}
    )
    assert spec is not None
    assert spec.model == "acme/tiny"
    assert spec.dtype == "float16"
    assert spec.tensor_parallel_size == 2
