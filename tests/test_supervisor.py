"""Unit tests for the pure argv builder + port preflight — no subprocess spawned."""

from __future__ import annotations

import socket

import pytest

from oumigo.config.spec import NodeSpec
from oumigo.worker.supervisor import PortUnavailable, build_argv, find_free_port


def test_build_argv_minimal() -> None:
    argv = build_argv(NodeSpec(model="acme/tiny"))
    assert argv[:3] == ["vllm", "serve", "acme/tiny"]
    assert "--host" in argv and "0.0.0.0" in argv
    assert argv[argv.index("--port") + 1] == "7001"
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


# --- port preflight -------------------------------------------------------------


def test_build_argv_port_override() -> None:
    argv = build_argv(NodeSpec(model="m", port=7001), port=7005)
    assert argv[argv.index("--port") + 1] == "7005"


def test_find_free_port_returns_preferred_when_free() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    # socket closed -> the port is free again; preflight should hand it straight back
    assert find_free_port("127.0.0.1", free) == free


def test_find_free_port_skips_occupied() -> None:
    occ = socket.socket()
    occ.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occ.bind(("127.0.0.1", 0))
    port = occ.getsockname()[1]
    occ.listen()
    try:
        assert find_free_port("127.0.0.1", port) > port  # skipped the occupied one
    finally:
        occ.close()


def test_find_free_port_raises_when_exhausted() -> None:
    occ = socket.socket()
    occ.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occ.bind(("127.0.0.1", 0))
    port = occ.getsockname()[1]
    occ.listen()
    try:
        with pytest.raises(PortUnavailable):
            find_free_port("127.0.0.1", port, max_tries=1)  # only the occupied port
    finally:
        occ.close()
