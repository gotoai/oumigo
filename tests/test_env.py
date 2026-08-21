"""Tests for the stdlib .env loader and the worker CLI wiring."""

from __future__ import annotations

import os

import pytest

from oumigo.common.env import load_env_file


def test_parses_pairs_comments_quotes_and_export(tmp_path, monkeypatch):
    for k in ("A", "B", "C", "D", "E"):
        monkeypatch.delenv(k, raising=False)
    envf = tmp_path / ".env"
    envf.write_text(
        "# a comment\n"
        "\n"
        "A=1\n"
        "export B=2\n"
        'C="quoted value"\n'
        "D='single'\n"
        "E=\n"  # empty value is valid
        "not_a_pair_line\n"
    )
    n = load_env_file(envf)
    assert n == 5
    assert os.environ["A"] == "1"
    assert os.environ["B"] == "2"
    assert os.environ["C"] == "quoted value"
    assert os.environ["D"] == "single"
    assert os.environ["E"] == ""


def test_existing_environment_wins_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED", "from-shell")
    (tmp_path / ".env").write_text("SHARED=from-file\n")
    load_env_file(tmp_path / ".env")
    assert os.environ["SHARED"] == "from-shell"  # explicit env not overridden


def test_override_true_replaces(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED", "from-shell")
    (tmp_path / ".env").write_text("SHARED=from-file\n")
    load_env_file(tmp_path / ".env", override=True)
    assert os.environ["SHARED"] == "from-file"


def test_missing_file_is_noop(tmp_path):
    assert load_env_file(tmp_path / "does-not-exist.env") == 0


def test_worker_run_loads_env_before_starting(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import oumigo.service.worker.coordinator as coordinator
    from oumigo.cli.main import app

    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    envf = tmp_path / ".env"
    envf.write_text("VLLM_USE_FLASHINFER_SAMPLER=0\n")

    called = {}
    monkeypatch.setattr(coordinator, "run_worker", lambda **kw: called.update(kw))

    result = CliRunner().invoke(
        app, ["worker", "run", "--manager-url", "http://x", "--env-file", str(envf)]
    )
    assert result.exit_code == 0, result.output
    assert called  # run_worker was reached
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"  # .env applied to environ


def test_flashinfer_sampler_defaults_to_off(monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    _apply_env_overrides(NodeSpec(model="m"))
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"  # backend child inherits the default


def test_flashinfer_sampler_env_wins_over_default(monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    monkeypatch.delenv("MODEL_NAME", raising=False)
    _apply_env_overrides(NodeSpec(model="m"))
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"  # explicit opt-in preserved


# --- MODEL_STORAGE_LOCATION: per-worker override of where the weights live ---------


def test_storage_location_env_overrides_the_fleet_spec(tmp_path, monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    local = tmp_path / "weights"
    local.mkdir()
    (local / "config.json").write_text("{}")
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("MODEL_STORAGE_LOCATION", f"file://{local}")

    spec = _apply_env_overrides(NodeSpec(model="acme/tiny"))
    assert spec.model_ref == str(local)      # backend loads from disk
    assert spec.model == "acme/tiny"         # clients still address the fleet name


def test_storage_location_env_none_falls_back_to_the_hub(tmp_path, monkeypatch):
    """A node without the files opts out of a fleet-wide location."""
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("MODEL_STORAGE_LOCATION", "none")
    spec = _apply_env_overrides(
        NodeSpec(model="acme/tiny", storage_location="file:///nonexistent/models/tiny")
    )
    assert spec.storage_location is None
    assert spec.model_ref == "acme/tiny"


def test_missing_local_weights_fail_before_the_backend_starts(monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_STORAGE_LOCATION", raising=False)
    with pytest.raises(SystemExit, match="does not exist on this worker"):
        _apply_env_overrides(NodeSpec(model="m", storage_location="file:///no/such/dir"))


def test_directory_without_config_json_is_rejected(tmp_path, monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_STORAGE_LOCATION", raising=False)
    with pytest.raises(SystemExit, match="no config.json"):
        _apply_env_overrides(NodeSpec(model="m", storage_location=f"file://{empty}"))


def test_bad_storage_location_env_exits_cleanly(monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("MODEL_STORAGE_LOCATION", "s3://bucket/model")
    with pytest.raises(SystemExit, match="MODEL_STORAGE_LOCATION"):
        _apply_env_overrides(NodeSpec(model="m"))
