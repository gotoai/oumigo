"""Tests for the stdlib .env loader and the worker CLI wiring."""

from __future__ import annotations

import os

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

    import oumigo.worker.coordinator as coordinator
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
