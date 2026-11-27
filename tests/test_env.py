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


def test_worker_options_come_from_the_env_file(tmp_path, monkeypatch):
    # typer reads `envvar=` before the env file is loaded; the documented worker setup
    # (WK_MANAGER_URL / WK_TRANSPORT in .env) must still reach run_worker.
    from typer.testing import CliRunner

    import oumigo.service.worker.coordinator as coordinator
    from oumigo.cli.main import app

    names = ("WK_MANAGER_URL", "WK_TRANSPORT", "WK_DISCOVER_TIMEOUT")
    for name in names:
        monkeypatch.setenv(name, "")   # registers the original state, so teardown restores it
        monkeypatch.delenv(name)       # and the env file's values loaded below don't leak
    envf = tmp_path / ".env"
    envf.write_text("WK_MANAGER_URL=https://m.example\nWK_TRANSPORT=tunnel\n"
                    "WK_DISCOVER_TIMEOUT=5\n")
    called = {}
    monkeypatch.setattr(coordinator, "run_worker", lambda **kw: called.update(kw))

    result = CliRunner().invoke(app, ["worker", "run", "--env-file", str(envf)])
    assert result.exit_code == 0, result.output
    assert (called["manager_url"], called["transport"], called["discover_timeout"]) == (
        "https://m.example", "tunnel", 5.0)

    # A flag still beats the env file; with nothing set, the defaults apply.
    result = CliRunner().invoke(app, ["worker", "run", "--env-file", str(envf),
                                      "--transport", "direct"])
    assert called["transport"] == "direct"
    for name in names:
        monkeypatch.delenv(name)
    result = CliRunner().invoke(app, ["worker", "run", "--env-file", str(tmp_path / "none")])
    assert (called["manager_url"], called["transport"], called["discover_timeout"]) == (
        None, "direct", 60.0)


def test_flashinfer_sampler_defaults_to_off(monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    _apply_env_overrides(NodeSpec(model="m", profile="m@8k"))
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"  # backend child inherits the default


def test_flashinfer_sampler_env_wins_over_default(monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    _apply_env_overrides(NodeSpec(model="m", profile="m@8k"))
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"  # explicit opt-in preserved


# --- WK_MODEL_STORAGE_LOCATION: per-worker override of where the weights live ---------


def test_storage_location_env_overrides_the_fleet_spec(tmp_path, monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    local = tmp_path / "weights"
    local.mkdir()
    (local / "config.json").write_text("{}")
    monkeypatch.setenv("WK_MODEL_STORAGE_LOCATION", f"file://{local}")

    spec = _apply_env_overrides(NodeSpec(model="acme/tiny", profile="tiny@8k"))
    assert spec.model_ref == str(local)      # backend loads from disk
    assert spec.served_name == "tiny@8k"     # clients still address the profile id


def test_storage_location_env_none_falls_back_to_the_hub(tmp_path, monkeypatch):
    """A node without the files opts out of a fleet-wide location."""
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    monkeypatch.setenv("WK_MODEL_STORAGE_LOCATION", "none")
    spec = _apply_env_overrides(
        NodeSpec(model="acme/tiny", profile="tiny@8k", storage_location="file:///nonexistent/models/tiny")
    )
    assert spec.storage_location is None
    assert spec.model_ref == "acme/tiny"


def test_missing_local_weights_fail_before_the_backend_starts(monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    monkeypatch.delenv("WK_MODEL_STORAGE_LOCATION", raising=False)
    with pytest.raises(SystemExit, match="does not exist on this worker"):
        _apply_env_overrides(NodeSpec(model="m", profile="m@8k", storage_location="file:///no/such/dir"))


def test_directory_without_config_json_is_rejected(tmp_path, monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv("WK_MODEL_STORAGE_LOCATION", raising=False)
    with pytest.raises(SystemExit, match="no config.json"):
        _apply_env_overrides(NodeSpec(model="m", profile="m@8k", storage_location=f"file://{empty}"))


def test_bad_storage_location_env_exits_cleanly(monkeypatch):
    from oumigo.config.spec import NodeSpec
    from oumigo.service.worker.coordinator import _apply_env_overrides

    monkeypatch.setenv("WK_MODEL_STORAGE_LOCATION", "s3://bucket/model")
    with pytest.raises(SystemExit, match="WK_MODEL_STORAGE_LOCATION"):
        _apply_env_overrides(NodeSpec(model="m", profile="m@8k"))


# --- retired names (removed in 0.5.5, renamed in 0.5.8): refused rather than ignored -


@pytest.mark.parametrize("name,expected", [
    ("MODEL_NAME", "removed in 0.5.5; choose a model with WK_MODEL_PROFILE"),
    ("MAX_MODEL_LEN", "removed in 0.5.5; choose a profile with that context length"),
    ("OUMIGO_MODEL_PROFILE", "renamed to WK_MODEL_PROFILE"),
    ("MODEL_STORAGE_LOCATION", "renamed to WK_MODEL_STORAGE_LOCATION"),
    ("OUMIGO_MANAGER_TOKEN", "a worker's join token is now WK_JOIN_TOKEN"),
])
def test_retired_variables_stop_the_worker(monkeypatch, name, expected):
    from oumigo.service.worker.coordinator import run_worker

    monkeypatch.setenv(name, "x")
    # Refused before the manager is even looked for.
    with pytest.raises(SystemExit, match=f"{name} is no longer read: {expected}"):
        run_worker("http://127.0.0.1:1", None)


def test_retired_names_are_checked_per_side_and_all_at_once() -> None:
    from oumigo.common.envnames import MANAGER, WORKER, refuse_retired, retired_in_env

    env = {"OUMIGO_CONFIG_FILE": "/m.yaml", "OUMIGO_HOST_ID": "pod-1", "MODEL_NAME": "  "}
    assert retired_in_env(MANAGER, env) == [
        "OUMIGO_CONFIG_FILE is no longer read: renamed to MG_CONFIG_FILE"]
    assert retired_in_env(WORKER, env) == [
        "OUMIGO_HOST_ID is no longer read: renamed to WK_HOST_ID"]  # blank MODEL_NAME: unset
    with pytest.raises(SystemExit, match="(?s)OUMIGO_CONFIG_FILE.*docs/environmental"):
        refuse_retired(MANAGER, env)
    refuse_retired(MANAGER, {"MG_CONFIG_FILE": "/m.yaml"})  # new names pass


def test_cli_refuses_retired_names(monkeypatch, tmp_path) -> None:
    from typer.testing import CliRunner

    from oumigo.cli.main import app

    monkeypatch.setenv("OUMIGO_MANAGER_TOKEN", "old")
    result = CliRunner().invoke(app, ["manager", "token", "list"])
    assert result.exit_code != 0 and "renamed to MG_JOIN_TOKEN" in str(result.exception)
    monkeypatch.delenv("OUMIGO_MANAGER_TOKEN")
    monkeypatch.setattr(os, "environ", dict(os.environ))  # the .env below must not leak
    (tmp_path / ".env").write_text("OUMIGO_TRANSPORT=tunnel\n")
    result = CliRunner().invoke(app, ["worker", "run", "--env-file", str(tmp_path / ".env")])
    assert result.exit_code != 0 and "renamed to WK_TRANSPORT" in str(result.exception)
