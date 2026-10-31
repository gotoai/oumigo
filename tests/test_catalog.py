"""The model catalog: shipped profiles, operator profiles, and the fleet's allow/default."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from oumigo.config.catalog import (
    SHIPPED_DIR,
    CatalogError,
    FleetCatalog,
    context_label,
    load_profile,
    load_profiles,
)
from oumigo.service.manager.settings import build_node_spec, get_agent_defaults, get_vllm_port

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "config_examples" / "catalog"


def _write(tree: Path, vendor: str, profile: dict) -> Path:
    path = tree / vendor / f"{profile['id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(profile))
    return path


def _profile(pid: str = "tiny-model@32k", **extra) -> dict:
    return {"id": pid, "model": "acme/tiny-model", "engine": {"max_model_len": 32768}, **extra}


# --- shipped profiles and templates ----------------------------------------------


@pytest.mark.parametrize("tree", [SHIPPED_DIR, TEMPLATES_DIR], ids=["shipped", "templates"])
def test_every_profile_is_valid_and_filed_under_its_vendor(tree: Path) -> None:
    paths = sorted(tree.rglob("*.yaml"))
    assert paths, f"no profiles under {tree}"
    for path in paths:
        profile = load_profile(path)                 # schema + id == file name
        assert path.parent.name == profile.vendor, path
        assert profile.description, path
        profile.node_spec(7001)                      # maps onto a valid NodeSpec


def test_shipped_catalog_is_the_tested_set() -> None:
    assert sorted(load_profiles()) == [
        "Muse-Glimmer-30B-W8A16@128k-c10",
        "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W8A16@128k-c10",
        "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@256k-c8",
        "Qwen3.5-9B@32k-bnb4",
        "Qwen3.8-27B-W8A16@128k-c10",
        "gemma-4-12B-it-qat-w4a16-ct@128k",
        "gemma-4-31B-it-qat-w4a16-ct@64k",
    ]
    assert all(p.tested for p in load_profiles().values())


@pytest.mark.parametrize("length,label", [(32768, "32k"), (65536, "64k"), (131072, "128k"),
                                          (262144, "256k"), (None, "default")])
def test_context_label(length: int | None, label: str) -> None:
    assert context_label(length) == label


# --- loading rules ---------------------------------------------------------------


def test_operator_profile_overrides_a_shipped_id(tmp_path: Path) -> None:
    shipped = load_profiles()["gemma-4-12B-it-qat-w4a16-ct@128k"]
    mine = shipped.model_dump(exclude_none=True)
    mine["engine"]["gpu_memory_utilization"] = 0.85
    _write(tmp_path, "google", mine)
    assert load_profiles([tmp_path])[shipped.id].engine.gpu_memory_utilization == 0.85


def test_duplicate_id_in_one_tree_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path, "acme", _profile())
    _write(tmp_path, "other", _profile())
    with pytest.raises(CatalogError, match="duplicate profile id"):
        load_profiles([tmp_path], shipped=False)


@pytest.mark.parametrize("bad,match", [
    ({"id": "tiny-model"}, "must look like"),                 # no @<context>
    ({"id": "tiny model@32k"}, "must look like"),
    ({"engine": {"max_model_len": 32768, "gpu_mem": 0.9}}, "gpu_mem"),   # typo: unknown key
    ({"caps": {"max_output_tokens": 0}}, "max_output_tokens"),
])
def test_invalid_profiles_are_rejected(tmp_path: Path, bad: dict, match: str) -> None:
    data = {**_profile(), **bad}
    path = tmp_path / "acme" / "tiny-model@32k.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(CatalogError, match=match):
        load_profile(path)


def test_id_must_match_the_file_name(tmp_path: Path) -> None:
    path = tmp_path / "acme" / "something-else@32k.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(_profile()))
    with pytest.raises(CatalogError, match="must equal the file name"):
        load_profile(path)


def test_missing_catalog_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="not found"):
        load_profiles([tmp_path / "nope"])


# --- the fleet: catalog: block -----------------------------------------------------


def test_catalog_block_allows_and_defaults(tmp_path: Path) -> None:
    _write(tmp_path, "acme", _profile())
    config = {"catalog": {"dirs": [str(tmp_path)],
                          "allow": ["tiny-model@32k", "gemma-4-12B-it-qat-w4a16-ct@128k"],
                          "default": "tiny-model@32k"},
              "worker": {"vllm_port": 7100}}
    fleet = FleetCatalog.from_config(config)
    assert sorted(fleet.profiles) == ["gemma-4-12B-it-qat-w4a16-ct@128k", "tiny-model@32k"]
    spec = build_node_spec(config)
    assert (spec.model, spec.max_model_len, spec.port) == ("acme/tiny-model", 32768, 7100)


@pytest.mark.parametrize("block,match", [
    ({"allow": []}, "at least one"),
    ({"allow": ["no-such@32k"]}, "unknown profiles: no-such@32k"),
    ({"allow": ["Qwen3.5-9B@32k-bnb4"], "default": "gemma-4-12B-it-qat-w4a16-ct@128k"},
     "not in catalog.allow"),
])
def test_bad_catalog_blocks(block: dict, match: str) -> None:
    with pytest.raises(CatalogError, match=match):
        FleetCatalog.from_config({"catalog": block})


def test_catalog_without_default_hands_out_nothing_by_default() -> None:
    config = {"catalog": {"allow": ["Qwen3.5-9B@32k-bnb4"]}}
    assert FleetCatalog.from_config(config).default is None
    assert build_node_spec(config) is None


def test_catalog_and_legacy_model_together_is_an_error() -> None:
    with pytest.raises(CatalogError, match="both"):
        FleetCatalog.from_config({"catalog": {"allow": ["Qwen3.5-9B@32k-bnb4"]},
                                  "model": {"name": "acme/x"}})


# --- the fleet: legacy model: block -----------------------------------------------


def test_legacy_model_block_becomes_a_one_profile_catalog() -> None:
    config = {"model": {"name": "acme/tiny-model", "port": 7005, "max_model_len": 65536,
                        "max_output_tokens": 512, "extra_args": ["--x"]}}
    fleet = FleetCatalog.from_config(config)
    assert list(fleet.profiles) == ["tiny-model@64k"] and fleet.default == "tiny-model@64k"
    spec = build_node_spec(config)
    # The legacy block's own defaults are kept (0.80 memory, 4 concurrent requests).
    assert (spec.model, spec.port, spec.max_model_len, spec.max_output_tokens) == (
        "acme/tiny-model", 7005, 65536, 512)
    assert (spec.gpu_memory_utilization, spec.max_concurrent_requests) == (0.80, 4)
    assert spec.extra_args == ["--x"]


def test_legacy_without_max_model_len_and_empty_config() -> None:
    assert FleetCatalog.from_config({"model": {"name": "acme/m"}}).default == "m@default"
    assert FleetCatalog.from_config({}).default is None
    assert build_node_spec({}) is None


def test_vllm_port_precedence() -> None:
    assert get_vllm_port({}) == 7001
    assert get_vllm_port({"model": {"port": 7005}}) == 7005
    assert get_vllm_port({"worker": {"vllm_port": 7100}, "model": {"port": 7005}}) == 7100


def test_agent_defaults_prefer_the_profile_then_the_fleet_block(tmp_path: Path) -> None:
    _write(tmp_path, "acme", _profile(agent={"turn_timeout": 30.0}))
    fleet_block = {"agent": {"turn_timeout": 90.0, "stall_timeout": 120.0}}
    with_agent = {"catalog": {"dirs": [str(tmp_path)], "allow": ["tiny-model@32k"],
                              "default": "tiny-model@32k"}, **fleet_block}
    assert get_agent_defaults(with_agent) == {"turn_timeout": 30.0, "stall_timeout": None}
    without = {"catalog": {"allow": ["Qwen3.5-9B@32k-bnb4"], "default": "Qwen3.5-9B@32k-bnb4"},
               **fleet_block}
    assert get_agent_defaults(without) == {"turn_timeout": 90.0, "stall_timeout": 120.0}
