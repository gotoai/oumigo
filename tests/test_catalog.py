"""The model catalog: shipped profiles, operator profiles, and the fleet's allow/default."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from oumigo.config.catalog import (
    SHIPPED_DIR,
    CatalogError,
    FleetCatalog,
    load_profile,
    load_profiles,
)
from oumigo.service.manager.settings import get_agent_defaults, get_vllm_port

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
        "gemma-4-31B-it-qat-w4a16-ct@128k",
        "harrier-oss-v1-0.6b@32k",
        "harrier-oss-v1-0.6b@8k",
        "ruri-v3-310m@8k",
    ]
    assert all(p.tested for p in load_profiles().values())


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
    ({"id": "tiny-model@default"}, "must look like"),         # the removed legacy form
    ({"engine": {"max_model_len": 32768, "gpu_mem": 0.9}}, "gpu_mem"),   # typo: unknown key
    ({"caps": {"max_output_tokens": 0}}, "max_output_tokens"),
    ({"task": "rerank"}, "task"),
    ({"embed": {"dimensions": 768}}, "needs `task: embed`"),
    ({"task": "embed", "caps": {"max_output_tokens": 100}}, "no use for caps.max_output_tokens"),
    ({"task": "embed", "embed": {"prompts": {"Query": "q: "}}}, "lowercase identifiers: Query"),
])
def test_invalid_profiles_are_rejected(tmp_path: Path, bad: dict, match: str) -> None:
    data = {**_profile(), **bad}
    path = tmp_path / "acme" / "tiny-model@32k.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(CatalogError, match=match):
        load_profile(path)


def test_an_embed_profile_carries_its_task_and_prompts(tmp_path: Path) -> None:
    path = _write(tmp_path, "acme", _profile(task="embed", embed={
        "dimensions": 768, "prompts": {"query": "q: ", "document": ""}}))
    profile = load_profile(path)
    assert profile.node_spec(7001).task == "embed"
    assert profile.prompts == {"query": "q: ", "document": ""}
    chat = load_profile(_write(tmp_path, "acme", _profile("chat-model@32k")))
    assert (chat.task, chat.node_spec(7001).task, chat.prompts) == ("chat", "chat", {})


def test_shipped_embedding_profiles_pin_their_pooling_and_prompts() -> None:
    profiles = load_profiles()
    ruri, harrier = profiles["ruri-v3-310m@8k"], profiles["harrier-oss-v1-0.6b@32k"]
    assert ruri.task == harrier.task == "embed"
    # ruri's config.json says CLS, the model is used with mean pooling.
    args = ruri.engine.extra_args
    assert args[args.index("--pooler-config") + 1] == '{"seq_pooling_type":"MEAN"}'
    assert ruri.prompts["query"] == "検索クエリ: " and ruri.prompts["document"] == "検索文書: "
    assert harrier.prompts["document"] == ""
    assert harrier.prompts["query"].endswith("\nQuery: ")
    assert (ruri.embed.dimensions, harrier.embed.dimensions) == (768, 1024)


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
    spec = fleet.default_profile.node_spec(get_vllm_port(config))
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


def test_a_catalog_may_have_no_default() -> None:
    assert FleetCatalog.from_config({"catalog": {"allow": ["Qwen3.5-9B@32k-bnb4"]}}).default is None


@pytest.mark.parametrize("config,match", [
    ({}, "needs a `catalog:` block"),                                  # no open fleet
    ({"provider": "LAN"}, "needs a `catalog:` block"),
    ({"model": {"name": "acme/x"}}, "`model:` block was removed in 0.5.5"),  # no legacy block
    ({"catalog": {"allow": ["Qwen3.5-9B@32k-bnb4"]}, "model": {"name": "acme/x"}},
     "`model:` block was removed"),
])
def test_a_fleet_needs_a_catalog_block(config: dict, match: str) -> None:
    with pytest.raises(CatalogError, match=match):
        FleetCatalog.from_config(config)


def test_an_empty_catalog_cannot_be_built() -> None:
    with pytest.raises(CatalogError, match="at least one"):
        FleetCatalog({}, None)


def test_vllm_port() -> None:
    assert get_vllm_port({}) == 7001
    assert get_vllm_port({"worker": {"vllm_port": 7100}}) == 7100


def test_agent_defaults_prefer_the_profile_then_the_fleet_block(tmp_path: Path) -> None:
    _write(tmp_path, "acme", _profile(agent={"turn_timeout": 30.0}))
    fleet_block = {"agent": {"turn_timeout": 90.0, "stall_timeout": 120.0}}
    with_agent = {"catalog": {"dirs": [str(tmp_path)], "allow": ["tiny-model@32k"],
                              "default": "tiny-model@32k"}, **fleet_block}
    assert get_agent_defaults(with_agent) == {"turn_timeout": 30.0, "stall_timeout": None}
    without = {"catalog": {"allow": ["Qwen3.5-9B@32k-bnb4"], "default": "Qwen3.5-9B@32k-bnb4"},
               **fleet_block}
    assert get_agent_defaults(without) == {"turn_timeout": 90.0, "stall_timeout": 120.0}
