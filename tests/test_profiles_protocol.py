"""Workers choose a catalog profile: /spec and /register on the manager, the worker's
own checks, and the name its backend serves under."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from oumigo.config.catalog import EngineSettings, FleetCatalog, Profile, Requires
from oumigo.config.spec import NodeSpec
from oumigo.service.manager.control.registry import Registry
from oumigo.service.manager.control.server import create_app
from oumigo.service.manager.credentials import Authenticator, AuthStore
from oumigo.service.worker import coordinator
from oumigo.service.worker.supervisor import build_argv, build_hf_argv

SMALL, BIG = "small-model@32k", "big-model@128k"


def _catalog(default: str | None = SMALL) -> FleetCatalog:
    return FleetCatalog({
        SMALL: Profile(id=SMALL, model="acme/small-model",
                       engine=EngineSettings(max_model_len=32768)),
        BIG: Profile(id=BIG, model="acme/big-model", engine=EngineSettings(max_model_len=131072),
                     requires=Requires(min_vram_gib=40)),
    }, default)


def _manager(catalog: FleetCatalog | None = None, enforced: bool = False):
    store = AuthStore()
    auth = Authenticator(store, None, wan=enforced)
    registry = Registry()
    app = create_app(registry, None, auth=auth, catalog=catalog or _catalog(), vllm_port=7100)
    return TestClient(app), registry, store


def _reg(node_id: str = "n1", **extra) -> dict:
    return {"node_id": node_id, "address": "10.0.0.5", "vllm_port": 7100, **extra}


# --- manager: /spec ------------------------------------------------------------------


def test_spec_hands_out_the_chosen_profile_or_the_default() -> None:
    client, _, _ = _manager()
    big = client.get("/spec", params={"profile": BIG}).json()
    assert big["profile"] == BIG
    assert (big["node_spec"]["model"], big["node_spec"]["max_model_len"]) == ("acme/big-model", 131072)
    assert big["node_spec"]["min_vram_gib"] == 40 and big["node_spec"]["port"] == 7100
    assert client.get("/spec").json()["profile"] == SMALL                       # the default
    assert client.get("/spec", params={"model": "acme/big-model"}).json()["profile"] == BIG


@pytest.mark.parametrize("params,match", [
    ({"profile": "other@8k"}, "not allowed in this fleet; allowed: big-model@128k, small-model@32k"),
    ({"model": "acme/unknown"}, "matches 0 allowed profiles"),
])
def test_spec_refuses_choices_the_fleet_does_not_allow(params: dict, match: str) -> None:
    client, _, _ = _manager()
    resp = client.get("/spec", params=params)
    assert resp.status_code == 403 and match in resp.json()["detail"]


def test_spec_without_a_default_needs_a_choice() -> None:
    client, _, _ = _manager(_catalog(default=None))
    resp = client.get("/spec")
    assert resp.status_code == 403 and "no default profile" in resp.json()["detail"]
    assert client.get("/spec", params={"profile": BIG}).status_code == 200


# --- manager: /register ----------------------------------------------------------------


def test_register_records_the_profile_and_returns_its_spec() -> None:
    client, registry, _ = _manager()
    resp = client.post("/register", json=_reg(model="acme/big-model", profile=BIG)).json()
    assert resp["accepted"] and resp["node_spec"]["profile"] == BIG
    assert registry.get("n1").profile == BIG and registry.get("n1").served_name == BIG


def test_a_pre_05_worker_is_refused_with_an_upgrade_hint() -> None:
    client, registry, _ = _manager()
    resp = client.post("/register", json=_reg("old", model="acme/big-model"))  # no profile field
    assert resp.status_code == 403 and "upgrade the worker to oumigo 0.5" in resp.json()["detail"]
    assert registry.get("old") is None


def test_register_refuses_a_profile_whose_weights_differ() -> None:
    client, _, _ = _manager()
    resp = client.post("/register", json=_reg(model="acme/small-model", profile=BIG))
    assert resp.status_code == 403 and "needs 'acme/big-model'" in resp.json()["detail"]


def test_a_refused_profile_does_not_spend_a_single_use_join_token() -> None:
    client, registry, store = _manager(enforced=True)
    _, token = store.create_join_token(uses=1)
    auth = {"Authorization": f"Bearer {token}"}
    for body in (_reg(profile="other@8k"), _reg()):          # not allowed / pre-0.5 worker
        refused = client.post("/register", json=body, headers=auth)
        assert refused.status_code == 403 and registry.get("n1") is None
    ok = client.post("/register", json=_reg(model="acme/small-model", profile=SMALL), headers=auth)
    assert ok.status_code == 200 and ok.json()["accepted"]


def test_open_fleet_accepts_the_workers_own_profile() -> None:
    store = AuthStore()
    registry = Registry()
    app = create_app(registry, None, auth=Authenticator(store, None, wan=False),
                     catalog=FleetCatalog({}, None))
    client = TestClient(app)
    assert client.get("/spec").json()["node_spec"] is None
    client.post("/register", json=_reg(model="acme/x", profile="x@8k"))
    assert registry.get("n1").served_name == "x@8k"


# --- worker side ------------------------------------------------------------------------


def _profile_spec(**kw) -> NodeSpec:
    return NodeSpec(model="acme/small-model", profile=SMALL, max_model_len=32768, **kw)


def test_a_profile_cannot_be_reshaped_by_max_model_len(monkeypatch) -> None:
    monkeypatch.setenv("MAX_MODEL_LEN", "8192")
    with pytest.raises(SystemExit, match="context length is part of the profile"):
        coordinator._apply_env_overrides(_profile_spec())
    monkeypatch.setenv("MAX_MODEL_LEN", "32768")                 # the same value is fine
    assert coordinator._apply_env_overrides(_profile_spec()).max_model_len == 32768


def test_a_profile_ignores_model_name_but_keeps_node_storage(monkeypatch, tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.delenv("MAX_MODEL_LEN", raising=False)
    monkeypatch.setenv("MODEL_NAME", "acme/something-else")
    monkeypatch.setenv("MODEL_STORAGE_LOCATION", f"file://{tmp_path}")
    spec = coordinator._apply_env_overrides(_profile_spec())
    assert (spec.model, spec.profile) == ("acme/small-model", SMALL)
    assert spec.local_path == tmp_path


def test_open_fleet_worker_names_its_own_profile(monkeypatch) -> None:
    for name in ("MAX_MODEL_LEN", "MODEL_STORAGE_LOCATION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_NAME", "acme/tiny-model")
    monkeypatch.setenv("MAX_MODEL_LEN", "16384")
    spec = coordinator._apply_env_overrides(None)
    assert (spec.model, spec.profile, spec.max_model_len) == ("acme/tiny-model", "tiny-model@16k", 16384)


def test_requirements_stop_a_worker_that_cannot_run_the_profile(monkeypatch) -> None:
    monkeypatch.setattr(coordinator, "probe_gpu0_vram_gib", lambda: 15.6)
    with pytest.raises(SystemExit, match="at least 40 GiB; GPU 0 has 15.6 GiB"):
        coordinator._check_requirements(_profile_spec(min_vram_gib=40), "vllm")
    coordinator._check_requirements(_profile_spec(min_vram_gib=15), "vllm")
    monkeypatch.setattr(coordinator, "probe_gpu0_vram_gib", lambda: None)   # unknown: skip
    coordinator._check_requirements(_profile_spec(min_vram_gib=40), "vllm")


def test_vllm_version_requirement(monkeypatch) -> None:
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.27.1")
    with pytest.raises(SystemExit, match="needs vLLM >=0.28; this worker has 0.27.1"):
        coordinator._check_requirements(_profile_spec(requires_vllm=">=0.28"), "vllm")
    coordinator._check_requirements(_profile_spec(requires_vllm=">=0.28"), "transformer")


def test_backends_serve_under_the_profile_id() -> None:
    for argv in (build_argv(_profile_spec()), build_hf_argv(_profile_spec())):
        assert argv[argv.index("--served-model-name") + 1] == SMALL
    # Without a profile (a pre-0.5 manager) the Hub id is served as before.
    assert "--served-model-name" not in build_argv(NodeSpec(model="acme/tiny"))


def test_chats_name_the_agents_model() -> None:
    from oumigo.api.agent.agent import OumiGoAgent

    for kwargs, expected in (({}, "any-model"), ({"model": SMALL}, SMALL)):
        agent = OumiGoAgent(data_url="http://d:7012", turn_timeout=None, stall_timeout=None, **kwargs)
        assert agent.create_chat()._payload([], stream=False)["model"] == expected
