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


def test_spec_refuses_a_profile_the_fleet_does_not_allow() -> None:
    client, _, _ = _manager()
    resp = client.get("/spec", params={"profile": "other@8k"})
    assert resp.status_code == 403
    assert resp.json()["detail"] == ("profile 'other@8k' is not allowed in this fleet; "
                                     "allowed: big-model@128k, small-model@32k")
    # The removed MODEL_NAME query is not a way to choose any more: the default is handed out.
    assert client.get("/spec", params={"model": "acme/big-model"}).json()["profile"] == SMALL


def test_spec_without_a_default_needs_a_choice() -> None:
    client, _, _ = _manager(_catalog(default=None))
    resp = client.get("/spec")
    assert resp.status_code == 403
    assert resp.json()["detail"] == (
        "Model profile cannot be resolved: this fleet has no default; set "
        "OUMIGO_MODEL_PROFILE to one of: big-model@128k, small-model@32k")
    assert client.get("/spec", params={"profile": BIG}).status_code == 200


# --- manager: /register ----------------------------------------------------------------


def test_register_records_the_profile_and_returns_its_spec() -> None:
    client, registry, _ = _manager()
    resp = client.post("/register", json=_reg(profile=BIG)).json()
    assert resp["accepted"] and resp["node_spec"]["profile"] == BIG
    record = registry.get("n1")
    assert (record.profile, record.served_name, record.model) == (BIG, BIG, "acme/big-model")


def test_register_needs_a_profile() -> None:
    client, registry, _ = _manager()
    resp = client.post("/register", json=_reg("old"))
    assert resp.status_code == 422 and registry.get("old") is None


def test_the_manager_takes_the_model_from_its_own_profile() -> None:
    """Weights and task come from the manager's copy of the profile; what an older worker
    still sends about them is ignored."""
    embed = Profile(id="embedder@8k", model="acme/embedder", task="embed")
    client, registry, _ = _manager(FleetCatalog({**_catalog().profiles, embed.id: embed}, SMALL))
    resp = client.post("/register", json=_reg("e", profile=embed.id)).json()
    assert resp["node_spec"]["task"] == "embed" and registry.get("e").task == "embed"
    client.post("/register", json=_reg("x", profile=SMALL, model="acme/other", task="embed"))
    assert (registry.get("x").model, registry.get("x").task) == ("acme/small-model", "chat")


def test_a_refused_registration_does_not_spend_a_single_use_join_token() -> None:
    client, registry, store = _manager(enforced=True)
    _, token = store.create_join_token(uses=1)
    auth = {"Authorization": f"Bearer {token}"}
    for body, code in ((_reg(profile="other@8k"), 403), (_reg(), 422)):  # not allowed / none
        refused = client.post("/register", json=body, headers=auth)
        assert refused.status_code == code and registry.get("n1") is None
    ok = client.post("/register", json=_reg(profile=SMALL), headers=auth)
    assert ok.status_code == 200 and ok.json()["accepted"]


def test_each_worker_gets_its_own_profiles_capacity() -> None:
    catalog = FleetCatalog({
        SMALL: Profile(id=SMALL, model="acme/small-model",
                       engine=EngineSettings(max_concurrent_requests=8)),
        BIG: Profile(id=BIG, model="acme/big-model",
                     engine=EngineSettings(max_concurrent_requests=2)),
    }, SMALL)
    client, registry, _ = _manager(catalog)
    client.post("/register", json=_reg("s", profile=SMALL))
    client.post("/register", json=_reg("b", profile=BIG))                  # not the default's 8
    client.post("/register", json=_reg("t", profile=SMALL, backend="transformer"))
    assert [registry.get(n).max_concurrent_requests for n in "sbt"] == [8, 2, 1]
    # A tunnel worker keeps its capacity plus one spare connection open.
    resp = client.post("/register", json=_reg("u", profile=BIG, transport="tunnel")).json()
    assert resp["tunnel_pool_size"] == 3


def test_run_server_wires_the_catalog_into_both_apps(tmp_path, monkeypatch) -> None:
    """The manager entry point builds the control plane and the router from one catalog."""
    import yaml

    from oumigo.service.manager.control import server

    config = tmp_path / "manager.yaml"
    config.write_text(yaml.safe_dump({
        "catalog": {"allow": ["ruri-v3-310m@8k"], "default": "ruri-v3-310m@8k"},
        "dashboard": {"enabled": False},
        "auth": {"state_dir": str(tmp_path / "state")},
    }))
    apps: dict = {}

    async def _capture(control, router, log_level) -> None:   # instead of serving
        apps.update(control=control[0], router=router[0])

    monkeypatch.setattr(server, "_serve_both", _capture)
    monkeypatch.setattr(server.signal, "signal", lambda *a: None)
    server.run_server("127.0.0.1", 0, None, "LAN", advertise=False, config_file=config)
    with TestClient(apps["control"]) as control:
        assert control.get("/spec").json()["profile"] == "ruri-v3-310m@8k"
    with TestClient(apps["router"]) as router:
        resp = router.post("/v1/embeddings", json={"model": "nope@8k", "input": "x"})
        assert resp.json()["error"]["message"].endswith("Available models: ruri-v3-310m@8k.")


# --- worker side ------------------------------------------------------------------------


def _profile_spec(**kw) -> NodeSpec:
    return NodeSpec(model="acme/small-model", profile=SMALL, max_model_len=32768, **kw)


def test_a_profile_keeps_the_nodes_own_storage(monkeypatch, tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("MODEL_STORAGE_LOCATION", f"file://{tmp_path}")
    spec = coordinator._apply_env_overrides(_profile_spec())
    assert (spec.model, spec.profile, spec.max_model_len) == ("acme/small-model", SMALL, 32768)
    assert spec.local_path == tmp_path


def test_a_worker_quits_when_the_manager_hands_out_no_profile(monkeypatch) -> None:
    for name in ("MODEL_NAME", "MAX_MODEL_LEN", coordinator.ENV_MODEL_PROFILE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(coordinator.client, "fetch_spec", lambda *a, **kw: (None, "LAN"))
    with pytest.raises(SystemExit, match="handed out no model profile; it needs oumigo 0.5.5"):
        coordinator.run_worker("http://manager:7014", None)


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


def test_an_embedding_profile_needs_the_vllm_backend() -> None:
    with pytest.raises(SystemExit, match="embedding model and needs the vLLM backend"):
        coordinator._check_requirements(_profile_spec(task="embed"), "transformer")
    coordinator._check_requirements(_profile_spec(task="embed"), "vllm")


def test_backends_serve_under_the_profile_id() -> None:
    for argv in (build_argv(_profile_spec()), build_hf_argv(_profile_spec())):
        assert argv[argv.index("--served-model-name") + 1] == SMALL


def test_chats_name_the_agents_model() -> None:
    from oumigo.api.agent.agent import OumiGoAgent

    for kwargs, expected in (({}, "any-model"), ({"model": SMALL}, SMALL)):
        agent = OumiGoAgent(data_url="http://d:7012", turn_timeout=None, stall_timeout=None, **kwargs)
        assert agent.create_chat()._payload([], stream=False)["model"] == expected
