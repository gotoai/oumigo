"""Worker identity: host-id + port hashing, persistence, and the env override."""

from __future__ import annotations

from oumigo.common.identity import compute_node_id
from oumigo.service.worker.identity import ENV_HOST_ID, HOST_ID_FILE, resolve_worker_identity


def test_identity_is_stable_and_incarnation_bumps(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(ENV_HOST_ID, raising=False)
    first, inc1, _ = resolve_worker_identity(7001, tmp_path)
    second, inc2, _ = resolve_worker_identity(7001, tmp_path)
    assert first == second            # same host + port -> same node
    assert (inc1, inc2) == (1, 2)     # but a new incarnation each start
    host_id = (tmp_path / HOST_ID_FILE).read_text().strip()
    assert first == compute_node_id(host_id, 7001)


def test_ports_on_one_host_get_distinct_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(ENV_HOST_ID, raising=False)
    a, _, _ = resolve_worker_identity(7001, tmp_path)
    b, _, _ = resolve_worker_identity(7002, tmp_path)
    assert a != b


def test_identity_survives_an_address_change(tmp_path, monkeypatch) -> None:
    # The LAN IP is no longer part of the hash, so a moved host keeps its node_id.
    monkeypatch.delenv(ENV_HOST_ID, raising=False)
    monkeypatch.setattr("oumigo.discovery.get_lan_ip", lambda: "10.0.0.1")
    a, _, _ = resolve_worker_identity(7001, tmp_path)
    monkeypatch.setattr("oumigo.discovery.get_lan_ip", lambda: "192.168.9.9")
    b, _, _ = resolve_worker_identity(7001, tmp_path)
    assert a == b


def test_env_host_id_overrides_the_persisted_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_HOST_ID, "pod-abc")
    node_id, _, _ = resolve_worker_identity(7001, tmp_path)
    assert node_id == compute_node_id("pod-abc", 7001)
    assert not (tmp_path / HOST_ID_FILE).exists()  # nothing minted when overridden
