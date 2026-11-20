"""oumigo_pods.py: several profiles in one pod — the GPU they need, and their nodes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy" / "pods"))

import oumigo_pods as pods  # noqa: E402

MANAGER = {"catalog": {"allow": ["harrier-oss-v1-0.6b@32k", "harrier-oss-v1-0.6b@8k",
                                 "ruri-v3-310m@8k", "gemma-4-12B-it-qat-w4a16-ct@128k"]}}
PAIR = ["harrier-oss-v1-0.6b@32k", "ruri-v3-310m@8k"]


def test_profile_list_takes_a_string_or_a_list() -> None:
    assert pods.profile_list(None) == []
    assert pods.profile_list("ruri-v3-310m@8k") == ["ruri-v3-310m@8k"]
    assert pods.profile_list("harrier-oss-v1-0.6b@32k, ruri-v3-310m@8k") == PAIR
    assert pods.profile_list(PAIR) == PAIR


def test_profiles_sharing_a_gpu_need_the_sum_of_their_minimums() -> None:
    path = Path("manager.yaml")
    assert pods.profile_min_vram(MANAGER, ["ruri-v3-310m@8k"], path) == 4
    assert pods.profile_min_vram(MANAGER, PAIR, path) == 12
    with pytest.raises(pods.Fail, match="no gpu_memory_target_gib"):
        pods.profile_min_vram(MANAGER, ["ruri-v3-310m@8k", "gemma-4-12B-it-qat-w4a16-ct@128k"],
                              path)
    with pytest.raises(pods.Fail, match="not in catalog.allow"):
        pods.profile_min_vram(MANAGER, ["ruri-v3-310m@8k", "nope@1k"], path)


def test_one_node_per_profile_on_consecutive_ports() -> None:
    from oumigo.common.identity import compute_node_id

    s = object.__new__(pods.Settings)
    s.profiles, s.vllm_port = PAIR, 7001
    assert s.node_ids("embed-1") == [compute_node_id("embed-1", 7001),
                                     compute_node_id("embed-1", 7002)]
    s.profiles = []
    assert s.node_ids("embed-1") == [compute_node_id("embed-1", 7001)]


def test_state_from_before_0_5_6_still_names_its_node() -> None:
    assert pods.state_node_ids({"node_id": "abc"}) == ["abc"]
    assert pods.state_node_ids({"node_ids": ["a", "b"]}) == ["a", "b"]
    assert pods.state_node_ids({}) == []
