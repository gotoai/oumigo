"""Smoke tests: the package imports and its core surfaces exist."""

from __future__ import annotations


def test_version_is_exposed() -> None:
    import oumigo

    assert oumigo.__version__


def test_node_states_present() -> None:
    from oumigo.protocol import NodeState

    assert NodeState.FAILED.value == "failed"
