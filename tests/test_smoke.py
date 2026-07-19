"""Smoke tests: the package imports and its core surfaces exist."""

from __future__ import annotations


def test_version_is_exposed() -> None:
    import oumigo

    assert oumigo.__version__


def test_node_states_present() -> None:
    from oumigo.protocol import NodeState, RunState

    assert NodeState.FAILED.value == "failed"
    assert NodeState.INITIALIZING.value == "initializing"
    assert NodeState.SERVING.value == "serving"
    assert {s.value for s in RunState} == {"idle", "executing"}


def test_heartbeat_carries_both_axes() -> None:
    from oumigo.protocol.messages import HeartbeatRequest
    from oumigo.protocol.states import NodeState, RunState

    hb = HeartbeatRequest(node_id="n", node_state=NodeState.SERVING, run_state=RunState.IDLE)
    assert hb.node_state is NodeState.SERVING
    # run_state is optional (None outside SERVING/DRAINING)
    assert HeartbeatRequest(node_id="n", node_state=NodeState.REGISTERING).run_state is None


def test_register_response_can_carry_node_spec() -> None:
    from oumigo.config.spec import NodeSpec
    from oumigo.protocol.messages import RegisterResponse

    spec = NodeSpec(model="acme/tiny")
    resp = RegisterResponse(accepted=True, node_id="n", node_spec=spec)
    # round-trips over the wire
    assert RegisterResponse.model_validate(resp.model_dump()).node_spec.model == "acme/tiny"
