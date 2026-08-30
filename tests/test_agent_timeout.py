"""Tests for the agent's wall-clock budget: fleet-declared defaults + turn enforcement.

The data plane is mocked at the httpx seam, and the clock is replaced outright, so a
test that exercises a 90-second budget still runs in microseconds.
"""

from __future__ import annotations

import copy

import httpx
import pytest

import oumigo.api.agent.agent as agent_mod
import oumigo.api.agent.chat as chat_mod
from oumigo.api.agent.agent import OumiGoAgent
from oumigo.service.manager.settings import get_agent_defaults


# --- scaffolding ----------------------------------------------------------------


class _FakeClock:
    """Stands in for `time` inside chat.py; only `monotonic` is ever called."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload

    def read(self):
        return b""


def _completion(content=None, tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish}]}


def _tc(name, args):
    import json as _json
    return {"id": "c1", "type": "function",
            "function": {"name": name, "arguments": _json.dumps(args)}}


def _install_post(monkeypatch, responder):
    """`responder(call_index, timeout)` returns a payload dict, or raises."""
    calls: list[dict] = []

    def fake_post(url, *, json=None, headers=None, timeout=None):
        calls.append({"json": copy.deepcopy(json), "timeout": timeout})
        return _Resp(responder(len(calls) - 1, timeout))

    monkeypatch.setattr(chat_mod.httpx, "post", fake_post)
    return calls


def _clock(monkeypatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(chat_mod, "time", clock)
    return clock


def _agent(**kw) -> OumiGoAgent:
    kw.setdefault("turn_timeout", None)
    kw.setdefault("stall_timeout", None)
    return OumiGoAgent(data_url="http://d:7012", token=None, **kw)


# --- config parsing -------------------------------------------------------------


def test_agent_block_is_parsed() -> None:
    cfg = {"agent": {"turn_timeout": 90.0, "stall_timeout": 120.0}}
    assert get_agent_defaults(cfg) == {"turn_timeout": 90.0, "stall_timeout": 120.0}


def test_missing_block_means_no_limits() -> None:
    assert get_agent_defaults({}) == {"turn_timeout": None, "stall_timeout": None}
    assert get_agent_defaults({"agent": {"turn_timeout": None}})["turn_timeout"] is None


def test_nonsense_values_are_rejected_at_load() -> None:
    with pytest.raises(ValueError, match="agent.turn_timeout"):
        get_agent_defaults({"agent": {"turn_timeout": "soon"}})
    with pytest.raises(ValueError, match="positive"):
        get_agent_defaults({"agent": {"stall_timeout": -1}})


# --- fleet lookup ---------------------------------------------------------------


def test_unset_timeouts_are_fetched_from_the_fleet(monkeypatch) -> None:
    asked: list[str] = []

    def fake_get(url, *, headers=None, timeout=None):
        asked.append(url)
        return _Resp({"turn_timeout": 90.0, "stall_timeout": 45.0})

    monkeypatch.setattr(agent_mod.httpx, "get", fake_get)
    agent = OumiGoAgent(data_url="http://d:7012")

    assert agent.turn_timeout == 90.0
    assert agent.stall_timeout == 45.0
    assert asked == ["http://d:7012/agent-defaults"]

    agent.turn_timeout, agent.stall_timeout      # cached: asked exactly once
    assert len(asked) == 1


def test_explicit_values_win_and_skip_the_lookup(monkeypatch) -> None:
    def boom(*a, **kw):
        raise AssertionError("must not consult the fleet when both values were given")

    monkeypatch.setattr(agent_mod.httpx, "get", boom)
    agent = OumiGoAgent(data_url="http://d:7012", turn_timeout=30.0, stall_timeout=10.0)
    assert (agent.turn_timeout, agent.stall_timeout) == (30.0, 10.0)


def test_explicit_none_means_no_timeout_not_ask_the_fleet(monkeypatch) -> None:
    monkeypatch.setattr(agent_mod.httpx, "get",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("asked")))
    agent = OumiGoAgent(data_url="http://d:7012", turn_timeout=None, stall_timeout=None)
    assert agent.turn_timeout is None and agent.stall_timeout is None


def test_unreachable_manager_falls_back_to_no_timeout(monkeypatch) -> None:
    def fake_get(url, *, headers=None, timeout=None):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(agent_mod.httpx, "get", fake_get)
    agent = OumiGoAgent(data_url="http://d:7012")
    # Inference must not break because an optional config endpoint was unreachable.
    assert agent.turn_timeout is None and agent.stall_timeout is None


def test_partial_declaration_only_fills_the_unset_side(monkeypatch) -> None:
    monkeypatch.setattr(agent_mod.httpx, "get",
                        lambda *a, **kw: _Resp({"turn_timeout": 90.0}))
    agent = OumiGoAgent(data_url="http://d:7012", stall_timeout=5.0)
    assert agent.turn_timeout == 90.0
    assert agent.stall_timeout == 5.0


# --- enforcement ----------------------------------------------------------------


def test_budget_is_handed_to_httpx_as_the_read_timeout(monkeypatch) -> None:
    _clock(monkeypatch)
    calls = _install_post(monkeypatch, lambda i, t: _completion(content="hi"))
    _agent(turn_timeout=90.0, stall_timeout=30.0).create_chat().request("q")

    # The stall limit is smaller than the remaining turn budget, so it governs.
    assert calls[0]["timeout"].read == 30.0
    assert calls[0]["timeout"].connect == chat_mod._CONNECT_TIMEOUT_S


def test_last_round_trip_cannot_overrun_the_turn_deadline(monkeypatch) -> None:
    clock = _clock(monkeypatch)

    def responder(i, timeout):
        clock.advance(80.0)  # the first call eats most of the budget
        if i == 0:
            return _completion(tool_calls=[_tc("noop", {})])
        return _completion(content="done")

    def noop() -> str:
        """A tool that does nothing."""
        return "ok"

    calls = _install_post(monkeypatch, responder)
    agent = _agent(tools=[noop], turn_timeout=90.0, stall_timeout=30.0)
    agent.create_chat().request("q")

    # 10s left of the 90s budget, so the second call gets 10s, not the 30s stall limit.
    assert calls[1]["timeout"].read == pytest.approx(10.0)


def test_no_timeout_configured_leaves_the_read_side_unbounded(monkeypatch) -> None:
    _clock(monkeypatch)
    calls = _install_post(monkeypatch, lambda i, t: _completion(content="hi"))
    _agent().create_chat().request("q")
    assert calls[0]["timeout"].read is None


def test_expiry_between_round_trips_ends_the_turn(monkeypatch) -> None:
    clock = _clock(monkeypatch)

    def responder(i, timeout):
        clock.advance(95.0)  # blows the 90s budget during the first round-trip
        return _completion(tool_calls=[_tc("noop", {})])

    def noop() -> str:
        """A tool that does nothing."""
        return "ok"

    calls = _install_post(monkeypatch, responder)
    resp = _agent(tools=[noop], turn_timeout=90.0).create_chat().request("q")

    assert resp.finish_reason == "timeout"
    assert len(calls) == 1          # the loop did not start another round-trip
    assert "[oumigo] Timed out" in resp.text


def test_partial_answer_survives_a_timeout(monkeypatch) -> None:
    _clock(monkeypatch)

    def responder(i, timeout):
        if i == 0:
            return _completion(tool_calls=[_tc("noop", {})])
        raise httpx.ReadTimeout("no bytes")

    def noop() -> str:
        """A tool that does nothing."""
        return "partial work"

    _install_post(monkeypatch, responder)
    resp = _agent(tools=[noop], turn_timeout=90.0).create_chat().request("q")

    assert resp.finish_reason == "timeout"
    assert resp.tool_calls_made[0]["result"] == "partial work"   # work done is not lost


def test_httpx_timeout_is_not_raised_at_the_caller(monkeypatch) -> None:
    _clock(monkeypatch)

    def responder(i, timeout):
        raise httpx.ReadTimeout("silence")

    _install_post(monkeypatch, responder)
    resp = _agent(turn_timeout=90.0).create_chat().request("q")   # must not raise

    assert resp.finish_reason == "timeout"
    assert resp.text.strip().startswith("[oumigo] Timed out")


def test_timeout_is_visible_to_a_streaming_consumer(monkeypatch) -> None:
    _clock(monkeypatch)
    _install_post(monkeypatch, lambda i, t: (_ for _ in ()).throw(httpx.ReadTimeout("x")))

    resp = _agent(turn_timeout=90.0).create_chat().request("q", stream=False)
    assert "[oumigo] Timed out" in "".join(resp)
