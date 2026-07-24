"""Unit tests for the Agent/Chat/request inference layer (oumigo.api.agent.chat).

The data plane is mocked at the httpx seam, so these exercise the whole client-side
surface — the tool loop, streaming/non-streaming parity, history, tool-error handling,
and the manager `create_agent` wiring — without a running manager or worker.
"""

from __future__ import annotations

import copy
import json

import pytest

from oumigo.api import api
from oumigo.api.agent import chat as chat_mod
from oumigo.api.agent.agent import OumigoAgent
from oumigo.api.agent.tool import tool


# --------------------------------------------------------------------------- #
# Tools + fakes
# --------------------------------------------------------------------------- #


@tool
def get_weather(city: str) -> str:
    """Get the weather.

    Args:
        city: The city.
    """
    return f"sunny in {city}"


@tool
def boom(x: int) -> str:
    """Always fails.

    Args:
        x: ignored.
    """
    raise ValueError("kaboom")


def _completion(content=None, tool_calls=None, finish="stop"):
    """An OpenAI-style non-streaming completion body."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
        finish = "tool_calls"
    return {"choices": [{"message": msg, "finish_reason": finish}]}


def _tc(name, args, call_id="call_1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Stream:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def iter_lines(self):
        return iter(self._lines)

    def read(self):
        return b""


def _install_post(monkeypatch, payloads):
    """Serve `payloads` (list of dicts) in order from httpx.post; capture request bodies."""
    sent: list[dict] = []
    it = iter(payloads)

    def fake_post(url, *, json=None, headers=None, timeout=None):
        sent.append(copy.deepcopy(json))  # snapshot: _run keeps mutating the messages list
        return _Resp(next(it))

    monkeypatch.setattr(chat_mod.httpx, "post", fake_post)
    return sent


def _install_stream(monkeypatch, line_batches):
    """Serve one SSE line-list per httpx.stream call, in order."""
    sent: list[dict] = []
    it = iter(line_batches)

    def fake_stream(method, url, *, json=None, headers=None, timeout=None):
        sent.append(copy.deepcopy(json))  # snapshot before _run mutates the messages list
        return _Stream(next(it))

    monkeypatch.setattr(chat_mod.httpx, "stream", fake_stream)
    return sent


def _agent(**kw):
    return OumigoAgent(data_url="http://d:7012", token=None, **kw)


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #


def test_simple_non_streaming_answer(monkeypatch):
    _install_post(monkeypatch, [_completion(content="Hello there.")])
    chat = _agent().create_chat()

    resp = chat.request("hi")

    assert resp.text == "Hello there."
    assert resp.finish_reason == "stop"
    assert resp.tool_calls_made == []
    assert list(resp) == ["Hello there."]  # a consumed response re-yields its full text


def test_non_streaming_tool_loop(monkeypatch):
    """Model asks for a tool, we execute it, feed the result back, model answers."""
    sent = _install_post(monkeypatch, [
        _completion(tool_calls=[_tc("get_weather", {"city": "Tokyo"})]),
        _completion(content="It's sunny in Tokyo."),
    ])
    chat = _agent(tools=[get_weather]).create_chat()

    resp = chat.request("weather in Tokyo?")

    assert resp.text == "It's sunny in Tokyo."
    assert resp.finish_reason == "stop"
    assert resp.tool_calls_made == [
        {"name": "get_weather", "arguments": {"city": "Tokyo"}, "result": "sunny in Tokyo"}
    ]
    # The second request carried the tool result back to the model.
    second = sent[1]["messages"]
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny in Tokyo"}
    # ...and the first request advertised the tool.
    assert sent[0]["tools"][0]["function"]["name"] == "get_weather"


def test_max_iterations_cap(monkeypatch):
    """A model that never stops calling tools is bounded; finish_reason says so."""
    _install_post(monkeypatch, [_completion(tool_calls=[_tc("get_weather", {"city": "X"})])] * 5)
    chat = _agent(tools=[get_weather], max_iterations=2).create_chat()

    resp = chat.request("loop forever")

    assert resp.finish_reason == "max_iterations"
    assert len(resp.tool_calls_made) == 2  # exactly the cap


def test_unknown_tool_is_fed_back_as_error(monkeypatch):
    _install_post(monkeypatch, [
        _completion(tool_calls=[_tc("nope", {})]),
        _completion(content="sorry"),
    ])
    chat = _agent(tools=[get_weather]).create_chat()

    resp = chat.request("call a missing tool")

    assert resp.tool_calls_made[0]["result"] == "Error: unknown tool 'nope'"
    assert resp.text == "sorry"


def test_tool_exception_is_fed_back_not_raised(monkeypatch):
    _install_post(monkeypatch, [
        _completion(tool_calls=[_tc("boom", {"x": 1})]),
        _completion(content="recovered"),
    ])
    chat = _agent(tools=[boom]).create_chat()

    resp = chat.request("trigger the failing tool")

    assert resp.tool_calls_made[0]["result"] == "Error: ValueError: kaboom"
    assert resp.text == "recovered"


# --------------------------------------------------------------------------- #
# History & system
# --------------------------------------------------------------------------- #


def test_history_and_system_are_sent(monkeypatch):
    sent = _install_post(monkeypatch, [
        _completion(content="first answer"),
        _completion(content="second answer"),
    ])
    chat = _agent().create_chat(system="You are terse.", max_history_turns=1)

    chat.request("q1")
    chat.request("q2")

    first_msgs = sent[0]["messages"]
    assert first_msgs[0] == {"role": "system", "content": "You are terse."}
    assert first_msgs[-1] == {"role": "user", "content": "q1"}

    # The second request replays the first exchange after the system prompt.
    second_msgs = sent[1]["messages"]
    assert second_msgs[0]["role"] == "system"
    assert {"role": "user", "content": "q1"} in second_msgs
    assert {"role": "assistant", "content": "first answer"} in second_msgs
    assert second_msgs[-1] == {"role": "user", "content": "q2"}


def test_history_is_trimmed_to_max_turns(monkeypatch):
    sent = _install_post(monkeypatch, [_completion(content=f"a{i}") for i in range(4)])
    chat = _agent().create_chat(max_history_turns=1)

    for i in range(3):
        chat.request(f"q{i}")
    chat.request("final")

    msgs = sent[-1]["messages"]  # only the most recent exchange should survive
    users = [m for m in msgs if m["role"] == "user"]
    assert users[0] == {"role": "user", "content": "q2"}  # q0/q1 trimmed away
    assert msgs[-1] == {"role": "user", "content": "final"}


def test_zero_history_is_stateless(monkeypatch):
    sent = _install_post(monkeypatch, [_completion(content="a"), _completion(content="b")])
    chat = _agent().create_chat(max_history_turns=0)

    chat.request("q1")
    chat.request("q2")

    assert sent[1]["messages"] == [{"role": "user", "content": "q2"}]


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def _sse(obj):
    return "data: " + json.dumps(obj)


def test_streaming_yields_deltas_and_accumulates_text(monkeypatch):
    _install_stream(monkeypatch, [[
        _sse({"choices": [{"delta": {"content": "Hel"}}]}),
        _sse({"choices": [{"delta": {"content": "lo"}}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]])
    chat = _agent().create_chat()

    resp = chat.request("hi", stream=True)
    pieces = list(resp)

    assert pieces == ["Hel", "lo"]
    assert resp.text == "Hello"          # full answer available after consumption
    assert resp.finish_reason == "stop"


def test_streaming_tool_loop_reassembles_calls(monkeypatch):
    """Streamed tool_call deltas are reassembled, executed, then the final answer streams."""
    _install_stream(monkeypatch, [
        [  # turn 1: a tool call streamed across chunks
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}]}}]}),
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"city":'}}]}}]}),
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '"Osaka"}'}}]}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            "data: [DONE]",
        ],
        [  # turn 2: the prose answer
            _sse({"choices": [{"delta": {"content": "Sunny in Osaka."}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ],
    ])
    chat = _agent(tools=[get_weather]).create_chat()

    resp = chat.request("weather?", stream=True)
    pieces = list(resp)

    assert "".join(pieces) == "Sunny in Osaka."   # only the final answer is yielded
    assert resp.tool_calls_made == [
        {"name": "get_weather", "arguments": {"city": "Osaka"}, "result": "sunny in Osaka"}
    ]
    assert resp.finish_reason == "stop"


# --------------------------------------------------------------------------- #
# Agent construction & manager wiring
# --------------------------------------------------------------------------- #


def test_duplicate_tool_names_rejected():
    with pytest.raises(ValueError, match="duplicate tool name"):
        _agent(tools=[get_weather, get_weather])


def test_plain_function_is_wrapped_via_strict_validator():
    """create_agent accepts a bare function and wraps it (strict @tool validation runs)."""
    def add(a: int, b: int) -> int:
        """Add.

        Args:
            a: x.
            b: y.
        """
        return a + b

    agent = _agent(tools=[add])
    assert "add" in agent.tools


def test_contents_must_be_str(monkeypatch):
    _install_post(monkeypatch, [_completion(content="x")])
    chat = _agent().create_chat()
    with pytest.raises(TypeError, match="contents must be a str"):
        chat.request(123)  # type: ignore[arg-type]


def test_manager_create_agent_threads_data_url_and_token():
    mgr = api.OumigoManager(
        control_url="http://m:7014", data_url="http://m:7012", token="secret"
    )
    agent = mgr.create_agent(tools=[get_weather], temperature=0.2, max_tokens=128)

    assert isinstance(agent, OumigoAgent)
    assert agent.data_url == "http://m:7012"
    assert agent.token == "secret"
    assert agent.sampling == {"temperature": 0.2, "max_tokens": 128}
    assert "get_weather" in agent.tools


def test_sampling_defaults_reach_the_payload(monkeypatch):
    sent = _install_post(monkeypatch, [_completion(content="ok")])
    agent = OumigoAgent(data_url="http://d:7012", sampling={"temperature": 0.7})

    agent.create_chat().request("hi")

    assert sent[0]["temperature"] == 0.7
    assert sent[0]["stream"] is False
