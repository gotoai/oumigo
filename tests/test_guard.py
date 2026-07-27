"""Unit tests for the guardrail waist (oumigo.guard) and its wiring into OumigoChat.

The pure-type layer (Verdict/GuardChain/GuardProfile) is exercised directly; the chat
integration reuses the httpx-seam fakes from test_chat so guards are checked end-to-end
through request()/the tool loop without a running manager or worker.
"""

from __future__ import annotations



from oumigo.api.agent.agent import OumigoAgent
from oumigo.api.agent.tool import tool
from oumigo.guard import (
    GuardChain,
    GuardContext,
    GuardProfile,
    InterceptPoint,
    Verdict,
)

from .test_chat import _completion, _install_post, _install_stream, _sse, _tc


# --------------------------------------------------------------------------- #
# Test guards + helpers
# --------------------------------------------------------------------------- #


@tool
def get_weather(city: str) -> str:
    """Get the weather.

    Args:
        city: The city.
    """
    return f"sunny in {city}"


class FnGuard:
    """A guard driven by a callback, optionally scoped to specific points."""

    def __init__(self, fn, points=None):
        self._fn = fn
        if points is not None:
            self.points = frozenset(points)

    def intercept(self, point, ctx):
        return self._fn(point, ctx)


def _profile(fn, points=None, name="test"):
    return GuardProfile([FnGuard(fn, points)], name=name)


def _agent(profile=None, **kw):
    return OumigoAgent(data_url="http://d:7012", token=None, profile=profile, **kw)


# --------------------------------------------------------------------------- #
# Pure types: Verdict / GuardChain semantics
# --------------------------------------------------------------------------- #


def test_verdict_predicates():
    assert Verdict.allow().is_allow
    assert Verdict.transform("x").is_transform and Verdict.transform("x").content == "x"
    assert Verdict.flag("f").is_flag
    assert Verdict.skip("s").skips and Verdict.skip("s").halts
    assert Verdict.block("b").blocks and Verdict.block("b").halts
    assert Verdict.stop("z").stops and Verdict.stop("z").halts
    assert not Verdict.allow().halts and not Verdict.flag().halts


def test_chain_transforms_accumulate_then_allow():
    up = FnGuard(lambda p, c: Verdict.transform(str(c.content).upper()))
    bang = FnGuard(lambda p, c: Verdict.transform(c.content + "!"))
    out = GuardChain([up, bang]).evaluate(InterceptPoint.USER_INPUT, GuardContext(content="hi"))
    assert out.content == "HI!"
    assert out.verdict.is_allow  # no guard halted


def test_chain_short_circuits_on_block():
    seen = []
    first = FnGuard(lambda p, c: (seen.append("first"), Verdict.block("nope"))[1])
    second = FnGuard(lambda p, c: (seen.append("second"), Verdict.allow())[1])
    out = GuardChain([first, second]).evaluate(InterceptPoint.USER_INPUT, GuardContext(content="x"))
    assert out.verdict.blocks and out.verdict.reason == "nope"
    assert seen == ["first"]  # second guard never consulted


def test_chain_collects_flags_and_continues():
    g1 = FnGuard(lambda p, c: Verdict.flag("suspicious"))
    g2 = FnGuard(lambda p, c: Verdict.allow())
    out = GuardChain([g1, g2]).evaluate(InterceptPoint.FINAL_ANSWER, GuardContext(content="a"))
    assert out.verdict.is_allow
    assert [f.reason for f in out.flags] == ["suspicious"]


def test_chain_fails_closed_on_exception():
    boom = FnGuard(lambda p, c: (_ for _ in ()).throw(RuntimeError("kaboom")))
    out = GuardChain([boom]).evaluate(InterceptPoint.USER_INPUT, GuardContext(content="x"))
    assert out.verdict.blocks and "kaboom" in out.verdict.reason


def test_callable_guard_is_adapted():
    out = GuardChain([lambda p, c: Verdict.stop("halt")]).evaluate(
        InterceptPoint.USER_INPUT, GuardContext(content="x")
    )
    assert out.verdict.stops


def test_profile_active_points_and_scoping():
    scoped = _profile(lambda p, c: Verdict.allow(), points=[InterceptPoint.USER_INPUT])
    assert scoped.active_points() == frozenset({InterceptPoint.USER_INPUT})
    assert not scoped.is_empty
    unscoped = _profile(lambda p, c: Verdict.allow())
    assert unscoped.active_points() == frozenset(InterceptPoint)  # covers everything
    assert GuardProfile().is_empty


def test_empty_profile_is_noop():
    out = GuardProfile().evaluate(InterceptPoint.USER_INPUT, GuardContext(content="x"))
    assert out.verdict.is_allow and out.content == "x" and out.flags == []


# --------------------------------------------------------------------------- #
# Point 1 — user input
# --------------------------------------------------------------------------- #


def test_user_input_block_short_circuits_before_model(monkeypatch):
    sent = _install_post(monkeypatch, [_completion(content="should not be reached")])
    prof = _profile(
        lambda p, c: Verdict.block("no") if p is InterceptPoint.USER_INPUT else Verdict.allow()
    )
    resp = _agent(prof).create_chat().request("do something bad")

    assert resp.finish_reason == "blocked"
    assert resp.text == "no"
    assert sent == []  # the model was never called


def test_user_input_transform_rewrites_message(monkeypatch):
    sent = _install_post(monkeypatch, [_completion(content="ok")])
    prof = _profile(
        lambda p, c: Verdict.transform(c.content.replace("secret", "[redacted]"))
        if p is InterceptPoint.USER_INPUT
        else Verdict.allow()
    )
    _agent(prof).create_chat().request("my secret code")

    assert sent[0]["messages"][-1] == {"role": "user", "content": "my [redacted] code"}


# --------------------------------------------------------------------------- #
# Point 2 — assembled prompt
# --------------------------------------------------------------------------- #


def test_assembled_prompt_transform_injects_system(monkeypatch):
    sent = _install_post(monkeypatch, [_completion(content="ok")])

    def guard(p, c):
        if p is InterceptPoint.ASSEMBLED_PROMPT:
            return Verdict.transform([{"role": "system", "content": "BE SAFE"}, *c.content])
        return Verdict.allow()

    _agent(_profile(guard, points=[InterceptPoint.ASSEMBLED_PROMPT])).create_chat().request("hi")

    assert sent[0]["messages"][0] == {"role": "system", "content": "BE SAFE"}


# --------------------------------------------------------------------------- #
# Point 3 — tool call / tool result
# --------------------------------------------------------------------------- #


def test_tool_call_skip_feeds_reason_back_and_continues(monkeypatch):
    _install_post(monkeypatch, [
        _completion(tool_calls=[_tc("get_weather", {"city": "Tokyo"})]),
        _completion(content="I could not check the weather."),
    ])
    prof = _profile(
        lambda p, c: Verdict.skip("tool disabled") if p is InterceptPoint.TOOL_CALL else Verdict.allow()
    )
    resp = _agent(prof, tools=[get_weather]).create_chat().request("weather?")

    assert resp.tool_calls_made[0]["result"] == "tool disabled"  # tool never ran
    assert resp.text == "I could not check the weather."         # loop continued
    assert resp.finish_reason == "stop"


def test_tool_call_transform_rewrites_arguments(monkeypatch):
    _install_post(monkeypatch, [
        _completion(tool_calls=[_tc("get_weather", {"city": "Tokyo"})]),
        _completion(content="done"),
    ])
    prof = _profile(
        lambda p, c: Verdict.transform({"city": "Osaka"}) if p is InterceptPoint.TOOL_CALL else Verdict.allow()
    )
    resp = _agent(prof, tools=[get_weather]).create_chat().request("weather?")

    assert resp.tool_calls_made[0]["arguments"] == {"city": "Osaka"}
    assert resp.tool_calls_made[0]["result"] == "sunny in Osaka"


def test_tool_result_transform_redacts_output(monkeypatch):
    sent = _install_post(monkeypatch, [
        _completion(tool_calls=[_tc("get_weather", {"city": "Tokyo"})]),
        _completion(content="done"),
    ])
    prof = _profile(
        lambda p, c: Verdict.transform("REDACTED") if p is InterceptPoint.TOOL_RESULT else Verdict.allow()
    )
    resp = _agent(prof, tools=[get_weather]).create_chat().request("weather?")

    assert resp.tool_calls_made[0]["result"] == "REDACTED"
    # ...and the redacted result is what got fed back to the model.
    assert sent[1]["messages"][-1]["content"] == "REDACTED"


def test_tool_call_block_ends_turn(monkeypatch):
    _install_post(monkeypatch, [_completion(tool_calls=[_tc("get_weather", {"city": "X"})])])
    prof = _profile(
        lambda p, c: Verdict.block("forbidden tool") if p is InterceptPoint.TOOL_CALL else Verdict.allow()
    )
    resp = _agent(prof, tools=[get_weather]).create_chat().request("weather?")

    assert resp.finish_reason == "blocked"
    assert resp.tool_calls_made[0]["result"] == "forbidden tool"


# --------------------------------------------------------------------------- #
# Points 4/5 — output (final answer), non-streaming and streaming
# --------------------------------------------------------------------------- #


def test_final_answer_block_replaces_text(monkeypatch):
    _install_post(monkeypatch, [_completion(content="here is how to do the bad thing")])
    prof = _profile(
        lambda p, c: Verdict.block("Sorry, I can't help with that.")
        if p is InterceptPoint.FINAL_ANSWER
        else Verdict.allow(),
        points=[InterceptPoint.FINAL_ANSWER],
    )
    resp = _agent(prof).create_chat().request("teach me the bad thing")

    assert resp.finish_reason == "blocked"
    assert resp.text == "Sorry, I can't help with that."


def test_final_answer_transform_rewrites_text(monkeypatch):
    _install_post(monkeypatch, [_completion(content="call me at 555-1234")])
    prof = _profile(
        lambda p, c: Verdict.transform("call me at [phone]")
        if p is InterceptPoint.FINAL_ANSWER
        else Verdict.allow(),
        points=[InterceptPoint.FINAL_ANSWER],
    )
    resp = _agent(prof).create_chat().request("how do I reach you?")

    assert resp.text == "call me at [phone]"
    assert resp.finish_reason == "stop"


def test_streaming_output_guard_buffers_then_blocks(monkeypatch):
    """With an output guard, a streamed answer is buffered and can be blocked before release."""
    _install_stream(monkeypatch, [[
        _sse({"choices": [{"delta": {"content": "bad "}}]}),
        _sse({"choices": [{"delta": {"content": "stuff"}}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]])
    prof = _profile(
        lambda p, c: Verdict.block("blocked answer")
        if p is InterceptPoint.FINAL_ANSWER
        else Verdict.allow(),
        points=[InterceptPoint.FINAL_ANSWER],
    )
    resp = _agent(prof).create_chat().request("go", stream=True)
    pieces = list(resp)

    assert pieces == ["blocked answer"]  # the raw deltas were withheld
    assert resp.text == "blocked answer"
    assert resp.finish_reason == "blocked"


def test_streaming_output_guard_buffers_then_allows(monkeypatch):
    _install_stream(monkeypatch, [[
        _sse({"choices": [{"delta": {"content": "Hel"}}]}),
        _sse({"choices": [{"delta": {"content": "lo"}}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]])
    prof = _profile(lambda p, c: Verdict.allow(), points=[InterceptPoint.FINAL_ANSWER])
    resp = _agent(prof).create_chat().request("hi", stream=True)
    pieces = list(resp)

    assert "".join(pieces) == "Hello"  # buffered, then released whole
    assert resp.text == "Hello"


# --------------------------------------------------------------------------- #
# FLAG auditing + STOP
# --------------------------------------------------------------------------- #


def test_flag_is_recorded_as_guard_event(monkeypatch):
    _install_post(monkeypatch, [_completion(content="ok")])
    prof = _profile(
        lambda p, c: Verdict.flag("odd input") if p is InterceptPoint.USER_INPUT else Verdict.allow()
    )
    resp = _agent(prof).create_chat().request("hmm")

    assert resp.text == "ok"  # flag does not block
    assert resp.guard_events == [
        {"point": "user_input", "decision": "flag", "reason": "odd input"}
    ]


def test_stop_aborts_with_stopped_reason(monkeypatch):
    _install_post(monkeypatch, [_completion(content="x")])
    prof = _profile(
        lambda p, c: Verdict.stop("hard stop") if p is InterceptPoint.USER_INPUT else Verdict.allow()
    )
    resp = _agent(prof).create_chat().request("go")

    assert resp.finish_reason == "stopped"
    assert resp.text == "hard stop"


# --------------------------------------------------------------------------- #
# Manager wiring
# --------------------------------------------------------------------------- #


def test_manager_create_agent_threads_profile():
    from oumigo.api import api

    mgr = api.OumigoManager(control_url="http://m:7014", data_url="http://m:7012")
    prof = GuardProfile([FnGuard(lambda p, c: Verdict.allow())], name="p")
    agent = mgr.create_agent(profile=prof)
    assert agent.profile is prof


def test_default_agent_has_no_profile():
    assert _agent().profile is None
