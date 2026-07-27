"""The Chat tier — a stateful conversation that runs one turn (and its tool loop) at a time.

An :class:`OumigoChat` is spawned by :meth:`oumigo.api.agent.agent.OumigoAgent.create_chat`.
It accumulates history and is *not* thread-safe: use one chat per session.
:meth:`OumigoChat.request` runs **one user turn to completion**, returning an
:class:`~oumigo.api.agent.response.OumigoResponse`.

Everything ultimately becomes an OpenAI-style ``POST /v1/chat/completions`` against the
manager's data plane (``data_url``), which the router proxies to a SERVING worker. When
the model asks to call a tool, the loop here executes the matching Python callback, feeds
the result back, and continues — until the model returns prose or the iteration cap is hit.

**Guardrails.** When the agent carries a non-empty :class:`oumigo.guard.GuardProfile`, the
loop consults it at the five intercept points — ``USER_INPUT`` (in :meth:`request`/:meth:`_run`),
``ASSEMBLED_PROMPT`` (every round-trip, in :meth:`_complete_turn`), ``TOOL_CALL`` / ``TOOL_RESULT``
(around :meth:`_execute_tool`), and ``ASSISTANT_TURN`` / ``FINAL_ANSWER`` (model output). A
``TRANSFORM`` rewrites the content in flight; ``SKIP`` on a tool call feeds its reason back to
the model; ``BLOCK`` ends the turn (``finish_reason="blocked"``) and ``STOP`` aborts the request
(``finish_reason="stopped"``), surfacing the verdict's reason as the answer text. Output guards
(``ASSISTANT_TURN`` / ``FINAL_ANSWER``) force a streamed turn to **buffer** (collect, evaluate,
then release) instead of passing deltas through live. An empty/absent profile is a strict no-op:
every guard branch is gated on :attr:`_guarding`, so the request path is byte-for-byte the
original one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING, Any

import httpx

from oumigo.api.agent.response import OumigoResponse
from oumigo.guard import GuardContext, InterceptPoint, OUTPUT_POINTS, Verdict

if TYPE_CHECKING:  # avoid a runtime import cycle: agent.py imports this module
    from oumigo.api.agent.agent import OumigoAgent

log = logging.getLogger("oumigo.api.agent.chat")

# The router overwrites the request's `model` with the fleet's real model name, so the
# value we send is a placeholder that only has to satisfy the OpenAI schema.
_MODEL_PLACEHOLDER = "oumigo"

# No total read timeout — a long generation must not be cut off; cap only the connect.
_TIMEOUT = httpx.Timeout(None, connect=10.0)

# Surfaced as the answer when a guard blocks/stops without giving a reason.
_DEFAULT_BLOCK_TEXT = "This request was blocked by a guardrail policy."


class OumigoChat:
    """A stateful conversation. Accumulates history across calls; single-threaded."""

    def __init__(
        self,
        agent: OumigoAgent,
        *,
        system: str | None = None,
        max_history_turns: int = 3,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self._agent = agent
        self._system = system
        self._max_history_turns = max(0, int(max_history_turns))
        # `history` is the rehydration trust boundary: only user/assistant turns are accepted
        # (see _normalize_history), so a store/blob can't inject a system prompt or tool result.
        self._history: list[dict[str, Any]] = _normalize_history(history)
        self._trim_history()
        self._url = f"{agent.data_url}/v1/chat/completions"
        self._headers = {"Authorization": f"Bearer {agent.token}"} if agent.token else {}
        # Guardrail fast path: everything below is skipped unless a non-empty profile is set,
        # keeping the unguarded request path identical to the original.
        profile = agent.profile
        self._profile = profile
        self._guarding = profile is not None and not profile.is_empty
        # A streamed turn must buffer (not pass deltas through live) only when an output guard
        # is registered — otherwise streaming keeps its original low-latency behavior.
        self._buffer_output = (
            self._guarding
            and profile is not None
            and bool(profile.active_points() & OUTPUT_POINTS)
        )

    @property
    def history(self) -> list[dict[str, str]]:
        """A copy of the carried conversation — the ``{"role","content"}`` user/assistant
        turns (most recent ``max_history_turns`` exchanges). Persist this per session and
        pass it back to :meth:`OumigoAgent.create_chat` to rehydrate the chat on a later,
        stateless request. Never includes the system prompt, tool calls/results, or reasoning.
        """
        return [dict(m) for m in self._history]

    # -- public ------------------------------------------------------------- #

    def request(self, contents: str, stream: bool = False) -> OumigoResponse:
        """Run one user turn to completion.

        Assembles ``[system?] + recent history + {"role": "user", "content": contents}``,
        POSTs to the data plane, and runs the agent loop (execute tools, feed results back,
        repeat) until the model returns prose or ``max_iterations`` model round-trips are
        reached (``finish_reason="max_iterations"``). The (user, final-answer) exchange is
        appended to this chat's history.

        Args:
            contents: The user's message (plain string).
            stream: If True, return a response you iterate for final-answer text deltas;
                if False, the returned response's ``.text`` is already complete.
        """
        if not isinstance(contents, str):
            raise TypeError(f"contents must be a str, got {type(contents).__name__}")

        messages = self._build_messages(contents)
        resp = OumigoResponse()
        resp._gen = self._run(resp, contents, messages, stream)
        if not stream:
            resp.consume()
        return resp

    # -- the loop ----------------------------------------------------------- #

    def _run(
        self,
        resp: OumigoResponse,
        user_contents: str,
        messages: list[dict[str, Any]],
        stream: bool,
    ) -> Iterator[tuple[str, str]]:
        """Drive the tool loop, yielding tagged ``(kind, delta)`` events.

        ``kind`` is ``"answer"`` (final-answer text) or ``"reasoning"`` (reasoning_content);
        :class:`OumigoResponse` reshapes these for ``__iter__`` (answer only) and
        :meth:`OumigoResponse.stream` (both). Runs up to ``max_iterations`` model round-trips.
        Each turn: get the assistant message; if it requests tools, execute them, append their
        results, and loop; else it's the final answer and we stop. On completion, record it in
        history. When a profile is active, the guard chain is consulted at each intercept point
        and a halting verdict ends the loop early with ``finish_reason`` ``"blocked"``/``"stopped"``.
        """
        # POINT 1 — user input: transform/flag, or block/stop before any model call.
        contents, verdict, flags = self._guard(InterceptPoint.USER_INPUT, user_contents)
        self._record_flags(resp, InterceptPoint.USER_INPUT, flags)
        if verdict.halts:
            yield from self._halt(resp, verdict)
            resp.finish_reason = "stopped" if verdict.stops else "blocked"
            self._remember(user_contents, resp.text)
            return
        if contents != user_contents:  # a transform rewrote the user's message in flight
            user_contents = contents if isinstance(contents, str) else user_contents
            messages[-1] = {"role": "user", "content": user_contents}

        reason = "max_iterations"
        for _ in range(self._agent.max_iterations):
            assistant, finish, answer = yield from self._complete_turn(resp, messages, stream)
            if assistant is None:  # POINT 2 (assembled prompt) halted the turn
                reason = finish or "blocked"
                break
            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []

            if self._buffer_output:
                # Output guards ran nothing inline; evaluate and release this turn now.
                halt_reason = yield from self._guard_output(
                    resp, messages, answer, bool(tool_calls)
                )
                if halt_reason:
                    reason = halt_reason
                    break

            if not tool_calls:
                reason = finish or "stop"
                break

            halted = False
            for tc in tool_calls:
                result, tverdict = self._execute_tool(resp, tc)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.get("id"), "content": result}
                )
                if tverdict is not None and (tverdict.blocks or tverdict.stops):
                    reason = "stopped" if tverdict.stops else "blocked"
                    halted = True
                    break
            if halted:
                break
        resp.finish_reason = reason
        self._remember(user_contents, resp.text)

    def _complete_turn(
        self, resp: OumigoResponse, messages: list[dict[str, Any]], stream: bool
    ) -> Generator[tuple[str, str], None, tuple[dict[str, Any] | None, str | None, str]]:
        """One model round-trip. Yields ``(kind, delta)`` events; returns ``(assistant, finish, answer)``.

        ``assistant`` is ``None`` when POINT 2 (assembled prompt) halted before the call.
        ``answer`` is this turn's prose; in ``_buffer_output`` mode it is *not* yet committed
        to ``resp.text`` or yielded — :meth:`_run` does that after the output guard runs.
        """
        # POINT 2 — assembled prompt: screen (and possibly rewrite) the outgoing messages.
        pverdict = self._guard_prompt(resp, messages)
        if pverdict.halts:
            yield from self._halt(resp, pverdict)
            return None, ("stopped" if pverdict.stops else "blocked"), ""

        if stream:
            assistant, finish, answer = yield from self._stream_turn(resp, messages)
            return assistant, finish, answer

        data = self._post_json(messages)
        resp.raw = data
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        reasoning = msg.get("reasoning_content") or ""
        resp._note_reasoning(reasoning)  # output-only; never echoed back
        if reasoning:
            yield "reasoning", reasoning
        assistant = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            assistant["tool_calls"] = msg["tool_calls"]
        content = msg.get("content") or ""
        if content and not self._buffer_output:
            resp.text += content
            yield "answer", content
        return assistant, choice.get("finish_reason"), content

    def _stream_turn(
        self, resp: OumigoResponse, messages: list[dict[str, Any]]
    ) -> Generator[tuple[str, str], None, tuple[dict[str, Any], str | None, str]]:
        """A streaming round-trip: yield ``(kind, delta)`` events, assemble ``(assistant, finish, answer)``.

        In ``_buffer_output`` mode the answer deltas are accumulated but withheld (neither
        committed to ``resp.text`` nor yielded) so the output guard can screen the full turn
        first; reasoning still streams live. Otherwise deltas pass through as before.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish: str | None = None

        for event in self._post_stream(messages):
            resp.raw = event
            choice = (event.get("choices") or [{}])[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
                yield "reasoning", delta["reasoning_content"]
            if delta.get("content"):
                content_parts.append(delta["content"])
                if not self._buffer_output:
                    resp.text += delta["content"]
                    yield "answer", delta["content"]
            for tcd in delta.get("tool_calls") or []:
                _accumulate_tool_call(tool_calls, tcd)

        resp._note_reasoning("".join(reasoning_parts))  # commit this turn's reasoning (turn-separated)
        answer = "".join(content_parts)
        assistant: dict[str, Any] = {"role": "assistant", "content": answer or None}
        if tool_calls:
            assistant["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        return assistant, finish, answer

    def _execute_tool(
        self, resp: OumigoResponse, tc: dict[str, Any]
    ) -> tuple[str, Verdict | None]:
        """Run one requested tool, returning ``(result, halt_verdict)``.

        ``result`` is the string fed back to the model; ``halt_verdict`` is the ``BLOCK``/
        ``STOP`` that ended the call, or ``None``. Never raises: an unknown tool, unparseable
        arguments, or an exception in the tool body all become an ``"Error: ..."`` string so
        the model can recover and the loop continues. When guards are active, the tool's
        arguments are screened before execution (``TOOL_CALL``) and its result after
        (``TOOL_RESULT``).
        """
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments") or "{}"
        tool = self._agent.tools.get(name)

        args: Any = raw_args
        halt: Verdict | None = None
        if tool is None:
            result = f"Error: unknown tool {name!r}"
        else:
            try:
                args = json.loads(raw_args) if raw_args else {}
            except ValueError as exc:
                result = f"Error: could not parse arguments ({exc})"
            else:
                args, result, halt = self._invoke_guarded(resp, tool, name, args)

        resp.tool_calls_made.append({"name": name, "arguments": args, "result": result})
        return result, halt

    def _invoke_guarded(
        self, resp: OumigoResponse, tool: Any, name: str, args: dict[str, Any]
    ) -> tuple[dict[str, Any], str, Verdict | None]:
        """Guard (POINT 3 pre), invoke, then guard (POINT 3 post) one tool call.

        Returns ``(args, result, halt_verdict)``. A ``SKIP`` short-circuits execution and feeds
        the reason back as the result (loop continues); ``BLOCK``/``STOP`` do the same but also
        end the turn/request. With no active guards this is a plain invoke — behavior unchanged.
        """
        # POINT 3 pre — tool call arguments.
        g_args, verdict, flags = self._guard(InterceptPoint.TOOL_CALL, args, tool_name=name)
        self._record_flags(resp, InterceptPoint.TOOL_CALL, flags)
        if verdict.halts:
            reason = verdict.reason or f"Tool {name!r} was blocked by a guardrail."
            return args, reason, (None if verdict.skips else verdict)
        if isinstance(g_args, dict):
            args = g_args

        try:
            out = tool.invoke(**args)
            result = out if isinstance(out, str) else json.dumps(out, default=str)
        except Exception as exc:  # noqa: BLE001 - surface to the model, don't crash
            log.warning("tool %s raised: %s", name, exc)
            return args, f"Error: {type(exc).__name__}: {exc}", None

        # POINT 3 post — tool result.
        g_result, rverdict, rflags = self._guard(InterceptPoint.TOOL_RESULT, result, tool_name=name)
        self._record_flags(resp, InterceptPoint.TOOL_RESULT, rflags)
        if rverdict.halts:
            reason = rverdict.reason or f"The result of tool {name!r} was withheld by a guardrail."
            return args, reason, (None if rverdict.skips else rverdict)
        if isinstance(g_result, str):
            result = g_result
        return args, result, None

    # -- guard chain -------------------------------------------------------- #

    def _guard(
        self,
        point: InterceptPoint,
        content: Any,
        *,
        tool_name: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, Verdict, list[Verdict]]:
        """Evaluate the chain at ``point``; returns ``(content_after_transforms, verdict, flags)``.

        Strict no-op when no profile is active — the ``ALLOW`` fast path keeps the unguarded
        request identical to the original.
        """
        if self._profile is None or self._profile.is_empty:
            return content, Verdict.allow(), []
        ctx = GuardContext(
            content=content, tool_name=tool_name, messages=messages, system=self._system
        )
        outcome = self._profile.evaluate(point, ctx)
        return outcome.content, outcome.verdict, outcome.flags

    def _guard_prompt(self, resp: OumigoResponse, messages: list[dict[str, Any]]) -> Verdict:
        """POINT 2 — screen the outgoing ``messages``, rewriting them in place on a transform."""
        if not self._guarding:
            return Verdict.allow()
        new_messages, verdict, flags = self._guard(
            InterceptPoint.ASSEMBLED_PROMPT, messages, messages=messages
        )
        self._record_flags(resp, InterceptPoint.ASSEMBLED_PROMPT, flags)
        if isinstance(new_messages, list) and new_messages is not messages:
            messages[:] = new_messages  # in-place so _payload sees the rewrite
        return verdict

    def _guard_output(
        self,
        resp: OumigoResponse,
        messages: list[dict[str, Any]],
        answer: str,
        has_tool_calls: bool,
    ) -> Generator[tuple[str, str], None, str]:
        """Screen a buffered turn's output, then release it. Returns a halt reason or ``""``.

        Called only in ``_buffer_output`` mode. Evaluates ``ASSISTANT_TURN`` on an intermediate
        (tool-calling) turn and ``FINAL_ANSWER`` on the final answer, commits the resolved text
        to ``resp.text`` and yields it, and keeps the recorded assistant message in sync.
        """
        point = InterceptPoint.ASSISTANT_TURN if has_tool_calls else InterceptPoint.FINAL_ANSWER
        text, verdict, flags = self._guard(point, answer, messages=messages)
        self._record_flags(resp, point, flags)
        if verdict.halts:
            yield from self._halt(resp, verdict)
            messages[-1] = {**messages[-1], "content": resp.text}
            return "stopped" if verdict.stops else "blocked"
        if not isinstance(text, str):
            text = answer
        if text:
            resp.text += text
            yield "answer", text
        if text != answer:  # a transform rewrote the turn; keep history/model view consistent
            messages[-1] = {**messages[-1], "content": text}
        return ""

    def _halt(self, resp: OumigoResponse, verdict: Verdict) -> Iterator[tuple[str, str]]:
        """Surface a blocking/stopping verdict as the answer text (one ``"answer"`` event)."""
        text = verdict.reason or _DEFAULT_BLOCK_TEXT
        resp.text += text
        log.info("guard %s: %s", verdict.decision.value, text)
        yield "answer", text

    def _record_flags(
        self, resp: OumigoResponse, point: InterceptPoint, flags: list[Verdict]
    ) -> None:
        """Record ``FLAG`` verdicts for audit (logged, and kept on the response)."""
        for f in flags:
            log.info("guard FLAG at %s: %s", point.value, f.reason)
            resp.guard_events.append(
                {"point": point.value, "decision": f.decision.value, "reason": f.reason}
            )

    # -- history & payload (the guardrail seam) ----------------------------- #

    def _build_messages(self, contents: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.extend(self._history)
        messages.append({"role": "user", "content": contents})
        return messages

    def _remember(self, user_contents: str, answer: str) -> None:
        """Append the (user, final-answer) exchange, trimmed to ``max_history_turns``."""
        if self._max_history_turns <= 0:
            return
        self._history.append({"role": "user", "content": user_contents})
        self._history.append({"role": "assistant", "content": answer})
        self._trim_history()

    def _trim_history(self) -> None:
        """Keep only the most recent ``max_history_turns`` (user, assistant) exchanges."""
        keep = 2 * self._max_history_turns
        if keep == 0:
            self._history = []
        elif len(self._history) > keep:
            self._history = self._history[-keep:]

    def _payload(self, messages: list[dict[str, Any]], stream: bool) -> dict[str, Any]:
        """Build the request body — the seam every model call passes through.

        Guardrails do not intercept here: the ``messages`` reaching this method have already
        been screened at the intercept points (``ASSEMBLED_PROMPT`` in :meth:`_complete_turn`
        rewrites them in place before the call). This stays a pure body builder.
        """
        body: dict[str, Any] = {
            "model": _MODEL_PLACEHOLDER,
            "messages": messages,
            "stream": stream,
        }
        if self._agent.tools:
            body["tools"] = [t.to_openai() for t in self._agent.tools.values()]
        body.update(self._agent.sampling)
        return body

    # -- HTTP --------------------------------------------------------------- #

    def _post_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        resp = httpx.post(
            self._url, json=self._payload(messages, stream=False),
            headers=self._headers, timeout=_TIMEOUT,
        )
        _raise_for_status(resp.status_code, resp)
        return resp.json()

    def _post_stream(self, messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Yield parsed SSE events from a streaming completion (drops the ``[DONE]`` marker)."""
        with httpx.stream(
            "POST", self._url, json=self._payload(messages, stream=True),
            headers=self._headers, timeout=_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                body = resp.read().decode(errors="replace")
                _raise_for_status(resp.status_code, body)
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except ValueError:
                    continue  # keep-alive/comment lines that aren't JSON


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #


def _normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Validate + clean externally supplied history — the rehydration trust boundary.

    Only ``user``/``assistant`` turns with string content are accepted, and each is rebuilt
    as a fresh ``{"role", "content"}`` dict. A ``system`` or ``tool`` role is **rejected**,
    and any smuggled extra keys (``tool_calls``, ``reasoning_content``, ...) are **dropped**,
    so rehydrating from a store or client blob can never inject a fake system prompt, a forged
    tool result, or spoofed tool calls. Raises ``ValueError`` on an untrusted/malformed entry.
    """
    clean: list[dict[str, Any]] = []
    for m in history or []:
        if not isinstance(m, dict):
            raise ValueError(f"history entries must be dicts, got {type(m).__name__}")
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            raise ValueError(
                f"history role must be 'user' or 'assistant', got {role!r} — system/tool "
                f"roles are server-owned and not accepted from stored/client history"
            )
        if not isinstance(content, str):
            raise ValueError(f"history {role} content must be a str, got {type(content).__name__}")
        clean.append({"role": role, "content": content})
    return clean


def _accumulate_tool_call(acc: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    """Fold one streamed ``tool_calls`` delta into the per-index accumulator."""
    idx = delta.get("index", 0)
    slot = acc.setdefault(
        idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
    )
    if delta.get("id"):
        slot["id"] = delta["id"]
    fn = delta.get("function") or {}
    if fn.get("name"):
        slot["function"]["name"] += fn["name"]
    if fn.get("arguments"):
        slot["function"]["arguments"] += fn["arguments"]


def _raise_for_status(status: int, detail: Any) -> None:
    """Turn a non-200 data-plane response into a clear RuntimeError."""
    if status == 200:
        return
    text = detail if isinstance(detail, str) else getattr(detail, "text", "")
    if status == 503:
        raise RuntimeError(f"no SERVING workers available (data plane 503): {text}")
    raise RuntimeError(f"data plane returned HTTP {status}: {text}")
