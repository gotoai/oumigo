"""The Chat tier — a stateful conversation that runs one turn (and its tool loop) at a time.

An :class:`OumigoChat` is spawned by :meth:`oumigo.api.agent.agent.OumigoAgent.create_chat`.
It accumulates history and is *not* thread-safe: use one chat per session.
:meth:`OumigoChat.request` runs **one user turn to completion**, returning an
:class:`~oumigo.api.agent.response.OumigoResponse`.

Everything ultimately becomes an OpenAI-style ``POST /v1/chat/completions`` against the
manager's data plane (``data_url``), which the router proxies to a SERVING worker. When
the model asks to call a tool, the loop here executes the matching Python callback, feeds
the result back, and continues — until the model returns prose or the iteration cap is hit.

Every model call funnels through the single private seam :meth:`OumigoChat._payload`; the
guardrail interceptor chain (a later version) slots in there without touching this public
API.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import httpx

from oumigo.api.agent.response import OumigoResponse

if TYPE_CHECKING:  # avoid a runtime import cycle: agent.py imports this module
    from oumigo.api.agent.agent import OumigoAgent

log = logging.getLogger("oumigo.api.agent.chat")

# The router overwrites the request's `model` with the fleet's real model name, so the
# value we send is a placeholder that only has to satisfy the OpenAI schema.
_MODEL_PLACEHOLDER = "oumigo"

# No total read timeout — a long generation must not be cut off; cap only the connect.
_TIMEOUT = httpx.Timeout(None, connect=10.0)


class OumigoChat:
    """A stateful conversation. Accumulates history across calls; single-threaded."""

    def __init__(
        self,
        agent: OumigoAgent,
        *,
        system: str | None = None,
        max_history_turns: int = 3,
    ) -> None:
        self._agent = agent
        self._system = system
        self._max_history_turns = max(0, int(max_history_turns))
        self._history: list[dict[str, Any]] = []
        self._url = f"{agent.data_url}/v1/chat/completions"
        self._headers = {"Authorization": f"Bearer {agent.token}"} if agent.token else {}

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
    ) -> Iterator[str]:
        """Drive the tool loop, yielding the final answer's text deltas.

        Runs up to ``max_iterations`` model round-trips. Each turn: get the assistant
        message; if it requests tools, execute them, append their results, and loop; else
        it's the final answer and we stop. On completion, record the turn in history.
        """
        reason = "max_iterations"
        for _ in range(self._agent.max_iterations):
            assistant, finish = yield from self._complete_turn(resp, messages, stream)
            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                reason = finish or "stop"
                break
            for tc in tool_calls:
                result = self._execute_tool(resp, tc)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.get("id"), "content": result}
                )
        resp.finish_reason = reason
        self._remember(user_contents, resp.text)

    def _complete_turn(
        self, resp: OumigoResponse, messages: list[dict[str, Any]], stream: bool
    ) -> Iterator[str]:
        """One model round-trip. Yields content deltas; returns ``(assistant_msg, finish)``."""
        if stream:
            assistant, finish = yield from self._stream_turn(resp, messages)
            return assistant, finish

        data = self._post_json(messages)
        resp.raw = data
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        resp._note_reasoning(msg.get("reasoning_content") or "")  # output-only; never echoed back
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            assistant["tool_calls"] = msg["tool_calls"]
        content = msg.get("content")
        if content:
            yield content
        return assistant, choice.get("finish_reason")

    def _stream_turn(
        self, resp: OumigoResponse, messages: list[dict[str, Any]]
    ) -> Iterator[str]:
        """A streaming model round-trip: yield content deltas, assemble the assistant msg."""
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
            # Reasoning deltas are accumulated but never yielded — only answer text streams.
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            if delta.get("content"):
                content_parts.append(delta["content"])
                yield delta["content"]
            for tcd in delta.get("tool_calls") or []:
                _accumulate_tool_call(tool_calls, tcd)

        resp._note_reasoning("".join(reasoning_parts))  # commit this turn's reasoning
        assistant: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts) or None}
        if tool_calls:
            assistant["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        return assistant, finish

    def _execute_tool(self, resp: OumigoResponse, tc: dict[str, Any]) -> str:
        """Run one requested tool, returning a string result to feed back to the model.

        Never raises: an unknown tool, unparseable arguments, or an exception in the tool
        body all become an ``"Error: ..."`` string so the model can recover and the loop
        continues.
        """
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments") or "{}"
        tool = self._agent.tools.get(name)

        args: Any = raw_args
        if tool is None:
            result = f"Error: unknown tool {name!r}"
        else:
            try:
                args = json.loads(raw_args) if raw_args else {}
            except ValueError as exc:
                result = f"Error: could not parse arguments ({exc})"
            else:
                try:
                    out = tool.invoke(**args)
                    result = out if isinstance(out, str) else json.dumps(out, default=str)
                except Exception as exc:  # noqa: BLE001 - surface to the model, don't crash
                    log.warning("tool %s raised: %s", name, exc)
                    result = f"Error: {type(exc).__name__}: {exc}"

        resp.tool_calls_made.append({"name": name, "arguments": args, "result": result})
        return result

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
        keep = 2 * self._max_history_turns
        if len(self._history) > keep:
            self._history = self._history[-keep:]

    def _payload(self, messages: list[dict[str, Any]], stream: bool) -> dict[str, Any]:
        """Build the request body — the single seam every model call passes through."""
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
