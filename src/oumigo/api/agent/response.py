"""The result of one :meth:`OumigoChat.request` — one type, streamed or not.

Response *parity*: ``request()`` hands back an ``OumigoResponse`` regardless of the
``stream`` flag, so callers never branch on it. Iterating the response yields the final
answer as **parsed text deltas** (``str``) — never raw SSE lines — and ``.text`` holds
the full answer: complete immediately for a non-streamed call, and complete once
iteration finishes for a streamed one.

    resp = chat.request("hi")                 # non-stream: .text ready now
    print(resp.text)

    resp = chat.request("hi", stream=True)    # stream: iterate for deltas
    for piece in resp:                        # (or `resp.stream()` — identical)
        print(piece, end="")
    print(resp.text)                          # same full answer, after consumption

To surface the model's thinking live, opt in with ``stream(get_reasoning=True)``::

    for text, reasoning in resp.stream(get_reasoning=True):
        ...                                   # reasoning deltas, then answer deltas

The agent loop runs *inside* the response's generator, so intermediate tool-calling
turns are consumed silently and only the final answer's text is yielded (reasoning is
kept on the separate ``reasoning`` channel, never mixed into ``text``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, overload


class OumigoResponse:
    """One request's result. Iterate for text deltas; read ``.text`` for the whole answer.

    Attributes:
        text: The assistant's final answer (accumulates as a stream is consumed).
        reasoning: The model's reasoning/thinking (``reasoning_content``), concatenated
            across the request's turns and kept out of ``text``. Populated only when the
            worker's vLLM runs a ``--reasoning-parser``; ``""`` otherwise. Output-only —
            for display/debugging, never fed back to the model (see the class notes).
        finish_reason: Why generation stopped — ``"stop"``, ``"length"``, or
            ``"max_iterations"`` when the tool loop hit its cap without a final answer.
        tool_calls_made: One ``{"name", "arguments", "result"}`` entry per tool the loop
            executed, in order — for observability.
        raw: The last raw completion payload from the data plane (escape hatch).

    A response is single-use for streaming: iterating drives the underlying request. Once
    consumed, re-iterating simply re-yields the full ``text`` a single time. Note that
    ``reasoning`` is *never* yielded by iteration — only ``text`` deltas are — so it fills
    in alongside and is read after consumption (``resp.reasoning``).
    """

    def __init__(self) -> None:
        self.text: str = ""
        self.reasoning: str = ""
        self.finish_reason: str | None = None
        self.tool_calls_made: list[dict[str, Any]] = []
        self.raw: dict[str, Any] | None = None
        self._gen: Iterator[tuple[str, str]] | None = None
        self._consumed: bool = False

    def _note_reasoning(self, text: str) -> None:
        """Append one turn's ``reasoning_content`` (internal; kept out of ``text``).

        Turns are separated by a blank line so a multi-step (tool-loop) request reads as
        distinct thoughts. Never resent to the model — see :meth:`OumigoChat._remember`.
        """
        if not text:
            return
        self.reasoning += ("\n\n" + text) if self.reasoning else text

    def __iter__(self) -> Iterator[str]:
        """Yield the final answer's text deltas (str). Reasoning is never yielded here."""
        if self._consumed or self._gen is None:
            # Already drained (or nothing to drive): re-yield the full answer once, so a
            # non-streamed response is still iterable and a re-iterated stream is harmless.
            if self.text:
                yield self.text
            return
        for kind, piece in self._gen:
            if kind == "answer":
                yield piece
        self._consumed = True

    @overload
    def stream(self, get_reasoning: Literal[False] = False) -> Iterator[str]: ...
    @overload
    def stream(self, get_reasoning: Literal[True]) -> Iterator[tuple[str, str]]: ...

    def stream(self, get_reasoning: bool = False):
        """Iterate a streamed response, optionally exposing reasoning alongside the answer.

        Default (``get_reasoning=False``) is identical to ``for piece in resp`` — yields the
        final answer's text deltas (``str``)::

            for piece in resp.stream():
                ...

        With ``get_reasoning=True``, yields ``(text_delta, reasoning_delta)`` pairs, exactly
        one non-empty per step (the other is ``""``, never ``None``, so ``if text:`` /
        ``if reasoning:`` read cleanly)::

            for text, reasoning in resp.stream(get_reasoning=True):
                if reasoning:
                    ...  # live "thinking" (reasoning_content)
                if text:
                    ...  # the answer

        Either mode drives the same one-shot request, so pick one call; ``text`` and
        ``reasoning`` accumulate as it runs. Once consumed, re-calling yields the accumulated
        answer (or ``(text, reasoning)`` pair) a single time.
        """
        if not get_reasoning:
            yield from self.__iter__()  # answer-only str deltas — same path as `for piece in resp`
            return
        if self._consumed or self._gen is None:
            if self.text or self.reasoning:
                yield self.text, self.reasoning
            return
        for kind, piece in self._gen:
            yield (piece, "") if kind == "answer" else ("", piece)
        self._consumed = True

    def consume(self) -> OumigoResponse:
        """Drive the request to completion (used internally for non-streaming calls)."""
        for _ in self:
            pass
        return self

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        preview = self.text if len(self.text) <= 60 else self.text[:57] + "..."
        return (
            f"OumigoResponse(finish_reason={self.finish_reason!r}, "
            f"tools={len(self.tool_calls_made)}, text={preview!r})"
        )
