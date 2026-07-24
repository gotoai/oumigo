"""The result of one :meth:`OumigoChat.request` — one type, streamed or not.

Response *parity*: ``request()`` hands back an ``OumigoResponse`` regardless of the
``stream`` flag, so callers never branch on it. Iterating the response yields the final
answer as **parsed text deltas** (``str``) — never raw SSE lines — and ``.text`` holds
the full answer: complete immediately for a non-streamed call, and complete once
iteration finishes for a streamed one.

    resp = chat.request("hi")                 # non-stream: .text ready now
    print(resp.text)

    resp = chat.request("hi", stream=True)    # stream: iterate for deltas
    for piece in resp:
        print(piece, end="")
    print(resp.text)                          # same full answer, after consumption

The agent loop runs *inside* the response's generator, so intermediate tool-calling
turns are consumed silently and only the final answer's text is yielded.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class OumigoResponse:
    """One request's result. Iterate for text deltas; read ``.text`` for the whole answer.

    Attributes:
        text: The assistant's final answer (accumulates as a stream is consumed).
        finish_reason: Why generation stopped — ``"stop"``, ``"length"``, or
            ``"max_iterations"`` when the tool loop hit its cap without a final answer.
        tool_calls_made: One ``{"name", "arguments", "result"}`` entry per tool the loop
            executed, in order — for observability.
        raw: The last raw completion payload from the data plane (escape hatch).

    A response is single-use for streaming: iterating drives the underlying request. Once
    consumed, re-iterating simply re-yields the full ``text`` a single time.
    """

    def __init__(self) -> None:
        self.text: str = ""
        self.finish_reason: str | None = None
        self.tool_calls_made: list[dict[str, Any]] = []
        self.raw: dict[str, Any] | None = None
        self._gen: Iterator[str] | None = None
        self._consumed: bool = False

    def __iter__(self) -> Iterator[str]:
        if self._consumed or self._gen is None:
            # Already drained (or nothing to drive): re-yield the full answer once, so a
            # non-streamed response is still iterable and a re-iterated stream is harmless.
            if self.text:
                yield self.text
            return
        for delta in self._gen:
            self.text += delta
            yield delta
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
