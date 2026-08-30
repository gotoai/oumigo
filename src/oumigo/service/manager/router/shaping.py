"""Model-capability shaping applied to chat requests as they pass through the router.

Two guards, both **off unless configured** (`model.max_audio_seconds` /
`model.max_output_tokens` in `manager.yaml`). They exist because a model's real
limits are *fleet* knowledge — the router is the only component that knows which
model the fleet serves — and because a client that trips them gets no useful
error from vLLM, only silence or a repetition loop.

**Why an audio cap.** Measured on `google/gemma-4-12B-it-qat-w4a16-ct`: a single
audio item is hard-capped by the processor at `audio_seq_length` (750 tokens x
40 ms = **30 s**) — audio past that is dropped silently, with no error. Clients
therefore chunk into <=30 s items and send many per request, which works until
the *total* gets long: past roughly 8 minutes the model stops emitting new text
and repeats a block until it hits the output ceiling. That collapse is stochastic
(24 items came back clean while 20 showed mild repetition), so the cap is a
probability boundary, not a cliff — pick it with margin.

**Why the output clamp.** The collapse is unbounded *output*, so bounding
`max_tokens` addresses it directly: the loop terminates instead of occupying a
worker slot for as long as the model will talk. oumi-gateway already server-sets
this for Kari traffic; the router's clamp is defence in depth for anything that
reaches the fleet without passing through a gateway.

Budgeting counts each item as ``min(measured, 30 s)``, because 30 s is all the
model ingests from one item however long it really is. WAV duration is read from
the RIFF header (stdlib only — the manager stays free of `av`/`soundfile`, and
nothing is fully base64-decoded); any other container is billed at the 30 s
ceiling, which is an upper bound on what the model will consume.
"""

from __future__ import annotations

import base64
import binascii
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("oumigo.manager.router.shaping")

# Gemma 4's per-item processor window: audio_seq_length (750) x audio_ms_per_token (40).
# Also the fallback bill for an item whose duration we cannot read.
ITEM_CEILING_S = 30.0

# Enough base64 to cover a RIFF header and its leading chunks (768 chars -> 576 bytes).
_HEADER_B64_CHARS = 768


@dataclass(frozen=True, slots=True)
class AudioShaping:
    """What the audio cap did to one request."""

    submitted_s: float          # total audio the client sent, as the model would count it
    kept_s: float               # total left after trimming
    total_items: int
    dropped_items: int
    limit_s: float | None

    @property
    def trimmed(self) -> bool:
        return self.dropped_items > 0


def _strip_ws(data: str) -> str:
    return "".join(data.split()) if (" " in data or "\n" in data or "\r" in data) else data


def _b64_byte_len(data: str) -> int:
    """Decoded length of a base64 string, without decoding it."""
    n = len(data)
    if n < 4:
        return 0
    return (n // 4) * 3 - data.count("=", n - 2)


def _b64_head(data: str) -> bytes:
    """Decode just the leading bytes — enough for a container header."""
    prefix = data[: (min(len(data), _HEADER_B64_CHARS) // 4) * 4]
    try:
        return base64.b64decode(prefix, validate=False)
    except (binascii.Error, ValueError):
        return b""


def _wav_seconds(head: bytes, total_bytes: int) -> float | None:
    """Duration of a RIFF/WAVE payload from its header, or None if it isn't one.

    Uses the fmt chunk's `byte_rate` and the *actual* payload length rather than the
    declared data-chunk size, which is wrong or zero in streamed and hand-built WAVs.
    """
    if len(head) < 12 or head[0:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    byte_rate = 0
    pos = 12
    while pos + 8 <= len(head):
        chunk_id = head[pos : pos + 4]
        chunk_size = int.from_bytes(head[pos + 4 : pos + 8], "little")
        body = pos + 8
        if chunk_id == b"fmt " and body + 12 <= len(head):
            byte_rate = int.from_bytes(head[body + 8 : body + 12], "little")
        elif chunk_id == b"data":
            if byte_rate <= 0:
                return None
            data_bytes = total_bytes - body
            if 0 < chunk_size < data_bytes:
                data_bytes = chunk_size
            return data_bytes / byte_rate if data_bytes > 0 else 0.0
        if chunk_size <= 0:
            break
        pos = body + chunk_size + (chunk_size & 1)  # chunks are word-aligned
    return None


def _part_b64(part: dict[str, Any]) -> str | None:
    """The base64 payload of an audio content part, for either accepted spelling."""
    kind = part.get("type")
    if kind == "input_audio":
        blob = part.get("input_audio")
        if isinstance(blob, dict):
            data = blob.get("data")
            return data if isinstance(data, str) else None
        return None
    if kind == "audio_url":
        url = part.get("audio_url")
        if isinstance(url, dict):
            url = url.get("url")
        # Only a data: URL can be measured here; an http(s) one would need a fetch.
        if isinstance(url, str) and url.startswith("data:") and "," in url:
            return url.split(",", 1)[1]
        return None
    return None


def _is_audio_part(part: object) -> bool:
    return isinstance(part, dict) and part.get("type") in ("input_audio", "audio_url")


def audio_part_seconds(part: dict[str, Any]) -> float:
    """How much audio the *model* will take from this part, in seconds.

    Capped at the per-item processor window: a 90 s clip contributes 30 s, because
    that is all the model ever sees of it.
    """
    data = _part_b64(part)
    if not data:
        return ITEM_CEILING_S
    data = _strip_ws(data)
    measured = _wav_seconds(_b64_head(data), _b64_byte_len(data))
    if measured is None or measured <= 0:
        return ITEM_CEILING_S  # unknown container: bill it at the ceiling
    return min(measured, ITEM_CEILING_S)


def _note(kept_s: float, dropped: int, total: int, limit_s: float) -> dict[str, Any]:
    return {
        "type": "text",
        "text": (
            f"[oumigo] This fleet processes at most {limit_s / 60:.0f} minutes of audio "
            f"per request. {dropped} of {total} audio clips were removed before the model "
            f"saw them; only the first {kept_s / 60:.1f} minutes were processed. "
            f"Say so in your reply, and tell the user to send the rest in a new request."
        ),
    }


def cap_audio(payload: dict[str, Any], limit_s: float | None) -> AudioShaping:
    """Trim audio parts in place until the request's total audio fits `limit_s`.

    Whole items are dropped from the end — trimming *within* an item would mean
    re-encoding, which the manager has no decoder for. A note is prepended to the
    message that lost parts so the model can say what happened; the caller also
    surfaces the numbers as response headers for clients that would rather read
    them than parse prose.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return AudioShaping(0.0, 0.0, 0, 0, limit_s)

    seconds: float = 0.0
    items = 0
    dropped = 0
    kept_s = 0.0
    budget_spent = False
    note_target: dict[str, Any] | None = None

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        kept: list[Any] = []
        lost_here = 0
        for part in content:
            if not _is_audio_part(part):
                kept.append(part)
                continue
            items += 1
            span = audio_part_seconds(part)
            seconds += span
            if limit_s is not None and (budget_spent or kept_s + span > limit_s):
                budget_spent = True
                dropped += 1
                lost_here += 1
                continue
            kept_s += span
            kept.append(part)

        if lost_here:
            message["content"] = kept
            if note_target is None:
                note_target = message

    if note_target is not None:
        # Written once, at the end, so it can quote the final totals.
        note_target["content"] = [
            _note(kept_s, dropped, items, float(limit_s or 0)),
            *note_target["content"],
        ]
        log.info(
            "audio cap: trimmed %d/%d clips (%.1fs submitted, %.1fs kept, limit %.0fs)",
            dropped, items, seconds, kept_s, limit_s or 0,
        )

    return AudioShaping(seconds, kept_s, items, dropped, limit_s)


def clamp_output_tokens(payload: dict[str, Any], ceiling: int | None) -> bool:
    """Bound the reply length. Returns True if the payload was changed.

    An absent `max_tokens` is *set* to the ceiling, not left alone — the runaway this
    guards against is precisely a request that named no limit and ran to the server's.
    """
    if ceiling is None or ceiling <= 0:
        return False
    changed = False
    for field in ("max_tokens", "max_completion_tokens"):
        if field == "max_completion_tokens" and field not in payload:
            continue  # only clamp the newer spelling when the client actually used it
        requested = payload.get(field)
        capped = (
            min(int(requested), ceiling)
            if isinstance(requested, int) and requested > 0
            else ceiling
        )
        if payload.get(field) != capped:
            payload[field] = capped
            changed = True
    return changed


def shaping_headers(shaping: AudioShaping) -> dict[str, str]:
    """Response headers describing what the audio cap did (empty when it did nothing)."""
    if not shaping.trimmed:
        return {}
    return {
        "x-oumigo-audio-trimmed": "true",
        "x-oumigo-audio-limit-seconds": f"{shaping.limit_s:.0f}",
        "x-oumigo-audio-submitted-seconds": f"{shaping.submitted_s:.1f}",
        "x-oumigo-audio-processed-seconds": f"{shaping.kept_s:.1f}",
        "x-oumigo-audio-clips-dropped": f"{shaping.dropped_items}/{shaping.total_items}",
    }
