"""Tests for the router's model-capability caps: audio trimming + output clamping."""

from __future__ import annotations

import base64
import struct

import pytest

from oumigo.service.manager.router.shaping import (
    ITEM_CEILING_S,
    audio_part_seconds,
    cap_audio,
    clamp_output_tokens,
    shaping_headers,
)


def _wav(seconds: float, rate: int = 16000, channels: int = 1, bits: int = 16) -> str:
    """A base64 PCM WAV of the requested duration — what Kari puts on the wire."""
    byte_rate = rate * channels * bits // 8
    data = b"\0" * int(seconds * byte_rate)
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate,
                                channels * bits // 8, bits)
        + b"data" + struct.pack("<I", len(data))
    )
    return base64.b64encode(header + data).decode()


def _audio(seconds: float) -> dict:
    return {"type": "input_audio", "input_audio": {"data": _wav(seconds), "format": "wav"}}


def _msg(*parts) -> dict:
    return {"role": "user", "content": list(parts)}


# --- measurement ----------------------------------------------------------------


@pytest.mark.parametrize("seconds", [1.0, 5.0, 15.0, 29.0, 30.0])
def test_wav_duration_is_read_from_the_header(seconds: float) -> None:
    assert audio_part_seconds(_audio(seconds)) == pytest.approx(seconds, abs=0.01)


def test_long_clip_is_billed_at_the_per_item_ceiling() -> None:
    # The model only ever ingests 30s of one item, so that is what the budget counts.
    assert audio_part_seconds(_audio(90.0)) == ITEM_CEILING_S


def test_unreadable_container_is_billed_at_the_ceiling() -> None:
    part = {"type": "input_audio", "input_audio": {"data": base64.b64encode(b"ID3\x04junk" * 40).decode(),
                                                   "format": "mp3"}}
    assert audio_part_seconds(part) == ITEM_CEILING_S


def test_other_sample_rates_and_stereo() -> None:
    assert audio_part_seconds(
        {"type": "input_audio", "input_audio": {"data": _wav(10.0, rate=44100, channels=2)}}
    ) == pytest.approx(10.0, abs=0.01)


def test_data_url_audio_part_is_measured_too() -> None:
    part = {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_wav(12.0)}"}}
    assert audio_part_seconds(part) == pytest.approx(12.0, abs=0.01)


# --- trimming -------------------------------------------------------------------


def test_no_limit_leaves_the_payload_untouched() -> None:
    payload = {"messages": [_msg(*[_audio(30.0)] * 20)]}
    result = cap_audio(payload, None)
    assert not result.trimmed
    assert result.submitted_s == pytest.approx(600.0, abs=0.5)
    assert len(payload["messages"][0]["content"]) == 20


def test_request_within_the_limit_is_untouched() -> None:
    payload = {"messages": [_msg({"type": "text", "text": "transcribe"}, *[_audio(30.0)] * 12)]}
    result = cap_audio(payload, 360.0)
    assert not result.trimmed
    assert shaping_headers(result) == {}
    assert len(payload["messages"][0]["content"]) == 13


def test_over_limit_drops_trailing_clips_and_keeps_order() -> None:
    payload = {"messages": [_msg({"type": "text", "text": "transcribe"}, *[_audio(30.0)] * 31)]}
    result = cap_audio(payload, 360.0)

    assert result.trimmed
    assert result.total_items == 31
    assert result.dropped_items == 19        # 12 kept = 360s
    assert result.kept_s == pytest.approx(360.0, abs=0.5)
    assert result.submitted_s == pytest.approx(930.0, abs=1.0)

    content = payload["messages"][0]["content"]
    audio = [p for p in content if p.get("type") == "input_audio"]
    assert len(audio) == 12
    assert content[0]["type"] == "text" and content[0]["text"].startswith("[OumiGo]")
    assert content[1] == {"type": "text", "text": "transcribe"}  # the client's own text survives


def test_note_states_the_real_numbers() -> None:
    payload = {"messages": [_msg(*[_audio(30.0)] * 20)]}
    cap_audio(payload, 300.0)
    note = payload["messages"][0]["content"][0]["text"]
    assert "5 minutes" in note        # the limit
    assert "10 of 20" in note         # clips removed
    assert "5.0 minutes" in note      # what was processed


def test_partial_clip_does_not_overflow_the_budget() -> None:
    # 4 x 30s + 1 x 10s against a 100s budget: the 4th clip would exceed it, so it goes.
    payload = {"messages": [_msg(_audio(30.0), _audio(30.0), _audio(30.0), _audio(30.0))]}
    result = cap_audio(payload, 100.0)
    assert result.dropped_items == 1
    assert result.kept_s <= 100.0


def test_audio_spread_across_messages_shares_one_budget() -> None:
    payload = {"messages": [_msg(*[_audio(30.0)] * 8), _msg(*[_audio(30.0)] * 8)]}
    result = cap_audio(payload, 300.0)
    assert result.total_items == 16
    assert result.dropped_items == 6
    kept = [p for m in payload["messages"] for p in m["content"] if p.get("type") == "input_audio"]
    assert len(kept) == 10


def test_text_only_request_is_ignored() -> None:
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    result = cap_audio(payload, 60.0)
    assert not result.trimmed and result.total_items == 0
    assert payload["messages"][0]["content"] == "hello"


def test_malformed_payloads_do_not_raise() -> None:
    assert cap_audio({}, 60.0).total_items == 0
    assert cap_audio({"messages": "nope"}, 60.0).total_items == 0
    assert cap_audio({"messages": [None, 3, {"role": "user"}]}, 60.0).total_items == 0


def test_headers_describe_the_trim() -> None:
    payload = {"messages": [_msg(*[_audio(30.0)] * 31)]}
    headers = shaping_headers(cap_audio(payload, 360.0))
    assert headers["x-oumigo-audio-trimmed"] == "true"
    assert headers["x-oumigo-audio-limit-seconds"] == "360"
    assert headers["x-oumigo-audio-clips-dropped"] == "19/31"


# --- output clamp ---------------------------------------------------------------


def test_clamp_is_off_when_unconfigured() -> None:
    payload = {"max_tokens": 100000}
    assert not clamp_output_tokens(payload, None)
    assert payload["max_tokens"] == 100000


def test_clamp_lowers_an_excessive_request() -> None:
    payload = {"max_tokens": 100000}
    assert clamp_output_tokens(payload, 4096)
    assert payload["max_tokens"] == 4096


def test_clamp_sets_a_ceiling_when_the_client_sent_none() -> None:
    # The runaway this guards against named no limit at all.
    payload = {"messages": []}
    assert clamp_output_tokens(payload, 4096)
    assert payload["max_tokens"] == 4096


def test_clamp_leaves_a_modest_request_alone() -> None:
    payload = {"max_tokens": 512}
    assert not clamp_output_tokens(payload, 4096)
    assert payload["max_tokens"] == 512


def test_clamp_handles_the_newer_spelling_only_when_present() -> None:
    payload = {"max_completion_tokens": 99999}
    assert clamp_output_tokens(payload, 4096)
    assert payload["max_completion_tokens"] == 4096
    assert payload["max_tokens"] == 4096

    payload = {"max_tokens": 10}
    clamp_output_tokens(payload, 4096)
    assert "max_completion_tokens" not in payload
