"""OpenAI-compatible inference server backed by Hugging Face Transformers.

Run as ``python -m oumigo.worker.hf_server --model <id> --host <h> --port <p>`` — the
coordinator spawns exactly this when a worker is started with ``--backend=transformer``
(see `oumigo.worker.supervisor.HFProcess`). It deliberately mirrors vLLM's HTTP
surface so the rest of the system can't tell the difference:

  * ``GET  /health``               — 200 only once the model is loaded (see below)
  * ``GET  /v1/models``            — the served model id
  * ``POST /v1/chat/completions``  — chat, JSON or SSE streaming (``stream=true``)
  * ``POST /v1/completions``       — text completion, JSON or SSE streaming
  * ``GET  /metrics``              — vLLM-style Prometheus text (request stats +
                                     ``vllm:num_requests_running`` for run-state)

The (multi-minute) model load happens **before** uvicorn starts, so ``/health`` is
simply unreachable while loading — exactly like vLLM, which binds its port only after
the model is ready. The coordinator therefore sees INITIALIZING until the model is up,
then SERVING. `torch`/`transformers` are imported lazily inside `HFEngine.load`, so
this module imports cheaply (and `--help` works) on a box without them.

This generates one request at a time (no batching); the worker reports
``max_concurrent_requests=1`` so the router admits a single in-flight request to it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

log = logging.getLogger("oumigo.worker.hf_server")

DEFAULT_MODEL = "google/gemma-4-E2B-it"  # small, open, ships a chat template
_STREAM_DONE = object()  # sentinel: the streamer iterator is exhausted


# --- request parsing (OpenAI chat / completions bodies) -------------------------


def _extract_messages(body: dict) -> list[dict[str, str]]:
    """Normalize an OpenAI chat body into `[{role, content}]` with string content."""
    out: list[dict[str, str]] = []
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):  # multimodal blocks -> concatenate their text
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        out.append({"role": str(msg.get("role") or "user"), "content": str(content or "")})
    return out or [{"role": "user", "content": ""}]


def _extract_prompt(body: dict) -> str:
    """Pull the prompt text out of a `/v1/completions` request body."""
    prompt = body.get("prompt")
    if isinstance(prompt, list):
        return " ".join(map(str, prompt))
    return str(prompt or "")


# --- HF engine (lazy torch/transformers; blocking calls run off the event loop) --


def _resolve_dtype(name: str, torch: Any, device: str):
    """Map a --dtype name to a torch dtype; `auto` picks per device."""
    if name != "auto":
        return getattr(torch, name)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class HFEngine:
    """Wraps one HF model: load, and blocking generate / stream helpers.

    Loads with an `AutoProcessor` (multimodal models like Gemma need one; falls back
    to `AutoTokenizer` for text LMs) and, on CUDA, 4-bit `bitsandbytes` quantization +
    `device_map="auto"`. Heavy imports happen in `load()` (not at module import), and
    every generation method is synchronous — callers drive them off the event loop via
    `asyncio.to_thread` / `run_in_executor` so the server keeps answering /health.
    """

    def __init__(
        self, model_id: str, *, quantize: str = "4bit", dtype: str = "auto",
        device: str = "auto", max_new_tokens: int = 8192, max_model_len: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.quantize = quantize  # "4bit" | "8bit" | "none" (only applied on CUDA)
        self.dtype_name = dtype
        self.device_name = device
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len  # advisory prompt cap (negotiated via MAX_MODEL_LEN)
        self.processor: Any = None      # multimodal processor, or None for text-only
        self.tokenizer: Any = None      # always a tokenizer (processor.tokenizer if multimodal)
        self.is_multimodal = False
        self.model: Any = None
        self.device: str = "cpu"
        self._torch: Any = None

    def load(self) -> None:
        """Import torch/transformers and load the processor/tokenizer + model (slow)."""
        import torch  # lazy: keep module import cheap so `--help` works without torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        device = self.device_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading %s (device=%s, quantize=%s)", self.model_id, device, self.quantize)

        # Prefer a processor (required by multimodal models); some return a bare
        # tokenizer, and text-only models have none -> fall back to AutoTokenizer.
        processor = tokenizer = None
        try:
            processor = AutoProcessor.from_pretrained(self.model_id)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:  # AutoProcessor handed back a plain tokenizer
                tokenizer, processor = processor, None
        except Exception as exc:  # noqa: BLE001 - no processor for this model; use the tokenizer
            log.debug("no AutoProcessor for %s (%s); using AutoTokenizer", self.model_id, exc)
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token_id is None:  # many causal LMs ship without one
            tokenizer.pad_token = tokenizer.eos_token

        quant = self._quant_config(torch, device)
        if quant is not None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=quant, device_map="auto", dtype="auto",
            )
            self.device = str(model.device)
        else:
            dtype = _resolve_dtype(self.dtype_name, torch, device)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
            model.to(device)
            self.device = device
        model.eval()
        # Pin pad_token_id on the generation config (not per-call) so `generate` never
        # warns about it and open-ended generation stops cleanly.
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        self.processor, self.tokenizer, self.is_multimodal = processor, tokenizer, processor is not None
        self.model, self._torch = model, torch
        log.info("model ready on %s (multimodal=%s)", self.device, self.is_multimodal)

    def _quant_config(self, torch: Any, device: str):
        """A BitsAndBytesConfig for --quantize, or None (full precision / non-CUDA)."""
        if self.quantize not in ("4bit", "8bit"):
            return None
        if device != "cuda":  # bitsandbytes needs a GPU; degrade gracefully
            log.warning("quantize=%s needs CUDA; loading full precision on %s", self.quantize, device)
            return None
        from transformers import BitsAndBytesConfig

        if self.quantize == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    # --- input building + generation kwargs ----------------------------------

    def _encode(self, body: dict, chat: bool) -> dict:
        """Tokenize a request into a dict of model-input tensors on the device."""
        tok = self.tokenizer
        if chat and self.is_multimodal:
            msgs = [
                {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                for m in _extract_messages(body)
            ]
            enc = self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
        elif chat and tok.chat_template:
            enc = tok.apply_chat_template(
                _extract_messages(body), add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
        elif chat:  # no chat template -> a plain role-tagged transcript
            messages = _extract_messages(body)
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            enc = tok(text, return_tensors="pt")
        else:
            enc = tok(_extract_prompt(body), return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items() if hasattr(v, "to")}

    def _gen_kwargs(self, body: dict, enc: dict) -> dict:
        max_new = int(body.get("max_tokens") or self.max_new_tokens)
        temperature = body.get("temperature")
        top_p = body.get("top_p")
        do_sample = temperature is not None and float(temperature) > 0
        kwargs: dict[str, Any] = {**enc, "max_new_tokens": max_new, "do_sample": do_sample}
        if do_sample:
            kwargs["temperature"] = float(temperature)
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
        return kwargs

    # --- blocking generation (called via asyncio.to_thread / executor) -------

    def generate(self, body: dict, chat: bool) -> tuple[str, int, int]:
        """Full (non-streaming) generation. Returns (text, prompt_toks, completion_toks)."""
        torch = self._torch
        enc = self._encode(body, chat)
        kwargs = self._gen_kwargs(body, enc)
        with torch.inference_mode():
            out = self.model.generate(**kwargs)
        input_len = int(enc["input_ids"].shape[-1])
        gen_ids = out[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text, input_len, int(gen_ids.shape[-1])

    def start_stream(self, body: dict, chat: bool):
        """Kick off streamed generation in a background thread.

        Returns (prompt_toks, streamer, thread): the streamer is a *blocking* iterator
        of text pieces; the caller pumps it off the event loop.
        """
        from transformers import TextIteratorStreamer

        torch = self._torch
        enc = self._encode(body, chat)
        input_len = int(enc["input_ids"].shape[-1])
        kwargs = self._gen_kwargs(body, enc)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs["streamer"] = streamer

        def _run() -> None:
            with torch.inference_mode():
                self.model.generate(**kwargs)

        thread = Thread(target=_run, name="hf-generate", daemon=True)
        thread.start()
        return input_len, streamer, thread

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)


def _next_piece(streamer) -> Any:
    """`next(streamer)` or the `_STREAM_DONE` sentinel when exhausted (executor-safe)."""
    try:
        return next(streamer)
    except StopIteration:
        return _STREAM_DONE


# --- vLLM-style request metrics -------------------------------------------------


class _VLLMStats:
    """Accumulates real per-request stats, rendered as vLLM-style Prometheus text.

    Exposed at GET /metrics; the coordinator's `MetricsCollector` scrapes it exactly as
    it would a real vLLM. Unlike vLLM (which exports an e2e-latency *histogram* the
    collector turns into bucket-edge estimates), this single-request-at-a-time backend
    keeps the e2e latency as **exact** running aggregates — count, min, max, and a
    streaming mean — and exports just those four (no percentile tiles: they'd be
    meaningless from 4 numbers, and estimating them from too-coarse buckets produced
    nonsense like mean > max). Token/finish-reason counters pass through unchanged and
    are lifetime-cumulative (reset on restart). `num_running` is a live gauge the
    coordinator reads for run-state (IDLE/EXECUTING).
    """

    _FINISH_REASONS = ("abort", "error", "length", "repetition", "stop")

    def __init__(self) -> None:
        self._lock = Lock()
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.success = {r: 0 for r in self._FINISH_REASONS}
        # Exact e2e-latency aggregates (seconds). mean is updated incrementally.
        self.latency_count = 0
        self.latency_min = 0.0
        self.latency_max = 0.0
        self.latency_mean = 0.0
        self.num_running = 0  # in-flight requests (this backend generates one at a time)

    def inc_running(self) -> None:
        with self._lock:
            self.num_running += 1

    def dec_running(self) -> None:
        with self._lock:
            self.num_running = max(0, self.num_running - 1)

    def record(
        self, latency_s: float, prompt_tokens: int, completion_tokens: int,
        finish_reason: str = "stop",
    ) -> None:
        """Fold one finished request into the counters + running latency aggregates."""
        x = float(latency_s)
        with self._lock:
            self.prompt_tokens_total += int(prompt_tokens)
            self.generation_tokens_total += int(completion_tokens)
            self.success[finish_reason] = self.success.get(finish_reason, 0) + 1
            n = self.latency_count
            if n == 0:
                self.latency_min = self.latency_max = self.latency_mean = x
            else:
                self.latency_min = min(self.latency_min, x)
                self.latency_max = max(self.latency_max, x)
                # streaming mean: new_mean = (x + mean*n) / (n + 1)
                self.latency_mean = (x + self.latency_mean * n) / (n + 1)
            self.latency_count = n + 1

    def render(self) -> str:
        """Prometheus exposition text for the vllm:* metrics."""
        with self._lock:
            lines = [
                f"vllm:num_requests_running {self.num_running}",
                f"vllm:prompt_tokens_total {self.prompt_tokens_total}",
                f"vllm:generation_tokens_total {self.generation_tokens_total}",
            ]
            for r in self._FINISH_REASONS:
                lines.append(
                    f'vllm:request_success_total{{finished_reason="{r}"}} {self.success[r]}'
                )
            # Pre-aggregated e2e summary: count always; min/max/mean only once we have a
            # sample (the collector passes these straight through — no histogram).
            base = "vllm:e2e_request_latency_seconds"
            lines.append(f"{base}_count {self.latency_count}")
            if self.latency_count > 0:
                lines.append(f"{base}_min {self.latency_min:.6f}")
                lines.append(f"{base}_max {self.latency_max:.6f}")
                lines.append(f"{base}_mean {self.latency_mean:.6f}")
        return "\n".join(lines) + "\n"


# --- OpenAI wire payloads (identical shape to the vLLM/router responses) ---------


def _chat_payload(text: str, model: str, p_toks: int, c_toks: int) -> dict:
    return {
        "id": f"chatcmpl-hf-{uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": p_toks, "completion_tokens": c_toks,
                  "total_tokens": p_toks + c_toks},
    }


def _completion_payload(text: str, model: str, p_toks: int, c_toks: int) -> dict:
    return {
        "id": f"cmpl-hf-{uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": p_toks, "completion_tokens": c_toks,
                  "total_tokens": p_toks + c_toks},
    }


def _sse(obj: dict) -> str:
    """One Server-Sent-Events frame carrying a JSON chunk."""
    return f"data: {json.dumps(obj)}\n\n"


def _chat_chunk(cid: str, created: int, model: str, *, delta: dict, finish: str | None = None) -> dict:
    return {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _text_chunk(cid: str, created: int, model: str, *, text: str, finish: str | None = None) -> dict:
    return {
        "id": cid, "object": "text_completion", "created": created, "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
    }


# --- the server -----------------------------------------------------------------


class HFServer:
    """Owns the loaded engine + request stats and builds the FastAPI app."""

    def __init__(self, engine: HFEngine) -> None:
        self.engine = engine
        self.stats = _VLLMStats()

    def build_app(self) -> FastAPI:
        app = FastAPI(title=f"oumigo HF server ({self.engine.model_id})")

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models() -> dict:
            return {"object": "list", "data": [{"id": self.engine.model_id, "object": "model"}]}

        @app.get("/metrics")
        async def metrics() -> Response:  # vLLM-style exposition, scraped by the collector
            return PlainTextResponse(self.stats.render())

        @app.post("/v1/chat/completions")
        async def chat(request: Request) -> Response:
            return await self._handle(request, chat=True)

        @app.post("/v1/completions")
        async def completions(request: Request) -> Response:
            return await self._handle(request, chat=False)

        return app

    async def _handle(self, request: Request, *, chat: bool) -> Response:
        body = await request.json()
        model = body.get("model") or self.engine.model_id

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return StreamingResponse(
                self._stream(body, model, chat=chat, include_usage=include_usage),
                media_type="text/event-stream",
            )

        self.stats.inc_running()
        t0 = time.monotonic()
        try:
            text, p_toks, c_toks = await asyncio.to_thread(self.engine.generate, body, chat)
        finally:
            self.stats.dec_running()
        self.stats.record(time.monotonic() - t0, p_toks, c_toks)
        log.info("served %s: prompt=%d -> completion=%d toks", "chat" if chat else "text", p_toks, c_toks)
        payload = (
            _chat_payload(text, model, p_toks, c_toks)
            if chat
            else _completion_payload(text, model, p_toks, c_toks)
        )
        return JSONResponse(payload)

    async def _stream(
        self, body: dict, model: str, *, chat: bool, include_usage: bool,
    ) -> AsyncIterator[str]:
        """Emit OpenAI SSE chunks as the HF streamer yields text (off the event loop).

        The run counter is held for the whole stream and released even on a mid-stream
        client disconnect (the generator's finally).
        """
        self.stats.inc_running()
        cid = f"{'chatcmpl' if chat else 'cmpl'}-hf-{uuid4().hex[:12]}"
        created = int(time.time())
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        pieces: list[str] = []
        try:
            input_len, streamer, thread = self.engine.start_stream(body, chat)
            if chat:
                yield _sse(_chat_chunk(cid, created, model, delta={"role": "assistant"}))
            while True:
                piece = await loop.run_in_executor(None, _next_piece, streamer)
                if piece is _STREAM_DONE:
                    break
                pieces.append(piece)
                if chat:
                    yield _sse(_chat_chunk(cid, created, model, delta={"content": piece}))
                else:
                    yield _sse(_text_chunk(cid, created, model, text=piece))
            await loop.run_in_executor(None, thread.join)
            if chat:
                yield _sse(_chat_chunk(cid, created, model, delta={}, finish="stop"))
            else:
                yield _sse(_text_chunk(cid, created, model, text="", finish="stop"))
            c_toks = self.engine.count_tokens("".join(pieces))
            if include_usage:
                usage = {"prompt_tokens": input_len, "completion_tokens": c_toks,
                         "total_tokens": input_len + c_toks}
                obj = "chat.completion.chunk" if chat else "text_completion"
                yield _sse({"id": cid, "object": obj, "created": created, "model": model,
                            "choices": [], "usage": usage})
            yield "data: [DONE]\n\n"
            self.stats.record(time.monotonic() - t0, input_len, c_toks)
        finally:
            self.stats.dec_running()


# --- entrypoint -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="oumigo.worker.hf_server",
        description="OpenAI-compatible HF-transformers inference server (spawned by the worker).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME") or DEFAULT_MODEL,
                        help="HF model id/path to serve (env: MODEL_NAME).")
    parser.add_argument("--host", default="0.0.0.0", help="Address to bind.")
    parser.add_argument("--port", type=int, default=7001, help="Port to bind.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--quantize", default="4bit", choices=["4bit", "8bit", "none"],
                        help="bitsandbytes quantization (CUDA only).")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="Default generation length when a request omits max_tokens.")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Advisory max context length (env: MAX_MODEL_LEN).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log at DEBUG.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Reduce CUDA fragmentation OOMs; must be set before torch initializes CUDA.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    engine = HFEngine(
        args.model, quantize=args.quantize, dtype=args.dtype, device=args.device,
        max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
    )
    # Load BEFORE serving so /health is unreachable until the model is ready — the
    # coordinator reads that as INITIALIZING, then SERVING (exactly like vLLM). A load
    # failure exits non-zero, which the coordinator treats as a crash (restart policy).
    engine.load()

    server = HFServer(engine)
    uvicorn.run(server.build_app(), host=args.host, port=args.port,
                log_config=None, access_log=False, log_level="info" if args.verbose else "warning")


if __name__ == "__main__":
    main()
