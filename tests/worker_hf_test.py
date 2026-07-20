#!/usr/bin/env python
"""A single *real* worker backed by Hugging Face Transformers (no vLLM).

Where `worker_stub_test.py` fakes a fleet of dummy workers with synthetic metrics,
this runs ONE genuine worker end-to-end against a test manager:

  * **real inference** — loads an actual HF model (`--model`) via `AutoProcessor`
    (multimodal models like Gemma) with 4-bit `bitsandbytes` quantization on CUDA
    (`--quantize`, needed to fit large models in VRAM), and serves the same OpenAI API
    the router forwards (non-streaming JSON *and* SSE streaming when `stream=true`),
    generating with `transformers` instead of vLLM;
  * **real metrics** — reuses the production `MetricsCollector`, so the reported
    `worker:*` (host) and `gpu:*` (NVML / nvidia-smi) series are the machine's real
    numbers, sampled and reported exactly as a real worker would. vLLM metrics are
    ignored: the collector is given no vLLM URL, so `vllm:*` is simply empty (and we
    never stamp `vllm:start_timestamp` — there is no vLLM);
  * **real control plane** — finds the manager (explicit `--manager-url`/env, else
    mDNS discovery on the LAN, exactly like the real worker), registers, learns its
    cadence + preferred port, then heartbeats its node/run state. The heartbeat starts
    *before* the model finishes loading, so the (minutes-long) INITIALIZING window is
    visible to the manager, flipping to SERVING once the model is ready.

Single worker only (no `--workers`, no dummy fleet). Same-host quick test:

    # test_manager.yaml — data_plane: {host: 127.0.0.1, port: 7017}
    oumigo manager serve -c test_manager.yaml --no-mdns --host 127.0.0.1 --port 7016
    python tests/worker_hf_test.py --host 127.0.0.1 \
        --manager-url http://127.0.0.1:7016 --model Qwen/Qwen2.5-0.5B-Instruct

Cross-host (worker on another LAN machine): start the manager advertising on the LAN
(bind 0.0.0.0, mDNS ON — do NOT pass --no-mdns), then just run the worker with no
--manager-url and it discovers the manager over mDNS; `--host` defaults to this host's
LAN IP so the manager's router can reach the served model:

    oumigo manager serve -c manager.yaml --host 0.0.0.0 --port 7014
    python tests/worker_hf_test.py            # discovers the manager on the LAN
    #  ...or point it explicitly: --manager-url http://<manager-LAN-IP>:7014

The manager needn't have a model configured: the worker reports its own preflight-
negotiated port, and takes the model from `--model` (falling back to the manager's
node_spec if you omit it).

A `.env` in the working directory (or `--env-file`) is loaded at startup and honored:
`MODEL_ID` sets the default model, `HF_HOME` the HF cache dir (with `~`/`$VARS`
expanded), and `HF_TOKEN` authenticates gated-model downloads (e.g. Gemma). Real
shell exports win over the file; an empty `HF_TOKEN=` is treated as anonymous.

NOTE: this is a runnable script, not a pytest module. It defines no `test_*`
functions and does nothing at import time — and crucially imports the heavy
`torch`/`transformers` deps *lazily* (inside functions), so pytest can still import
this file during collection on a machine without them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import AsyncIterator
from threading import Thread
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from oumigo.protocol.messages import (
    HeartbeatRequest,
    HeartbeatResponse,
    NodeCapabilities,
    RegisterRequest,
    RegisterResponse,
)
from oumigo.common.env import load_env_file  # stdlib .env loader (reused from the worker)
from oumigo.discovery import (  # same mDNS discovery the real worker uses
    DEFAULT_DISCOVER_TIMEOUT,
    discover_manager,
    get_lan_ip,
)
from oumigo.protocol.states import NodeState, RunState
from oumigo.worker.metrics import MetricsCollector  # real host+GPU sampling/reporting
from oumigo.worker.supervisor import PortUnavailable, find_free_port  # reuse worker port logic

log = logging.getLogger("oumigo.worker_hf")

DEFAULT_BASE_PORT = 7001    # preferred serving port when the manager has no model.port
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # small, open, has a chat template
_STREAM_DONE = object()     # sentinel: the streamer iterator is exhausted


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
    to `AutoTokenizer` for plain text LMs) and, on CUDA, 4-bit `bitsandbytes`
    quantization + `device_map="auto"` — the proven recipe that fits a 12B Gemma in
    ~8GB VRAM. Heavy imports happen in `load()` (not at module import), and every
    generation method is synchronous — callers drive them off the event loop via
    `asyncio.to_thread` / `run_in_executor` so the control plane keeps ticking.
    """

    def __init__(
        self, model_id: str, *, quantize: str = "4bit", dtype: str = "auto",
        device: str = "auto", max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.quantize = quantize  # "4bit" | "8bit" | "none" (only applied on CUDA)
        self.dtype_name = dtype
        self.device_name = device
        self.max_new_tokens = max_new_tokens
        self.processor: Any = None      # multimodal processor, or None for text-only
        self.tokenizer: Any = None      # always a tokenizer (processor.tokenizer if multimodal)
        self.is_multimodal = False
        self.model: Any = None
        self.device: str = "cpu"
        self._torch: Any = None

    def load(self) -> None:
        """Import torch/transformers and load the processor/tokenizer + model (slow)."""
        import torch  # lazy: keep module import cheap so pytest can collect this file
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        log.info("Starts to load the model.")
        device = self.device_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Use device: {device}")

        # Prefer a processor (required by multimodal models like Gemma); some return a
        # bare tokenizer, and text-only models have none -> fall back to AutoTokenizer.
        processor = tokenizer = None
        try:
            log.info(f"To load model {self.model_id}")
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
        log.info(
            "loading %s (device=%s, quantize=%s) - this can take a while",
            self.model_id, device, self.quantize if quant is not None else "none",
        )
        if quant is not None:
            log.info("Loading...")
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
            # Multimodal chat templates (Gemma) want content as typed parts.
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

        Returns (prompt_toks, streamer, thread): the streamer is a *blocking*
        iterator of text pieces; the caller pumps it off the event loop.
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


# --- the worker -----------------------------------------------------------------


class HFWorker:
    """One real worker: registration, OpenAI serving over HF, heartbeat, real metrics."""

    def __init__(
        self, manager_url: str, token: str | None, host: str,
    ) -> None:
        self.manager_url = manager_url.rstrip("/")
        self.token = token
        self.host = host
        self.preferred_port = DEFAULT_BASE_PORT  # updated from node_spec at registration
        self.port: int | None = None             # actual port, chosen by preflight
        self.spec_model: str | None = None        # model the manager configured, if any

        self.node_id = str(uuid4())
        self.started_at = time.time()              # worker:start_timestamp
        self.heartbeat_interval = 10
        self.node_state: NodeState = NodeState.REGISTERING
        self.inflight = 0

        self.engine: HFEngine | None = None
        self.metrics: MetricsCollector | None = None
        self._stop = asyncio.Event()
        self._server: uvicorn.Server | None = None

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def _sleep_or_stop(self, delay: float) -> bool:
        """Sleep up to `delay`, returning True if a stop was requested meanwhile."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=max(0.0, delay))
            return True
        except asyncio.TimeoutError:
            return False

    # --- registration ---------------------------------------------------------

    async def register(self, client: httpx.AsyncClient) -> None:
        """Register (retrying until the manager answers), learning port + cadence + model."""
        req = RegisterRequest(
            node_id=self.node_id,
            address=self.host,
            incarnation=0,
            state=NodeState.REGISTERING,
            capabilities=NodeCapabilities(),
        )
        while not self._stop.is_set():
            try:
                resp = await client.post(
                    f"{self.manager_url}/register",
                    json=req.model_dump(mode="json"),
                    headers=self._headers,
                    timeout=10.0,
                )
                resp.raise_for_status()
                parsed = RegisterResponse.model_validate(resp.json())
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("register failed (%s); retrying", exc)
                if await self._sleep_or_stop(2.0):
                    return
                continue
            self.heartbeat_interval = parsed.heartbeat_interval_s or 10
            if parsed.node_spec is not None:
                self.preferred_port = parsed.node_spec.port
                self.spec_model = parsed.node_spec.model
            log.info(
                "registered as %s @ %s (preferred port %d, heartbeat=%ds)",
                self.node_id[:8], self.host, self.preferred_port, self.heartbeat_interval,
            )
            return

    # --- OpenAI-compatible serving --------------------------------------------

    def build_app(self) -> FastAPI:
        app = FastAPI(title=f"oumigo HF worker ({self.node_id[:8]})")

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models() -> dict:
            mid = self.engine.model_id if self.engine else "hf-model"
            return {"object": "list", "data": [{"id": mid, "object": "model"}]}

        @app.post("/v1/chat/completions")
        async def chat(request: Request) -> Response:
            return await self._handle(request, chat=True)

        @app.post("/v1/completions")
        async def completions(request: Request) -> Response:
            return await self._handle(request, chat=False)

        return app

    async def _handle(self, request: Request, *, chat: bool) -> Response:
        assert self.engine is not None
        body = await request.json()
        model = body.get("model") or self.engine.model_id

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return StreamingResponse(
                self._stream(body, model, chat=chat, include_usage=include_usage),
                media_type="text/event-stream",
            )

        self.inflight += 1
        try:
            text, p_toks, c_toks = await asyncio.to_thread(self.engine.generate, body, chat)
        finally:
            self.inflight -= 1
        log.info(
            "served %s: prompt=%d toks -> completion=%d toks",
            "chat" if chat else "completion", p_toks, c_toks,
        )
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

        The in-flight counter is held for the whole stream and released even on a
        mid-stream client disconnect (the generator's finally).
        """
        assert self.engine is not None
        self.inflight += 1
        cid = f"{'chatcmpl' if chat else 'cmpl'}-hf-{uuid4().hex[:12]}"
        created = int(time.time())
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
            if include_usage:
                c_toks = self.engine.count_tokens("".join(pieces))
                usage = {"prompt_tokens": input_len, "completion_tokens": c_toks,
                         "total_tokens": input_len + c_toks}
                obj = "chat.completion.chunk" if chat else "text_completion"
                yield _sse({"id": cid, "object": obj, "created": created, "model": model,
                            "choices": [], "usage": usage})
            yield "data: [DONE]\n\n"
            log.info("streamed %s: prompt=%d toks", "chat" if chat else "completion", input_len)
        finally:
            self.inflight -= 1

    # --- heartbeat + metrics --------------------------------------------------

    async def _send_heartbeat(self, client: httpx.AsyncClient) -> None:
        serving = self.node_state in (NodeState.SERVING, NodeState.DRAINING)
        run_state = None
        if serving:
            run_state = RunState.EXECUTING if self.inflight > 0 else RunState.IDLE
        req = HeartbeatRequest(
            node_id=self.node_id, node_state=self.node_state, run_state=run_state,
            vllm_port=self.port,  # tell the router which (preflight-negotiated) port to target
        )
        try:
            resp = await client.post(
                f"{self.manager_url}/heartbeat",
                json=req.model_dump(mode="json"),
                headers=self._headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            hb = HeartbeatResponse.model_validate(resp.json())
            if not hb.known:  # manager forgot us -> re-register
                log.info("unknown to manager; re-registering")
                await self.register(client)
        except (httpx.HTTPError, ValueError) as exc:
            log.debug("heartbeat failed: %s", exc)

    async def heartbeat_loop(self, client: httpx.AsyncClient) -> None:
        await self._send_heartbeat(client)  # announce our state immediately (INITIALIZING first)
        while not await self._sleep_or_stop(self.heartbeat_interval):
            await self._send_heartbeat(client)

    def start_metrics(self) -> None:
        """Start the real host+GPU metrics collector. vLLM metrics are ignored.

        `vllm_url=None` makes the vLLM scraper a no-op (empty `vllm:*`), and we never
        call `mark_serving()`, so no `vllm:start_timestamp` is stamped either — only
        genuine `worker:*` and `gpu:*` (plus `worker:start_timestamp`) are reported.
        """
        self.metrics = MetricsCollector(
            self.manager_url, self.token, self.node_id,
            vllm_url=None, worker_start=self.started_at,
        )
        self.metrics.start()
        log.info("real metrics collector started (worker:* + gpu:*, vllm:* ignored)")

    def stop_metrics(self) -> None:
        if self.metrics is not None:
            self.metrics.stop()
            self.metrics = None

    def stop(self) -> None:
        self.node_state = NodeState.STOPPED
        self._stop.set()
        if self._server is not None:
            self._server.should_exit = True


# --- OpenAI wire payloads (identical shape to the vLLM/stub responses) -----------


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


# --- orchestration --------------------------------------------------------------


async def _resolve_manager_url(manager_url: str | None, discover_timeout: float) -> str:
    """Return the manager URL: an explicit URL/env wins; otherwise mDNS-discover it.

    Mirrors the real worker: with no URL we browse the LAN (the manager must be
    advertising — i.e. not started with --no-mdns). The blocking browse runs off the
    event loop so a Ctrl-C during discovery still lands.
    """
    if manager_url:
        return manager_url
    log.info(
        "no --manager-url; discovering a manager on the LAN via mDNS (up to %.0fs)...",
        discover_timeout,
    )
    found = await asyncio.to_thread(discover_manager, discover_timeout)
    if not found:
        raise SystemExit(
            f"could not discover a manager on the LAN within {discover_timeout:.0f}s; "
            "set --manager-url / $OUMIGO_MANAGER_URL (and check the manager isn't --no-mdns)"
        )
    log.info("discovered manager at %s", found)
    return found


async def _run(args: argparse.Namespace) -> None:
    manager_url = await _resolve_manager_url(args.manager_url, args.discover_timeout)
    worker = HFWorker(manager_url=manager_url, token=args.token, host=args.host)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.stop)
        except NotImplementedError:  # pragma: no cover - non-Unix
            pass

    async with httpx.AsyncClient() as client:
        await worker.register(client)
        if worker._stop.is_set():
            return

        model_id = args.model or worker.spec_model
        if not model_id:
            raise SystemExit("no model to serve: pass --model or configure a model on the manager")
        worker.engine = HFEngine(
            model_id, quantize=args.quantize, dtype=args.dtype,
            device=args.device, max_new_tokens=args.max_new_tokens,
        )

        # Heartbeat first (INITIALIZING) so the manager sees the model-load window
        # rather than a silent node; then load the (slow) model off the event loop.
        worker.node_state = NodeState.INITIALIZING
        hb_task = asyncio.create_task(worker.heartbeat_loop(client))
        try:
            await asyncio.to_thread(worker.engine.load)
        except Exception:  # noqa: BLE001 - surface load failures (with traceback), don't hang
            log.exception("model load failed")
            worker.stop()
            hb_task.cancel()
            return

        # Preflight a free port (reusing the worker's find_free_port) and serve on it.
        try:
            worker.port = find_free_port(worker.host, worker.preferred_port)
        except PortUnavailable as exc:
            log.error("%s", exc)
            worker.stop()
            hb_task.cancel()
            return
        config = uvicorn.Config(
            worker.build_app(), host=worker.host, port=worker.port,
            log_config=None, access_log=False, log_level="warning",
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # we own shutdown
        worker._server = server
        serve_task = asyncio.create_task(server.serve())

        while not server.started and not serve_task.done() and not worker._stop.is_set():
            await asyncio.sleep(0.05)
        if not server.started:
            worker.stop()
            await asyncio.gather(hb_task, serve_task, return_exceptions=True)
            print("HF worker failed to start (see error above).")
            return

        worker.node_state = NodeState.SERVING
        worker.start_metrics()  # real worker:* + gpu:* sampling begins now
        print(
            f"HF worker up: model={model_id} on {worker.host}:{worker.port}, "
            f"manager={worker.manager_url}. Ctrl-C to stop."
        )
        await asyncio.gather(hb_task, serve_task, return_exceptions=True)

        # Clean shutdown: announce STOPPED once (client still open), then stop metrics.
        worker.node_state = NodeState.STOPPED
        await worker._send_heartbeat(client)
    worker.stop_metrics()
    log.info("HF worker exiting")


def _load_env_file(path: str) -> None:
    """Load `.env` into os.environ (real env wins), then normalize the HF_* vars.

    The shared loader keeps values verbatim, so we post-process the two that need it:
    expand `~`/`$VARS` in HF_HOME (huggingface_hub would otherwise make a literal `~`
    dir), and drop an empty HF_TOKEN so downloads read as anonymous rather than
    presenting a blank token. Done before transformers is imported, so HF_HOME/HF_TOKEN
    are in place when the model loads.
    """
    load_env_file(path)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = os.path.expanduser(os.path.expandvars(hf_home))
    if not os.environ.get("HF_TOKEN", "").strip():
        os.environ.pop("HF_TOKEN", None)


def main() -> None:
    # Load .env first so MODEL_ID/HF_HOME/HF_TOKEN shape the defaults below and are
    # inherited by transformers. A pre-parse honors --env-file before the main
    # parser's defaults (e.g. --model from MODEL_ID) are computed.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=os.environ.get("OUMIGO_ENV_FILE", ".env"))
    env_file = pre.parse_known_args()[0].env_file
    _load_env_file(env_file)
    # Reduce CUDA fragmentation OOMs; must be set before torch initializes CUDA.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(
        prog="worker_hf_test.py",
        description="Single real HF-Transformers worker for manager testing (no vLLM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-file", default=env_file,
        help="Path to a KEY=VALUE .env file, loaded at startup (working dir by default).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_ID") or os.environ.get("OUMIGO_HF_MODEL") or DEFAULT_MODEL,
        help="HF model id/path to serve (env: MODEL_ID; falls back to the manager's node_spec).",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Where to run the model.",
    )
    parser.add_argument(
        "--quantize", default="4bit", choices=["4bit", "8bit", "none"],
        help="bitsandbytes quantization (CUDA only; needed to fit large models in VRAM).",
    )
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype when NOT quantizing (auto: bf16/fp16 on GPU, fp32 on CPU).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Default generation length when the request omits max_tokens.",
    )
    parser.add_argument(
        "--host", default=get_lan_ip(),
        help="Address to bind and advertise (defaults to this host's LAN IP so the "
             "manager's router can reach it; use 127.0.0.1 for a same-host test).",
    )
    parser.add_argument(
        "--manager-url", default=os.environ.get("OUMIGO_MANAGER_URL"),
        help="Manager control-plane URL. If omitted, discover it on the LAN via mDNS.",
    )
    parser.add_argument(
        "--discover-timeout", type=float, default=DEFAULT_DISCOVER_TIMEOUT,
        help="Seconds to browse the LAN for a manager when --manager-url is omitted.",
    )
    parser.add_argument(
        "--token", default=os.environ.get("OUMIGO_MANAGER_TOKEN"),
        help="Shared bearer token, if the manager requires auth.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Log at DEBUG.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
