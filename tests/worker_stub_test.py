#!/usr/bin/env python
"""Fake-worker fleet for testing the manager end-to-end (no GPU, no vLLM).

Spins up N independent stub workers in one process, all on 127.0.0.1. Each worker:
  * **preflight-negotiates its own port** — reusing the real worker's
    `find_free_port` — fanning out from the manager's `model.port` (or 7001 if none),
    then *reports that port* on its heartbeat so the router targets it. N stubs
    competing for ports on one host is exactly the multi-replica-per-host case;
  * self-registers, heartbeats SERVING (run_state reflects in-flight work), and
    reports dummy grid-aligned metrics — all independently;
  * answers the OpenAI API the router forwards (non-streaming JSON *and* SSE
    streaming when `stream=true`): it samples words *with replacement* from the
    prompt to a length 50%–1000% of the prompt, and waits `--request-latency`
    seconds ±50% before responding (streaming spreads that delay across tokens).

Run against a DEDICATED test manager on the same host — not your production one.
No port flag and no matching to configure: because each worker reports its own
negotiated port, the manager needn't even have a model configured. Give the test
manager its own control/data ports:

    # test_manager.yaml
    #   data_plane: {host: 127.0.0.1, port: 7017}
    #   dashboard:  {enabled: false}
    oumigo manager serve -c test_manager.yaml --no-mdns --host 127.0.0.1 --port 7016
    python tests/worker_stub_test.py --workers 3 --manager-url http://127.0.0.1:7016

NOTE: this is a runnable script, not a pytest module. It deliberately defines no
`test_*` functions and does nothing at import time, so pytest collects nothing
from it even though the filename matches `*_test.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import signal
import time
from collections.abc import AsyncIterator, Callable
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from oumigo.protocol.messages import (
    HeartbeatRequest,
    HeartbeatResponse,
    MetricPoint,
    MetricsReport,
    NodeCapabilities,
    RegisterRequest,
    RegisterResponse,
)
from oumigo.protocol.states import NodeState, RunState
from oumigo.service.worker.metrics import M_VLLM_START, M_WORKER_START, grid_timestamp
from oumigo.service.worker.supervisor import PortUnavailable, find_free_port  # reuse the worker's logic

log = logging.getLogger("oumigo.worker_stub")

_FALLBACK_WORDS = "the quick brown fox jumps over the lazy dog".split()
GRID_S = 5.0  # metric sampling grid, matching the real worker
HOST = "127.0.0.1"  # all fake workers share the loopback host; ports fan out per worker
DEFAULT_BASE_PORT = 7001  # preferred starting port when the manager has no model.port


# --- pure response synthesis ----------------------------------------------------


def _extract_prompt(body: dict, chat: bool) -> str:
    """Pull the prompt text out of a chat or completions request body."""
    if chat:
        parts: list[str] = []
        for msg in body.get("messages") or []:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):  # multimodal content blocks
                parts.extend(str(p.get("text", "")) for p in content if isinstance(p, dict))
        return " ".join(parts)
    prompt = body.get("prompt")
    if isinstance(prompt, list):
        return " ".join(map(str, prompt))
    return str(prompt or "")


def _synthesize(prompt: str) -> tuple[str, int, int]:
    """Sample words with replacement from `prompt` to 50%–1000% of its length.

    Returns (text, prompt_word_count, completion_word_count).
    """
    words = prompt.split() or _FALLBACK_WORDS
    factor = random.uniform(0.5, 10.0)
    target = max(1, round(len(words) * factor))
    text = " ".join(random.choice(words) for _ in range(target))
    return text, len(words), target


# --- one fake worker ------------------------------------------------------------


class StubWorker:
    """A single independent fake worker: registration, serving, heartbeat, metrics."""

    def __init__(
        self, index: int, manager_url: str, token: str | None, request_latency: float,
    ) -> None:
        self.index = index
        self.host = HOST
        self.manager_url = manager_url.rstrip("/")
        self.token = token
        self.request_latency = request_latency
        self.preferred_port = DEFAULT_BASE_PORT  # updated from node_spec at registration
        self.port: int | None = None             # actual port, chosen by preflight

        self.node_id = str(uuid4())
        self.started_at = time.time()          # worker:start_timestamp
        self.serving_since: float | None = None  # vllm:start_timestamp (set once serving)
        self.heartbeat_interval = 10
        self.inflight = 0
        self.req_count = 0

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
        """Register (retrying until the manager answers), learning preferred port + cadence."""
        req = RegisterRequest(
            node_id=self.node_id,
            address=self.host,
            incarnation=0,
            state=NodeState.REGISTERING,
            capabilities=NodeCapabilities(gpu="StubGPU", vram_gb=16.0),
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
                log.warning("worker %d register failed (%s); retrying", self.index, exc)
                if await self._sleep_or_stop(2.0):
                    return
                continue
            self.heartbeat_interval = parsed.heartbeat_interval_s or 10
            # Preferred port = the manager's model.port (like a real worker), else a
            # default. The actual port is preflight-negotiated and reported on the
            # heartbeat, so it need not match anything the manager configured.
            if parsed.node_spec is not None:
                self.preferred_port = parsed.node_spec.port
            log.info(
                "worker %d registered as %s @ %s (preferred port %d, heartbeat=%ds)",
                self.index, self.node_id[:8], self.host, self.preferred_port,
                self.heartbeat_interval,
            )
            return

    # --- OpenAI-compatible serving --------------------------------------------

    def build_app(self) -> FastAPI:
        app = FastAPI(title=f"oumigo stub worker {self.index}")

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models() -> dict:
            return {"object": "list", "data": [{"id": "stub-model", "object": "model"}]}

        @app.post("/v1/chat/completions")
        async def chat(request: Request) -> Response:
            return await self._handle(request, chat=True)

        @app.post("/v1/completions")
        async def completions(request: Request) -> Response:
            return await self._handle(request, chat=False)

        return app

    async def _handle(self, request: Request, *, chat: bool) -> Response:
        body = await request.json()
        prompt = _extract_prompt(body, chat)
        model = body.get("model") or "stub-model"
        text, p_words, c_words = _synthesize(prompt)
        delay = self.request_latency * (1.0 + random.uniform(-0.5, 0.5))  # ±50%

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return StreamingResponse(
                self._stream(
                    text, model, p_words, c_words, delay,
                    chat=chat, include_usage=include_usage,
                ),
                media_type="text/event-stream",
            )

        self.inflight += 1
        self.req_count += 1
        try:
            await asyncio.sleep(delay)
        finally:
            self.inflight -= 1
        log.info(
            "worker %d served %s: prompt=%d words -> completion=%d words in %.1fs",
            self.index, "chat" if chat else "completion", p_words, c_words, delay,
        )
        payload = (
            _chat_payload(text, model, p_words, c_words)
            if chat
            else _completion_payload(text, model, p_words, c_words)
        )
        return JSONResponse(payload)

    async def _stream(
        self, text: str, model: str, p_words: int, c_words: int, delay: float,
        *, chat: bool, include_usage: bool,
    ) -> AsyncIterator[str]:
        """Emit OpenAI SSE chunks, spreading `delay` across word-by-word tokens.

        Concatenating the streamed pieces reproduces the non-streaming text. The
        in-flight counter is held for the whole stream (and released even if the
        client disconnects mid-stream, via the generator's finally).
        """
        self.inflight += 1
        self.req_count += 1
        words = text.split()
        per_token = delay / max(1, len(words))
        cid = f"{'chatcmpl' if chat else 'cmpl'}-stub-{uuid4().hex[:12]}"
        created = int(time.time())
        try:
            if chat:
                yield _sse(_chat_chunk(cid, created, model, delta={"role": "assistant"}))
            for i, word in enumerate(words):
                await asyncio.sleep(per_token)
                piece = word if i == 0 else f" {word}"
                if chat:
                    yield _sse(_chat_chunk(cid, created, model, delta={"content": piece}))
                else:
                    yield _sse(_text_chunk(cid, created, model, text=piece))
            if chat:
                yield _sse(_chat_chunk(cid, created, model, delta={}, finish="stop"))
            else:
                yield _sse(_text_chunk(cid, created, model, text="", finish="stop"))
            if include_usage:
                usage = {"prompt_tokens": p_words, "completion_tokens": c_words,
                         "total_tokens": p_words + c_words}
                obj = "chat.completion.chunk" if chat else "text_completion"
                yield _sse({"id": cid, "object": obj, "created": created, "model": model,
                            "choices": [], "usage": usage})
            yield "data: [DONE]\n\n"
            log.info(
                "worker %d streamed %s: prompt=%d words -> completion=%d words in ~%.1fs",
                self.index, "chat" if chat else "completion", p_words, c_words, delay,
            )
        finally:
            self.inflight -= 1

    # --- heartbeat + metrics loops --------------------------------------------

    async def _send_heartbeat(self, client: httpx.AsyncClient) -> None:
        run_state = RunState.EXECUTING if self.inflight > 0 else RunState.IDLE
        req = HeartbeatRequest(
            node_id=self.node_id, node_state=NodeState.SERVING, run_state=run_state,
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
                log.info("worker %d unknown to manager; re-registering", self.index)
                await self.register(client)
        except (httpx.HTTPError, ValueError) as exc:
            log.debug("worker %d heartbeat failed: %s", self.index, exc)

    async def heartbeat_loop(self, client: httpx.AsyncClient) -> None:
        await self._send_heartbeat(client)  # announce SERVING immediately
        while not await self._sleep_or_stop(self.heartbeat_interval):
            await self._send_heartbeat(client)

    def _dummy_metrics(self) -> dict[str, float]:
        """A plausible grid slot of gauges + the lifetime start stamps."""
        load = self.inflight
        vram_total = 16 * 1024**3
        vram_used = vram_total * min(0.98, 0.35 + 0.05 * load + random.uniform(-0.02, 0.02))
        # GPU util tracks whether the (dummy) vLLM is serving: high when serving,
        # near-idle otherwise. Fractions [0.7,1.0]/[0.0,0.2] scaled onto the 0-100 pct.
        serving = self.serving_since is not None
        gpu_util_pct = round((random.uniform(0.85, 1.0) if serving else random.uniform(0.0, 0.2)) * 100, 2)
        metrics = {
            "gpu:#0_util_pct": gpu_util_pct,
            "gpu:#0_vram_total_bytes": float(vram_total),
            "gpu:#0_vram_used_bytes": round(vram_used, 1),
            "gpu:#0_vram_used_pct": round(100.0 * vram_used / vram_total, 2),
            "gpu:#0_temp_c": round(40 + 5 * load + random.uniform(-2, 2), 1),
            "gpu:#0_power_w": round(60 + 30 * load + random.uniform(-5, 5), 1),
            "worker:cpu_cores": float(os.cpu_count() or 8),
            "worker:cpu_util_pct": round(max(0.0, 5 + 10 * load + random.uniform(-2, 4)), 2),
            "worker:mem_total_bytes": float(32 * 1024**3),
            "worker:mem_used_pct": round(30 + random.uniform(-3, 3), 2),
            M_WORKER_START: self.started_at,
        }
        if self.serving_since is not None:
            metrics[M_VLLM_START] = self.serving_since
        if self.req_count:  # a light e2e-latency summary reflecting the configured latency
            base = self.request_latency
            metrics["vllm:e2e_request_latency_seconds_count"] = float(self.req_count)
            metrics["vllm:e2e_request_latency_seconds_mean"] = round(base, 3)
            metrics["vllm:e2e_request_latency_seconds_min"] = round(base * 0.5, 3)
            metrics["vllm:e2e_request_latency_seconds_median"] = round(base, 3)
            metrics["vllm:e2e_request_latency_seconds_max"] = round(base * 1.5, 3)
        return metrics

    async def metrics_loop(self, client: httpx.AsyncClient) -> None:
        while not self._stop.is_set():
            now = time.time()
            target = math.floor(now / GRID_S) * GRID_S + GRID_S
            if await self._sleep_or_stop(target - now):
                return
            report = MetricsReport(
                node_id=self.node_id,
                points=[
                    MetricPoint(timestamp=grid_timestamp(int(target)), metric=k, value=v)
                    for k, v in self._dummy_metrics().items()
                ],
            )
            try:
                resp = await client.post(
                    f"{self.manager_url}/metrics",
                    json=report.model_dump(mode="json"),
                    headers=self._headers,
                    timeout=10.0,
                )
                resp.raise_for_status()
            except (httpx.HTTPError, ValueError) as exc:
                log.debug("worker %d metrics report failed: %s", self.index, exc)

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.should_exit = True


def _chat_payload(text: str, model: str, p_toks: int, c_toks: int) -> dict:
    return {
        "id": f"chatcmpl-stub-{uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": p_toks,
            "completion_tokens": c_toks,
            "total_tokens": p_toks + c_toks,
        },
    }


def _completion_payload(text: str, model: str, p_toks: int, c_toks: int) -> dict:
    return {
        "id": f"cmpl-stub-{uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": p_toks,
            "completion_tokens": c_toks,
            "total_tokens": p_toks + c_toks,
        },
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


# --- fleet orchestration --------------------------------------------------------


async def _run_fleet(args: argparse.Namespace) -> None:
    workers = [
        StubWorker(
            index=i,
            manager_url=args.manager_url,
            token=args.token,
            request_latency=args.request_latency,
        )
        for i in range(args.workers)
    ]

    def stop_all() -> None:
        for w in workers:
            w.stop()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_all)
        except NotImplementedError:  # pragma: no cover - non-Unix
            pass

    async with httpx.AsyncClient() as client:
        for w in workers:  # register first so each learns its preferred port
            await w.register(client)

        # Bring workers up one at a time: each preflights a free port (reusing the
        # worker's find_free_port) and binds it before the next preflights — so ports
        # fan out from the preferred one with no start-up race. Same logic a real
        # worker runs; here N of them compete for ports on one host.
        tasks: list[asyncio.Task] = []
        next_port = workers[0].preferred_port if workers else DEFAULT_BASE_PORT
        for w in workers:
            try:
                w.port = find_free_port(w.host, next_port)
            except PortUnavailable as exc:
                log.error("worker %d: %s", w.index, exc)
                stop_all()
                await asyncio.gather(*tasks, return_exceptions=True)
                print("stub fleet failed to start (see error above).")
                return
            config = uvicorn.Config(
                w.build_app(), host=w.host, port=w.port,
                log_config=None, access_log=False, log_level="warning",
            )
            server = uvicorn.Server(config)
            server.install_signal_handlers = lambda: None  # fleet owns shutdown
            w._server = server
            st = asyncio.create_task(_serve_guarded(w, stop_all))
            tasks.append(st)

            while not server.started and not st.done() and not w._stop.is_set():
                await asyncio.sleep(0.05)  # let this port bind before the next preflight
            if not server.started:
                stop_all()
                await asyncio.gather(*tasks, return_exceptions=True)
                print("stub fleet failed to start (see error above).")
                return

            next_port = w.port + 1  # next worker scans upward from here
            w.serving_since = time.time()
            log.info("worker %d serving on %s:%d", w.index, w.host, w.port)
            tasks.append(asyncio.create_task(w.heartbeat_loop(client)))
            tasks.append(asyncio.create_task(w.metrics_loop(client)))

        ports = ", ".join(str(w.port) for w in workers)
        print(
            f"stub fleet up: {len(workers)} worker(s) on {HOST}:[{ports}], "
            f"latency={args.request_latency}s ±50%, manager={args.manager_url}. Ctrl-C to stop."
        )
        await asyncio.gather(*tasks, return_exceptions=True)


async def _serve_guarded(worker: StubWorker, stop_all: Callable[[], None]) -> None:
    """Run one worker's server, turning a bind failure into a clean fleet shutdown.

    uvicorn raises SystemExit(3) if the port is taken; caught here so it doesn't
    abort the whole event loop with a traceback. Preflight makes this rare (a TOCTOU
    race with another process grabbing the port between preflight and bind).
    """
    assert worker._server is not None
    try:
        await worker._server.serve()
    except (SystemExit, OSError) as exc:
        log.error(
            "worker %d could not bind %s:%s (%s) — a port conflict slipped past the "
            "preflight. Stopping the fleet.",
            worker.index, worker.host, worker.port, exc,
        )
        stop_all()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="worker_stub_test.py",
        description="Fake-worker fleet for manager testing (no GPU/vLLM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # renders "(default: …)"
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of stub workers.")
    parser.add_argument(
        "--request-latency", type=float, default=20.0,
        help="Base per-request latency in seconds (±50%% random noise applied).",
    )
    parser.add_argument(
        "--manager-url",
        default=os.environ.get("OUMIGO_MANAGER_URL", "http://127.0.0.1:7014"),
        help="Manager control-plane URL (same host as this stub).",
    )
    parser.add_argument(
        "--token", default=os.environ.get("OUMIGO_MANAGER_TOKEN"),
        help="Shared bearer token, if the manager requires auth.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Log at DEBUG.")
    args = parser.parse_args()

    if not 1 <= args.workers <= 250:
        parser.error("--workers must be between 1 and 250.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run_fleet(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
