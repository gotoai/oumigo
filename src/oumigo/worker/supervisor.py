"""vLLMProcess — the subprocess seam.

Builds the `vllm serve` argv from a NodeSpec, spawns it, polls /health, exposes a
coarse run-state, and shuts down cleanly. `build_argv` is kept a pure function so
"given this spec, produce exactly this argv" is trivially unit-testable without
spawning anything — and this module never imports vllm (it's an optional,
GPU-only dependency); it only ever runs `vllm serve` as a child process.
"""

from __future__ import annotations

import errno
import logging
import socket
import subprocess

import httpx

from oumigo.config.spec import NodeSpec
from oumigo.protocol.states import RunState

log = logging.getLogger("oumigo.worker.vllm")


class PortUnavailable(RuntimeError):
    """No free port was found in the scanned range (all in use)."""


def find_free_port(host: str, preferred: int, max_tries: int = 64) -> int:
    """First bindable TCP port at/after ``preferred``, scanning up to ``max_tries``.

    A **preflight** so vLLM never loads a model on a doomed port — vLLM binds the
    port only after the (multi-minute) model load, so detecting a conflict by
    launching and watching it fail would cost a full load per attempt. Ports already
    in use (``EADDRINUSE``) are skipped; any other bind error is *not* a conflict and
    is re-raised rather than papered over by incrementing. ``host`` mirrors what vLLM
    binds (``spec.host``, typically ``0.0.0.0``) so a clash on any interface counts.
    """
    for candidate in range(preferred, preferred + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # mirror uvicorn
            try:
                sock.bind((host, candidate))
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    continue  # taken — try the next port
                raise  # a different failure (perms, bad host): not ours to retry
            return candidate
    raise PortUnavailable(
        f"no free port in [{preferred}, {preferred + max_tries}) on {host}"
    )


def build_argv(spec: NodeSpec, port: int | None = None) -> list[str]:
    """Translate a NodeSpec into an exact `vllm serve` argv. Pure function.

    ``port`` overrides ``spec.port`` — the coordinator may pick a different free port
    when the preferred one is taken on the host. Defaults to ``spec.port``.
    """
    argv = [
        "vllm",
        "serve",
        spec.model,
        "--host",
        spec.host,
        "--port",
        str(port if port is not None else spec.port),
        "--dtype",
        spec.dtype,
        "--tensor-parallel-size",
        str(spec.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(spec.gpu_memory_utilization),
    ]
    if spec.max_model_len is not None:
        argv += ["--max-model-len", str(spec.max_model_len)]
    if spec.download_dir:
        argv += ["--download-dir", spec.download_dir]
    argv += list(spec.extra_args)
    return argv


class VLLMProcess:
    """Spawns and supervises a single `vllm serve` child process.

    Health is polled locally over 127.0.0.1 (vLLM binds 0.0.0.0). The process
    inherits this worker's environment, so HF_HOME / VLLM_CACHE_ROOT / HF_TOKEN set
    in the worker's .env apply to the child unchanged.
    """

    def __init__(self, spec: NodeSpec, port: int | None = None) -> None:
        self.spec = spec
        self.port = port if port is not None else spec.port
        self.argv = build_argv(spec, self.port)
        self._proc: subprocess.Popen | None = None
        self._local = f"http://127.0.0.1:{self.port}"

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Spawn `vllm serve`. Non-blocking — readiness is observed via `is_healthy`."""
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError("vLLM process already running")
        log.info("starting vLLM: %s", " ".join(self.argv))
        # Inherit stdout/stderr so vLLM's own logs stream to the coordinator's console.
        self._proc = subprocess.Popen(self.argv)  # noqa: S603 - argv built from a typed spec

    def poll(self) -> int | None:
        """Return the child's exit code, or None while it is still running."""
        if self._proc is None:
            return None
        return self._proc.poll()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, grace_s: float = 30.0) -> int | None:
        """Ask vLLM to exit (SIGTERM), escalate to SIGKILL after `grace_s`.

        Returns the exit code (or None if it was never started).
        """
        if self._proc is None:
            return None
        if self._proc.poll() is not None:
            return self._proc.returncode
        log.info("stopping vLLM (SIGTERM, grace=%.0fs)", grace_s)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            log.warning("vLLM did not exit within %.0fs; sending SIGKILL", grace_s)
            self._proc.kill()
            self._proc.wait()
        return self._proc.returncode

    # --- observation ---------------------------------------------------------

    def is_healthy(self, timeout_s: float = 2.0) -> bool:
        """True once vLLM answers 200 on /health (model loaded, ready to serve)."""
        try:
            resp = httpx.get(f"{self._local}/health", timeout=timeout_s)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def run_state(self, timeout_s: float = 2.0) -> RunState:
        """Best-effort IDLE/EXECUTING from vLLM's /metrics.

        Coarse by design (see docs/worker-node-states.md): any in-flight or queued
        request reads as EXECUTING. Falls back to IDLE if /metrics is unavailable or
        unparseable — richer load telemetry is the metrics phase's job, not this.
        """
        try:
            resp = httpx.get(f"{self._local}/metrics", timeout=timeout_s)
            if resp.status_code != 200:
                return RunState.IDLE
            in_flight = _sum_gauges(
                resp.text, ("vllm:num_requests_running", "vllm:num_requests_waiting")
            )
            return RunState.EXECUTING if in_flight > 0 else RunState.IDLE
        except httpx.HTTPError:
            return RunState.IDLE


def _sum_gauges(prometheus_text: str, metric_names: tuple[str, ...]) -> float:
    """Sum the values of the named gauges across all label sets. Tolerant parser."""
    total = 0.0
    wanted = tuple(metric_names)
    for line in prometheus_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not line.startswith(wanted):
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return total
