"""The backend subprocess seam — vLLM or HF-transformers.

Both backends are supervised identically: build an argv from a NodeSpec, spawn it,
poll /health, read a coarse run-state off /metrics, and shut down cleanly. They
differ only in the argv (`build_argv` for `vllm serve`, `build_hf_argv` for the
in-repo `oumigo.service.worker.hf_server`); the HF server deliberately mimics vLLM's HTTP
surface (/health, /v1/*, vLLM-style /metrics) so `_ServerProcess` observes both the
same way. `build_argv`/`build_hf_argv` are pure functions so "given this spec,
produce exactly this argv" is unit-testable without spawning anything — and this
module never imports vllm/torch (both are optional, GPU-only); it only runs them as
child processes.
"""

from __future__ import annotations

import errno
import logging
import os
import signal
import socket
import subprocess
import sys

import httpx

from oumigo.common.proc import die_with_parent_preexec
from oumigo.config.spec import NodeSpec
from oumigo.protocol.states import RunState

log = logging.getLogger("oumigo.service.worker.vllm")


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


def build_hf_argv(spec: NodeSpec, port: int | None = None) -> list[str]:
    """Translate a NodeSpec into an argv for the in-repo HF-transformers server. Pure.

    Launches ``python -m oumigo.service.worker.hf_server`` — a small OpenAI-compatible server
    that generates with `transformers` and exposes the same HTTP surface as vLLM, so
    the coordinator supervises, routes to, and scrapes it identically. HF cache/token
    come from the inherited environment (HF_HOME / HF_TOKEN), not argv.
    """
    argv = [
        sys.executable,
        "-m",
        "oumigo.service.worker.hf_server",
        "--model",
        spec.model,
        "--host",
        spec.host,
        "--port",
        str(port if port is not None else spec.port),
        "--dtype",
        spec.dtype,
    ]
    if spec.max_model_len is not None:
        argv += ["--max-model-len", str(spec.max_model_len)]
    argv += list(spec.extra_args)
    return argv


class _ServerProcess:
    """Spawns and supervises one backend child (vLLM or HF server), observing it over HTTP.

    Health is polled locally over 127.0.0.1 (the backend binds 0.0.0.0). The child
    inherits this worker's environment, so HF_HOME / VLLM_CACHE_ROOT / HF_TOKEN set in
    the worker's .env apply unchanged. Subclasses only supply the argv + a `name`.
    """

    name = "backend"

    def __init__(self, argv: list[str], port: int) -> None:
        self.port = port
        self.argv = argv
        self._proc: subprocess.Popen | None = None
        self._pgid: int | None = None
        self._local = f"http://127.0.0.1:{self.port}"

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Spawn the backend. Non-blocking — readiness is observed via `is_healthy`."""
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError(f"{self.name} process already running")
        log.info("starting %s: %s", self.name, " ".join(self.argv))
        # Own session/process group (`start_new_session` -> setsid): the backend becomes
        # a group leader, and the extra processes it forks via multiprocessing (vLLM's
        # EngineCore + tensor-parallel workers) inherit this pgid. That lets `stop()` reap
        # the *whole* tree by group, so a hard-killed API server can't orphan an EngineCore
        # still holding VRAM. `die_with_parent_preexec` (PR_SET_PDEATHSIG) is the backstop:
        # if the coordinator itself dies, the kernel SIGTERMs the backend. stdout/stderr are
        # still inherited, so the backend's logs stream to the coordinator as before.
        self._proc = subprocess.Popen(  # noqa: S603 - argv built from a typed spec
            self.argv,
            start_new_session=True,
            preexec_fn=die_with_parent_preexec(),
        )
        # setsid makes the child its own group leader, so pgid == pid — capture it now
        # (it stays valid for killpg even after the leader exits, while members remain).
        self._pgid = self._proc.pid

    def poll(self) -> int | None:
        """Return the child's exit code, or None while it is still running."""
        if self._proc is None:
            return None
        return self._proc.poll()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _killpg(self, sig: int) -> None:
        """Send `sig` to the backend's whole process group (leader + EngineCore + workers).

        No-op if the group is already gone (`ProcessLookupError`) or the platform lacks
        process groups (`AttributeError`, e.g. Windows) — there `stop` falls back to
        single-process signals via the `_proc` handle.
        """
        if self._pgid is None:
            return
        try:
            os.killpg(self._pgid, sig)
        except (ProcessLookupError, AttributeError):
            pass

    def stop(self, grace_s: float = 30.0) -> int | None:
        """Stop the backend and its whole process group cleanly, then sweep stragglers.

        SIGTERM the group (the leader runs vLLM's graceful shutdown; EngineCore also gets
        it), escalate to SIGKILL after `grace_s`, then a final group SIGKILL reaps any
        member still alive even though the leader has already exited — so no EngineCore is
        left orphaned holding VRAM. Returns the exit code (or None if never started).
        """
        if self._proc is None:
            return None
        if self._proc.poll() is None:
            log.info("stopping %s group (SIGTERM, grace=%.0fs)", self.name, grace_s)
            self._killpg(signal.SIGTERM)
            try:
                self._proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                log.warning("%s did not exit within %.0fs; SIGKILL group", self.name, grace_s)
                self._killpg(signal.SIGKILL)
                self._proc.wait()
        # Final sweep: the leader is gone, but a lingering EngineCore in the same group
        # would otherwise survive (this is exactly what leaks VRAM). killpg by the leader's
        # old pid still reaches it while any member remains; ESRCH if the group is empty.
        self._killpg(signal.SIGKILL)
        return self._proc.returncode

    # --- observation ---------------------------------------------------------

    def is_healthy(self, timeout_s: float = 2.0) -> bool:
        """True once the backend answers 200 on /health (model loaded, ready to serve)."""
        try:
            resp = httpx.get(f"{self._local}/health", timeout=timeout_s)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def run_state(self, timeout_s: float = 2.0) -> RunState:
        """Best-effort IDLE/EXECUTING from the backend's /metrics.

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


class VLLMProcess(_ServerProcess):
    """Supervises a single `vllm serve` child process."""

    name = "vLLM"

    def __init__(self, spec: NodeSpec, port: int | None = None) -> None:
        self.spec = spec
        resolved = port if port is not None else spec.port
        super().__init__(build_argv(spec, resolved), resolved)


class HFProcess(_ServerProcess):
    """Supervises a single `oumigo.service.worker.hf_server` (HF-transformers) child process."""

    name = "HF-transformers"

    def __init__(self, spec: NodeSpec, port: int | None = None) -> None:
        self.spec = spec
        resolved = port if port is not None else spec.port
        super().__init__(build_hf_argv(spec, resolved), resolved)


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
