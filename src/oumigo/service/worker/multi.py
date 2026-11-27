"""Several profiles on one node: one worker process per profile, started in turn.

`WK_MODEL_PROFILE=harrier-oss-v1-0.6b@32k,ruri-v3-310m@8k` makes `oumigo worker run`
a small supervisor instead of a worker. It starts one `oumigo worker run` child per
profile, with the same options and environment except the single profile, and starts
the next only once the previous one serves. Two things make the order matter:

* vLLM checks at startup that the memory its profile asks for is free, so two backends
  loading at once can each see the other's memory as free and one fails later. List the
  largest profile first; each profile needs a fixed `gpu_memory_target_gib` to share.
* each worker takes the first free port from the fleet's (7001, 7002, ...), and the port
  is part of its node id. Starting in a fixed order keeps every profile on the same port,
  and so the same node, across restarts.

A child reports that it serves through a pipe (`--ready-fd`), so the supervisor never
guesses ports or polls. All children share the host id and the state directory, which
identity already allows (one host, several ports).

A child that exits with 0 was closed (a STOP from the manager): the others keep
serving, and the supervisor exits 0 once the last one has closed. A child that exits
otherwise — its backend failed for good, it could not start, or the manager refused it —
takes the set down: the supervisor stops the others and exits non-zero, so a container
platform restarts the whole set rather than leaving half of it running. A signal is
passed on to every child, which drains and stops its backend as a single worker would; a
second signal forces them, as it does a single worker.
"""

from __future__ import annotations

import logging
import os
import select
import signal
import subprocess
import sys
import time

from oumigo.common.envnames import WK_MODEL_PROFILE
from oumigo.common.proc import die_with_parent_preexec
from oumigo.service.worker.coordinator import DEFAULT_DRAIN_TIMEOUT_S, STOP_GRACE_S

log = logging.getLogger(__name__)

ENV_MODEL_PROFILE = WK_MODEL_PROFILE
# A child reports it serves with "serving <drain limit in seconds>\n".
READY = b"serving"
# Added to the longest drain limit + the backend's stop grace: how long the supervisor
# waits for children to stop on their own before it kills one that hangs.
STOP_MARGIN_S = 30.0
POLL_S = 1.0
WORKER_RUN = (sys.executable, "-m", "oumigo.cli.main", "worker", "run")


def parse_profiles(value: str | None) -> list[str]:
    """The profile ids in `value`, in order: none, one, or a comma-separated list."""
    ids = [p.strip() for p in (value or "").split(",") if p.strip()]
    if dupes := sorted({p for p in ids if ids.count(p) > 1}):
        raise ValueError(f"listed more than once: {', '.join(dupes)}")
    return ids


def ready_callback(fd: int):
    """For a child: the callback that tells the supervisor, once, that it serves, and its
    drain limit."""

    def ready(drain_timeout_s: float) -> None:
        try:
            os.write(fd, READY + f" {drain_timeout_s:g}\n".encode())
        except OSError as exc:  # the supervisor is gone; PDEATHSIG stops us shortly
            log.warning("could not report serving to the supervisor: %s", exc)
        finally:
            os.close(fd)

    return ready


def run_profiles(profiles: list[str], args: list[str], env: dict[str, str] | None = None,
                 command: tuple[str, ...] = WORKER_RUN) -> int:
    """Run `oumigo worker run <args>` once per profile, each started once the previous one
    serves, and supervise them. Returns the exit code for the whole set."""
    env = dict(os.environ if env is None else env)
    children: list[tuple[str, subprocess.Popen]] = []
    signalled: set[int] = set()  # pids already sent the stop signal
    drains: list[float] = []     # the children's drain limits, as they report them
    stopping = False

    def on_signal(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        for _, child in children:
            if child.poll() is None:
                child.send_signal(signum)  # a second signal forces, as for one worker
                signalled.add(child.pid)

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for number, profile in enumerate(profiles, start=1):
            if stopping:
                return 0
            log.info("starting %s (%d of %d)", profile, number, len(profiles))
            child, ready_fd = _start(command, profile, args, env)
            children.append((profile, child))
            ready, drain = _wait_ready(child, ready_fd, lambda: stopping)
            if not ready:
                if stopping:
                    return 0
                child.wait()
                if child.returncode == 0:
                    log.info("worker for %s closed before serving", profile)
                    continue
                log.error("worker for %s exited (code %s) before serving; stopping the others",
                          profile, child.returncode)
                return 1
            if drain is not None:
                drains.append(drain)
            log.info("%s serving (%d of %d)", profile, number, len(profiles))
        log.info("all %d profiles started", len(profiles))
        closed: set[int] = set()
        while not stopping:
            for profile, child in children:
                if child.pid in closed or child.poll() is None or stopping:
                    continue
                if child.returncode != 0:
                    log.error("worker for %s exited (code %s); stopping the others",
                              profile, child.returncode)
                    return 1
                closed.add(child.pid)
                log.info("worker for %s closed; %d still running", profile,
                         len(children) - len(closed))
            if len(closed) == len(children):
                log.info("every worker has closed")
                return 0
            time.sleep(POLL_S)
        return 0
    finally:
        _stop_all(children, signalled, _stop_grace(drains))
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _start(command: tuple[str, ...], profile: str, args: list[str],
           env: dict[str, str]) -> tuple[subprocess.Popen, int]:
    read_fd, write_fd = os.pipe()
    argv = [*command, *args, "--ready-fd", str(write_fd)]
    try:
        # Own session: a terminal's Ctrl-C reaches only this process, which passes it on
        # once (a second copy would force the child). PDEATHSIG covers a hard kill of it.
        child = subprocess.Popen(  # noqa: S603 - fixed argv (sys.executable + module path)
            argv, env={**env, ENV_MODEL_PROFILE: profile}, pass_fds=(write_fd,),
            start_new_session=True, preexec_fn=die_with_parent_preexec())
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)  # the child holds the only write end: EOF means it exited
    return child, read_fd


def _wait_ready(child: subprocess.Popen, fd: int, stopping) -> tuple[bool, float | None]:
    """(True, drain limit) once the child reports serving; (False, None) if it exits
    first or we are stopping. The drain limit is None if the child didn't send one."""
    with os.fdopen(fd, "rb", buffering=0) as pipe:
        message = b""
        while not stopping():
            readable, _, _ = select.select([pipe], [], [], POLL_S)
            if not readable:
                continue
            chunk = pipe.read(64)
            if not chunk:  # EOF: the child exited
                return message.startswith(READY), _parse_drain(message)
            message += chunk
            if b"\n" in message:
                return message.startswith(READY), _parse_drain(message)
    return False, None


def _parse_drain(message: bytes) -> float | None:
    words = message.split()
    try:
        return float(words[1]) if len(words) > 1 else None
    except ValueError:
        return None


def _stop_grace(drains: list[float]) -> float:
    """How long the children may take to stop: the longest drain limit, the backend's
    stop grace and a margin."""
    return max([*drains, DEFAULT_DRAIN_TIMEOUT_S]) + STOP_GRACE_S + STOP_MARGIN_S


def _stop_all(children: list[tuple[str, subprocess.Popen]], signalled: set[int],
              grace_s: float) -> None:
    """Stop the running children, newest first, and wait up to `grace_s` for them."""
    alive = [child for _, child in reversed(children) if child.poll() is None]
    for child in alive:
        if child.pid not in signalled:
            child.terminate()
    deadline = time.monotonic() + grace_s
    for child in alive:
        try:
            child.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            log.warning("worker pid %d did not stop in time; killing it", child.pid)
            child.kill()
            child.wait()
