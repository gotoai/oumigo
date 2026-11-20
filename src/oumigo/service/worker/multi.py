"""Several profiles on one node: one worker process per profile, started in turn.

`OUMIGO_MODEL_PROFILE=harrier-oss-v1-0.6b@32k,ruri-v3-310m@8k` makes `oumigo worker run`
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

The children are one unit: if any exits that was not asked to — its backend failed for
good, or it could not start — the supervisor stops the others and exits non-zero, so a
container platform restarts the whole set rather than leaving half of it running. A
signal is passed on to every child, which drains and stops its backend as a single
worker would; a second signal forces them, as it does a single worker.
"""

from __future__ import annotations

import logging
import os
import select
import signal
import subprocess
import sys
import time

from oumigo.common.proc import die_with_parent_preexec

log = logging.getLogger(__name__)

ENV_MODEL_PROFILE = "OUMIGO_MODEL_PROFILE"
READY = b"serving\n"
# Longer than a child needs to stop on its own: drain (up to 120 s), then the backend's
# stop grace (30 s). Only a child that hangs past it is killed.
STOP_GRACE_S = 180.0
POLL_S = 1.0
WORKER_RUN = (sys.executable, "-m", "oumigo.cli.main", "worker", "run")


def parse_profiles(value: str | None) -> list[str]:
    """The profile ids in `value`, in order: none, one, or a comma-separated list."""
    ids = [p.strip() for p in (value or "").split(",") if p.strip()]
    if dupes := sorted({p for p in ids if ids.count(p) > 1}):
        raise ValueError(f"listed more than once: {', '.join(dupes)}")
    return ids


def ready_callback(fd: int):
    """For a child: the callback that tells the supervisor, once, that it serves."""

    def ready() -> None:
        try:
            os.write(fd, READY)
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
            if not _wait_ready(child, ready_fd, lambda: stopping):
                if stopping:
                    return 0
                child.wait()
                log.error("worker for %s exited (code %s) before serving; stopping the others",
                          profile, child.returncode)
                return 1
            log.info("%s serving (%d of %d)", profile, number, len(profiles))
        log.info("all %d profiles serving", len(profiles))
        while not stopping:
            for profile, child in children:
                if child.poll() is not None and not stopping:
                    log.error("worker for %s exited (code %s); stopping the others",
                              profile, child.returncode)
                    return 1
            time.sleep(POLL_S)
        return 0
    finally:
        _stop_all(children, signalled)
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


def _wait_ready(child: subprocess.Popen, fd: int, stopping) -> bool:
    """True once the child reports serving; False if it exits first or we are stopping."""
    with os.fdopen(fd, "rb", buffering=0) as pipe:
        while not stopping():
            readable, _, _ = select.select([pipe], [], [], POLL_S)
            if readable:
                return pipe.read(len(READY)) == READY  # b"": the child exited
    return False


def _stop_all(children: list[tuple[str, subprocess.Popen]], signalled: set[int]) -> None:
    """Stop the running children, newest first, and wait for them to finish."""
    alive = [child for _, child in reversed(children) if child.poll() is None]
    for child in alive:
        if child.pid not in signalled:
            child.terminate()
    deadline = time.monotonic() + STOP_GRACE_S
    for child in alive:
        try:
            child.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            log.warning("worker pid %d did not stop in time; killing it", child.pid)
            child.kill()
            child.wait()
