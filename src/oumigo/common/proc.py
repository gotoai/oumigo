"""Child-process helpers: spawn helpers that die with their parent, and stop them cleanly.

The manager spawns helper processes — the reporting-plane dashboard, and (in console
mode) the control-plane server itself. Two mechanisms together guarantee none of them
outlive the parent and orphan:

1. **Graceful, in-process** (`terminate`): the parent runs this on shutdown — SIGTERM,
   then SIGKILL after a grace period. Covers normal exit, Ctrl-C, and SIGTERM.
2. **Kernel-backed** (`die_with_parent_preexec`): Linux ``PR_SET_PDEATHSIG`` arms the
   child to receive SIGTERM the instant its parent dies — so even a ``kill -9`` on the
   parent (which leaves no chance to run mechanism 1) still takes the child down. This
   is what prevents the orphaned dashboards a hard-killed manager would otherwise leave.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
from collections.abc import Callable

_PR_SET_PDEATHSIG = 1  # value of PR_SET_PDEATHSIG from <sys/prctl.h>
_libc = ctypes.CDLL("libc.so.6", use_errno=True) if sys.platform == "linux" else None


def die_with_parent_preexec() -> Callable[[], None] | None:
    """Return a `preexec_fn` that makes the spawned child die when THIS parent dies.

    Pass straight to ``subprocess.Popen(preexec_fn=...)``. Linux-only (uses prctl
    ``PR_SET_PDEATHSIG``); returns None elsewhere, so on other platforms the child just
    relies on the graceful `terminate` path. The fork→exec race (parent dies before the
    child arms the signal) is closed by comparing against the *real* parent pid — not
    ``getppid() == 1``, which is wrong under a subreaper like ``systemd --user`` (there
    an orphan reparents to the subreaper's pid, not init's 1).
    """
    if _libc is None:
        return None
    parent_pid = os.getpid()

    def _preexec() -> None:
        _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
        if os.getppid() != parent_pid:  # parent already exited during fork->exec
            os._exit(0)

    return _preexec


def terminate(child: subprocess.Popen | None, grace_s: float = 5.0) -> None:
    """Stop a child cleanly: SIGTERM, then SIGKILL if it outlives `grace_s`. No-op if
    the child was never started or has already exited."""
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
