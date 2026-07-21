"""Spawn the control-plane server as a child process and run the console over HTTP.

Used by `oumigo manager run` on a TTY: the server lives in its own process (fault
isolation, no async-in-thread), and the console attaches to it as an HTTP client.
The child is started in a new session so a terminal Ctrl-C reaches only the console,
which owns the child's lifecycle — and armed with `PR_SET_PDEATHSIG`, so a hard kill
of the console still takes the server (and, transitively, its dashboard) down rather
than orphaning it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx

from oumigo.common.proc import die_with_parent_preexec, terminate
from oumigo.manager.console import ManagerConsole


def run_with_child_server(
    host: str,
    port: int,
    token: str | None,
    forward_args: list[str],
    verbose: bool,
    startup_timeout: float = 15.0,
) -> None:
    env = dict(os.environ)
    if token is not None:
        env["OUMIGO_MANAGER_TOKEN"] = token  # pass secret via env, never argv

    cmd = [sys.executable, "-m", "oumigo.cli.main", "manager", "serve", *forward_args]
    child = subprocess.Popen(  # noqa: S603 - fixed argv
        cmd, env=env, start_new_session=True, preexec_fn=die_with_parent_preexec()
    )

    base_url = f"http://127.0.0.1:{port}"
    if not _wait_healthy(child, base_url, startup_timeout):
        print("error: control-plane server did not start", file=sys.stderr)
        terminate(child)
        raise SystemExit(1)

    console = ManagerConsole(base_url=base_url, server_pid=child.pid, verbose=verbose)
    try:
        console.run()
    finally:
        # Stop the server child on any console exit; it in turn stops the dashboard.
        terminate(child)


def _wait_healthy(child: subprocess.Popen, base_url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if child.poll() is not None:  # child exited before becoming healthy
            return False
        try:
            httpx.get(base_url + "/healthz", timeout=0.5)
            return True
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(0.1)
    return False
