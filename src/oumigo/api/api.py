"""Programmatic Python API — spawn/attach a manager and workers from code.

The CLI (`oumigo manager run`, `oumigo worker run`) is the operator-facing entry
point; this module is the *library* one. Each function is a thin wrapper over the
same machinery the CLI drives:

* **discover** an existing manager on the LAN (`oumigo.discovery`),
* **inject config** by writing a throwaway `manager.yaml` and passing it through
  the existing `manager serve` resolution (no new config path),
* **spawn** the relevant CLI subcommand as a child that *dies with this process*
  (`common.proc.die_with_parent_preexec` — Linux `PR_SET_PDEATHSIG`, the same
  mechanism the manager uses for its dashboard/server children), and
* **block until ready**, observing readiness over HTTP (`/healthz` for the
  manager, `/workers` state for a worker) rather than by parsing child logs.

Two handles are returned — `OumigoManager` and `OumigoWorker`. Both are context
managers and both register an ``atexit`` stop as a belt-and-suspenders companion
to the kernel-backed death signal.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from oumigo import discovery
from oumigo.api.manager.manager import OumigoManager
from oumigo.api.worker.worker import _WORKER_STOP_GRACE_S, OumigoWorker
from oumigo.common.proc import die_with_parent_preexec, terminate
from oumigo.protocol.states import NodeState

log = logging.getLogger("oumigo.api")

# vLLM/HF child processes inherit the worker's environment, so these env vars set
# on the spawned `worker run` reach the model server unchanged (see supervisor.py).
_ENV_MANAGER_TOKEN = "OUMIGO_MANAGER_TOKEN"
_ENV_HF_HOME = "HF_HOME"
_ENV_HF_TOKEN = "HF_TOKEN"
_ENV_VLLM_CACHE = "VLLM_CACHE_ROOT"
_ENV_MODEL_NAME = "MODEL_NAME"

# Default model block for the manager to hand workers. Mirrors the manager.yaml
# `model:` schema read by oumigo.service.manager.settings.build_node_spec.
DEFAULT_MODEL: dict[str, Any] = {
    "name": "google/gemma-4-E2B",
    "port": 7001,
    "dtype": "auto",
    "gpu_memory_utilization": 0.80,
    "max_model_len": 512,
    "max_concurrent_requests": 1,
}

# Sentinel for "argument not supplied" — lets config_file values fill only the slots
# the caller left blank, while any explicit value (even a default-looking one) wins.
_UNSET: Any = object()

# The most recent manager created/attached in this process. `oumigo_create_worker`
# falls back to it, so code that just called `oumigo_get_or_create_manager` can call
# `oumigo_create_worker()` with no args and reuse that manager rather than re-running
# mDNS (which is fragile inside a notebook and ambiguous when several managers exist).
_last_manager: OumigoManager | None = None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def oumigo_get_or_create_manager(
    bearer_token: str | None = _UNSET,
    config_file: str | Path | None = None,
    data_host: str = _UNSET,
    data_port: int = _UNSET,
    control_host: str = _UNSET,
    control_port: int = _UNSET,
    provider: str = _UNSET,
    model: dict[str, Any] | None = _UNSET,
    *,
    discover_timeout: float = 3.0,
    startup_timeout: float = 20.0,
) -> OumigoManager:
    """Return a manager, reusing a live LAN one or spawning a die-with-parent child.

    Settings resolve as **explicit call argument > config_file (YAML) > built-in
    default**: when ``config_file`` names a readable YAML file its values seed
    provider / data-plane bind / control-plane bind / model / bearer_token, and any
    argument passed in the call overrides the file. A missing or unparseable
    ``config_file`` is ignored (a warning is logged), so the call args alone suffice.

    Then:

    1. Browse the LAN (mDNS) for up to ``discover_timeout`` seconds. If a manager is
       advertising, return a handle to it (``owned=False``) — nothing is spawned.
    2. Otherwise write a throwaway ``manager.yaml`` (the resolved config, preserving
       any extra keys carried in from ``config_file``), spawn ``oumigo manager serve``
       as a child armed with ``PR_SET_PDEATHSIG`` so it exits when this process does,
       and block until its control plane answers ``/healthz`` (up to ``startup_timeout``).

    The ``model`` dict mirrors the manager.yaml ``model:`` block (name/port/dtype/
    gpu_memory_utilization/max_model_len/max_concurrent_requests/...).
    """
    global _last_manager

    # Resolve effective settings: explicit call arg > config_file value > default.
    cfg = _load_config_file(config_file)
    data_cfg = cfg.get("data_plane") or {}
    control_cfg = cfg.get("control_plane") or {}
    bearer_token = _pick(bearer_token, cfg.get("bearer_token"), None)
    provider = _pick(provider, cfg.get("provider"), "LAN")
    data_host = _pick(data_host, data_cfg.get("host"), "0.0.0.0")  # noqa: S104 - bind all interfaces
    data_port = int(_pick(data_port, data_cfg.get("port"), 7012))
    control_host = _pick(control_host, control_cfg.get("host"), "0.0.0.0")  # noqa: S104 - bind all
    control_port = int(_pick(control_port, control_cfg.get("port"), 7014))
    model = _pick(model, cfg.get("model"), DEFAULT_MODEL)

    found = discovery.discover_manager(discover_timeout)
    if found:
        log.info("found a live manager on the LAN at %s; reusing it", found)
        host = found.split("://", 1)[-1].split(":", 1)[0]
        _last_manager = OumigoManager(
            control_url=found.rstrip("/"),
            data_url=f"http://{host}:{data_port}",
            token=bearer_token,
            provider=provider,
            owned=False,
        )
        return _last_manager

    log.info("no manager on the LAN; spawning one as a child of this process")
    config_path = _write_manager_config(
        provider=provider,
        data_host=data_host,
        data_port=data_port,
        control_host=control_host,
        control_port=control_port,
        model=model,
        base=cfg,
    )

    env = dict(os.environ)
    if bearer_token:
        env[_ENV_MANAGER_TOKEN] = bearer_token  # secret via env, never argv
    argv = [
        sys.executable, "-m", "oumigo.cli.main", "manager", "serve",
        "--config-file", config_path,
        "--host", control_host,
        "--port", str(control_port),
    ]
    child = _spawn_child(argv, env)

    # Probe over loopback even when bound to 0.0.0.0 — this process is co-located.
    probe_host = "127.0.0.1" if control_host in ("0.0.0.0", "") else control_host
    control_url = f"http://{probe_host}:{control_port}"
    manager = OumigoManager(
        control_url=control_url,
        data_url=f"http://{probe_host}:{data_port}",
        token=bearer_token,
        provider=provider,
        owned=True,
        _child=child,
        _config_path=config_path,
    )
    atexit.register(manager.stop)

    if not _wait_manager_healthy(child, manager, startup_timeout):
        manager.stop()
        raise RuntimeError(
            f"manager control plane did not become healthy within {startup_timeout:.0f}s "
            f"(check the child's logs)"
        )
    log.info("manager healthy at %s (data plane %s)", control_url, manager.data_url)
    _last_manager = manager
    return manager


def oumigo_create_worker(
    bearer_token: str | None = None,
    hf_home: str = "~/.hf_cache",
    vllm_cache_root: str = "~/.vllm_cache",
    hf_token: str | None = None,
    model_name: str | None = None,
    *,
    manager: OumigoManager | None = None,
    backend: str = "vllm",
    manager_url: str | None = None,
    discover_timeout: float = 10.0,
    serving_timeout: float | None = None,
    poll_interval: float = 2.0,
) -> OumigoWorker:
    """Spawn a worker child and block until its replica is SERVING.

    Always spawns a fresh worker on this host (create semantics). The child is armed
    with ``PR_SET_PDEATHSIG`` so it dies with this process. Blocks until the manager
    reports the new node as ``SERVING`` — this includes the (multi-minute, or on a first
    download much longer) model load.

    ``serving_timeout`` is **None by default: wait indefinitely** — the API takes no view
    on how long a load/download should take; watch the worker's logs and interrupt if you
    want to give up. Pass a number of seconds to impose your own cap (on which the worker
    is torn down and a ``RuntimeError`` is raised). Either way the wait still ends early on
    a genuine failure — the child process exiting, or the node reaching ``failed`` — since
    those are definitive, not duration judgments.

    The manager to register with is resolved as: ``manager`` handle > ``manager_url`` >
    the last manager created in this process (via ``oumigo_get_or_create_manager``) >
    mDNS discovery. Preferring the handle avoids re-running mDNS, which is unreliable
    inside a Jupyter notebook and ambiguous when several managers advertise. A passed
    ``manager`` also supplies its ``bearer_token`` when one isn't given here.

    ``hf_home`` / ``vllm_cache_root`` / ``hf_token`` / ``model_name`` are passed to the
    replica via the child's environment (``HF_HOME`` / ``VLLM_CACHE_ROOT`` / ``HF_TOKEN``
    / ``MODEL_NAME``). When ``model_name`` is None the worker serves whatever model the
    manager configured.
    """
    if manager is None:
        manager = _last_manager  # reuse the manager this process created/attached
    if manager is not None:
        if not manager_url:
            manager_url = manager.control_url
        if bearer_token is None:
            bearer_token = manager.token
    if not manager_url:
        manager_url = discovery.discover_manager(discover_timeout)
        if not manager_url:
            raise RuntimeError(
                f"no manager found on the LAN within {discover_timeout:.0f}s; start one "
                f"first with oumigo_get_or_create_manager (its handle is reused "
                f"automatically), or pass manager=/manager_url="
            )
    manager_url = manager_url.rstrip("/")

    address = discovery.get_lan_ip()
    before = _worker_incarnations(manager_url, address)  # snapshot to spot our fresh (re)start

    env = dict(os.environ)
    env[_ENV_HF_HOME] = os.path.expanduser(os.path.expandvars(hf_home))
    env[_ENV_VLLM_CACHE] = os.path.expanduser(os.path.expandvars(vllm_cache_root))
    if hf_token:
        env[_ENV_HF_TOKEN] = hf_token
    if model_name:
        env[_ENV_MODEL_NAME] = model_name
    if bearer_token:
        env[_ENV_MANAGER_TOKEN] = bearer_token

    argv = [
        sys.executable, "-m", "oumigo.cli.main", "worker", "run",
        "--manager-url", manager_url,
        "--backend", backend,
    ]
    child = _spawn_child(argv, env)

    worker = _wait_worker_serving(
        child, manager_url, address, before, serving_timeout, poll_interval, backend, model_name
    )
    atexit.register(worker.stop)
    log.info(
        "worker %s SERVING at %s:%d (model=%s)",
        worker.node_id, worker.address, worker.port, worker.model,
    )
    return worker


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _pick(passed: Any, yaml_val: Any, default: Any) -> Any:
    """Effective value: explicit call arg (not ``_UNSET``) > config-file value > default."""
    if passed is not _UNSET:
        return passed
    if yaml_val is not None:
        return yaml_val
    return default


def _load_config_file(config_file: str | Path | None) -> dict[str, Any]:
    """Load a manager-config YAML into a dict, or ``{}`` if absent/unreadable/not a mapping.

    Honors the "if config_file is a readable YAML file" contract: a missing file, a
    parse error, or a non-mapping document is ignored (with a warning) rather than
    raising, so an explicit call argument on its own is still sufficient.
    """
    if not config_file:
        return {}
    path = Path(config_file).expanduser()
    if not path.is_file():
        log.warning("config_file %s is not a readable file; ignoring it", path)
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        log.warning("config_file %s could not be read as YAML (%s); ignoring it", path, exc)
        return {}
    if not isinstance(data, dict):
        log.warning("config_file %s did not parse to a mapping; ignoring it", path)
        return {}
    return data


def _write_manager_config(
    *,
    provider: str,
    data_host: str,
    data_port: int,
    model: dict[str, Any],
    control_host: str | None = None,
    control_port: int | None = None,
    base: dict[str, Any] | None = None,
) -> str:
    """Write a throwaway manager.yaml and return its path (removed on manager.stop()).

    Starts from ``base`` (an already-loaded config_file, so extra keys like ``dashboard``
    / ``heartbeat`` survive) and overlays the resolved provider / binds / model.
    """
    config: dict[str, Any] = dict(base or {})
    config["provider"] = provider
    config["data_plane"] = {"host": data_host, "port": int(data_port)}
    if control_host is not None and control_port is not None:
        config["control_plane"] = {"host": control_host, "port": int(control_port)}
    config["model"] = dict(model)
    config.pop("bearer_token", None)  # never persist the shared secret into a temp file

    fd, path = tempfile.mkstemp(prefix="oumigo-manager-", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    log.debug("wrote temporary manager config to %s", path)
    return path


def _spawn_child(argv: list[str], env: dict[str, str]) -> subprocess.Popen:
    """Spawn a CLI subcommand as a child that dies with this process.

    New session isolates it from a terminal Ctrl-C (this process owns its lifecycle),
    while ``PR_SET_PDEATHSIG`` guarantees it exits even if this process is hard-killed.
    """
    return subprocess.Popen(  # noqa: S603 - fixed argv (sys.executable + module path)
        argv, env=env, start_new_session=True, preexec_fn=die_with_parent_preexec()
    )


def _wait_manager_healthy(
    child: subprocess.Popen, manager: OumigoManager, timeout: float
) -> bool:
    """Poll the control plane until healthy, failing fast if the child exits first."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if child.poll() is not None:
            return False  # exited before becoming healthy
        if manager.is_healthy(timeout_s=0.5):
            return True
        time.sleep(0.2)
    return False


def _worker_incarnations(manager_url: str, address: str) -> dict[str, int]:
    """Snapshot ``{node_id: incarnation}`` for workers already at ``address``.

    ``node_id`` is a deterministic hash of host:port, so a re-launched worker keeps its
    id and the manager may still remember a previous one at the same address. We can't
    identify our fresh worker by "a node_id that wasn't here before" — that id can
    already be present. Instead we detect its ``incarnation`` (bumped and persisted on
    every start) rising past this snapshot, or a brand-new node_id.
    """
    out: dict[str, int] = {}
    for rec in _list_workers(manager_url):
        node_id = rec.get("node_id")
        if node_id and rec.get("address") == address:
            out[node_id] = int(rec.get("incarnation") or 0)
    return out


def _wait_worker_serving(
    child: subprocess.Popen,
    manager_url: str,
    address: str,
    before: dict[str, int],
    timeout: float | None,
    poll_interval: float,
    backend: str,
    model_name: str | None,
) -> OumigoWorker:
    """Block until *our* fresh worker on this host reports SERVING.

    "Ours" is a worker at ``address`` that is either brand-new (node_id not in the
    ``before`` snapshot) or has a higher ``incarnation`` than we saw — so a manager that
    still remembers a prior worker at the same host:port doesn't mask the one we just
    spawned. Terminates the child and raises on a definitive failure — the child exiting
    or the worker reaching FAILED — so a failed launch never leaks a half-started replica.
    ``timeout`` is a wall-clock cap in seconds, or **None to wait indefinitely** (the
    caller owns the give-up decision); a finite cap that elapses also tears the child down.
    """
    deadline = None if timeout is None else time.time() + timeout
    while deadline is None or time.time() < deadline:
        if child.poll() is not None:
            terminate(child, grace_s=_WORKER_STOP_GRACE_S)
            raise RuntimeError(
                f"worker process exited (code={child.returncode}) before SERVING; "
                f"check its logs"
            )
        for rec in _list_workers(manager_url):
            node_id = rec.get("node_id")
            if rec.get("address") != address or not node_id:
                continue
            prior = before.get(node_id)
            incarnation = int(rec.get("incarnation") or 0)
            if prior is not None and incarnation <= prior:
                continue  # a pre-existing worker the manager still remembers, not ours
            # /workers serializes state as the lowercase NodeState value — compare against
            # NodeState.*.value (not an uppercase literal), so the match can't silently fail.
            state = str(rec.get("state") or "").lower()
            if state == NodeState.SERVING.value:
                return OumigoWorker(
                    manager_url=manager_url,
                    address=address,
                    port=int(rec.get("port") or 0),
                    model=rec.get("model") or model_name or "",
                    backend=rec.get("backend") or backend,
                    node_id=node_id,
                    _child=child,
                )
            if state == NodeState.FAILED.value:
                terminate(child, grace_s=_WORKER_STOP_GRACE_S)
                raise RuntimeError(
                    f"worker {node_id} reached FAILED (the replica failed to start, e.g. "
                    f"the vLLM restart policy was exhausted); check its logs"
                )
        time.sleep(poll_interval)

    terminate(child, grace_s=_WORKER_STOP_GRACE_S)
    raise RuntimeError(
        f"worker did not reach SERVING within {timeout:.0f}s (model load too slow or "
        f"the replica failed to start); check its logs"
    )


def _list_workers(manager_url: str) -> list[dict]:
    try:
        resp = httpx.get(f"{manager_url}/workers", timeout=5.0)
        resp.raise_for_status()
        return list(resp.json().get("workers", []))
    except httpx.HTTPError:
        return []
