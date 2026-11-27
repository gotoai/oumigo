"""Console entrypoint: ``oumigo <subcommand>``.

One CLI with subcommands grouped by layer:
  * ``oumigo worker ...``  — worker-node coordinator (L1)
  * ``oumigo manager ...`` — manager: control plane + router (L3)
  * ``oumigo cluster ...`` — cluster-level operations
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from oumigo import __version__
from oumigo.common import envnames as en

app = typer.Typer(help="OumiGo — run and manage vLLM replica fleets.", no_args_is_help=True)

worker_app = typer.Typer(help="Worker-node coordinator (L1).", no_args_is_help=True)
manager_app = typer.Typer(help="Manager node: control plane + router (L3).", no_args_is_help=True)


@manager_app.callback()
def _manager_env_check() -> None:
    """Refuse retired variable names before any manager command reads its environment."""
    from oumigo.common.envnames import MANAGER, refuse_retired

    refuse_retired(MANAGER)
cluster_app = typer.Typer(help="Cluster-level operations.", no_args_is_help=True)

app.add_typer(worker_app, name="worker")
app.add_typer(manager_app, name="manager")
app.add_typer(cluster_app, name="cluster")

token_app = typer.Typer(help="Worker join tokens (run on the manager host).", no_args_is_help=True)
node_app = typer.Typer(help="Registered worker nodes (run on the manager host).", no_args_is_help=True)
manager_app.add_typer(token_app, name="token")
apikey_app = typer.Typer(help="Client API keys for the router (run on the manager host).",
                         no_args_is_help=True)
manager_app.add_typer(node_app, name="node")
manager_app.add_typer(apikey_app, name="apikey")


@app.command("version")
def version() -> None:
    """Print the OumiGo version."""
    typer.echo(__version__)


@worker_app.command("run")
def worker_run(
    manager_url: Optional[str] = typer.Option(
        None,
        "--manager-url",
        envvar=en.WK_MANAGER_URL,
        help="Manager control-plane URL. If unset, discover on the LAN via mDNS.",
    ),
    token_file: Optional[Path] = typer.Option(
        None, "--token-file", help="Path to a file containing the join token."
    ),
    state_dir: Optional[Path] = typer.Option(
        None, "--state-dir", help="Directory holding the persisted node identity."
    ),
    transport: Optional[str] = typer.Option(
        None,
        "--transport",
        envvar=en.WK_TRANSPORT,
        help="How the router reaches this worker: 'direct' (router dials this host; LAN) or "
        "'tunnel' (this worker dials out to the manager; no inbound port needed). "
        "Default: direct.",
    ),
    discover_timeout: Optional[float] = typer.Option(
        None,
        "--discover-timeout",
        envvar=en.WK_DISCOVER_TIMEOUT,
        help="Seconds to wait for mDNS discovery of the manager (when no --manager-url). "
        "Default: 60.",
    ),
    env_file: Optional[Path] = typer.Option(
        Path(".env"),
        "--env-file",
        help="Load KEY=VALUE vars from this file into the environment (inherited by the "
        "backend). Existing environment variables win. Skipped if the file is absent.",
    ),
    gpu_id: Optional[str] = typer.Option(
        None,
        "--gpu-id",
        envvar=en.WK_GPU_ID,
        help="The GPU this worker runs on, as nvidia-smi numbers them (from 0) or by UUID "
        "(GPU-…); comma-separated for a model sharded over several. Default: "
        "$CUDA_VISIBLE_DEVICES if set, else GPU 0.",
    ),
    backend: str = typer.Option(
        "vllm",
        "--backend",
        help="Inference backend to supervise: 'vllm' (default) or 'transformer' "
        "(HF-transformers server; max_concurrent_requests forced to 1).",
    ),
    ready_fd: Optional[int] = typer.Option(
        None, "--ready-fd", hidden=True,
        help="Internal, for the multi-profile supervisor: write to this file descriptor "
        "once the backend serves.",
    ),
) -> None:
    """Load the env file, resolve identity, find the manager, register, and supervise the backend.

    The env file is applied before anything reads the environment, so the node's own
    vars (WK_MODEL_PROFILE, WK_MODEL_STORAGE_LOCATION / HF_HOME / HF_TOKEN, plus VLLM_*)
    reach the coordinator and the backend child process it spawns.

    WK_MODEL_PROFILE may list several profiles, comma-separated and largest first:
    one worker per profile then runs on this node, each started once the previous one
    serves (see oumigo.service.worker.multi).

    WK_DRAIN_TIMEOUT (seconds) caps how long a stopping worker waits for requests in
    progress; by default the profile's agent.turn_timeout, else 120. Exit code: 0 after a
    normal stop (closed by the manager, or a signal), non-zero on failure.
    """
    import logging

    from oumigo.common.env import load_env_file
    from oumigo.common.envnames import WK_JOIN_TOKEN, WORKER, refuse_retired
    from oumigo.service.manager.auth import read_token_file
    from oumigo.service.worker.coordinator import BACKENDS, run_worker
    from oumigo.service.worker.multi import parse_profiles, ready_callback, run_profiles

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if backend not in BACKENDS:
        typer.echo(f"error: unknown --backend {backend!r}; choose {' or '.join(BACKENDS)}.", err=True)
        raise typer.Exit(2)

    if env_file is not None:
        load_env_file(env_file)
    refuse_retired(WORKER)
    # typer resolves `envvar=` while parsing, before the env file is loaded: fill the
    # options a flag or the real environment left unset from the env file's values.
    manager_url = manager_url or os.environ.get(en.WK_MANAGER_URL, "").strip() or None
    transport = transport or os.environ.get(en.WK_TRANSPORT, "").strip() or "direct"
    gpu_id = gpu_id or os.environ.get(en.WK_GPU_ID, "").strip() or None
    if discover_timeout is None:
        try:
            discover_timeout = float(os.environ.get(en.WK_DISCOVER_TIMEOUT) or 60.0)
        except ValueError:
            typer.echo("error: WK_DISCOVER_TIMEOUT must be a number of seconds.", err=True)
            raise typer.Exit(2)

    if transport not in ("direct", "tunnel"):
        typer.echo(f"error: unknown --transport {transport!r}; choose direct or tunnel.", err=True)
        raise typer.Exit(2)

    try:
        profiles = parse_profiles(os.environ.get(en.WK_MODEL_PROFILE))
    except ValueError as exc:
        typer.echo(f"error: WK_MODEL_PROFILE: {exc}", err=True)
        raise typer.Exit(2)
    if len(profiles) > 1:
        # Each child resolves its join token and the manager itself, from these options
        # and the environment it inherits.
        args = ["--transport", transport, "--discover-timeout", str(discover_timeout),
                "--backend", backend]
        for flag, value in (("--manager-url", manager_url), ("--token-file", token_file),
                            ("--state-dir", state_dir), ("--env-file", env_file),
                            ("--gpu-id", gpu_id)):
            if value is not None:
                args += [flag, str(value)]
        raise typer.Exit(run_profiles(profiles, args))

    # Join token: $WK_JOIN_TOKEN, else --token-file.
    token = os.environ.get(WK_JOIN_TOKEN, "").strip() or None
    if token is None:
        try:
            token = read_token_file(token_file)
        except OSError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1)
    if token is None:
        typer.echo("info: no join token set; registering without authentication.", err=True)

    code = run_worker(
        manager_url=manager_url,
        token=token,
        state_dir=state_dir,
        discover_timeout=discover_timeout,
        backend=backend,
        transport=transport,
        gpu_id=gpu_id,
        on_serving=ready_callback(ready_fd) if ready_fd is not None else None,
    )
    raise typer.Exit(code)


def _parse_yes_no(value: str) -> bool:
    """Typer callback: parse a ``YES``/``NO`` option value into a bool (case-insensitive)."""
    normalized = value.strip().upper()
    if normalized in ("YES", "Y"):
        return True
    if normalized in ("NO", "N"):
        return False
    raise typer.BadParameter("expected YES or NO")


def _resolve_manager_runtime(
    config_file: Optional[Path], token_file: Optional[Path], host: Optional[str], port: Optional[int]
) -> dict:
    """Resolve config file, token, provider, and bind host/port for the manager."""
    from oumigo.config import resolve_config_file
    from oumigo.providers import create_provider
    from oumigo.service.manager.auth import resolve_manager_token
    from oumigo.service.manager.settings import get_catalog, get_provider_name, load_manager_yaml

    config_file = resolve_config_file(config_file)
    try:
        token = resolve_manager_token(token_file)
    except OSError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    manager_config = load_manager_yaml(config_file)
    try:
        provider = create_provider(get_provider_name(manager_config))
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    control_plane = manager_config.get("control_plane", {}) or {}
    heartbeat_cfg = manager_config.get("heartbeat", {}) or {}
    bind_host = host or control_plane.get("host", "0.0.0.0")

    # CUSTOM assumes an untrusted WAN: refuse bind layouts that would expose an
    # unauthenticated or plain-HTTP surface (see oumigo.service.manager.posture).
    from oumigo.service.manager.posture import check_posture
    from oumigo.service.manager.settings import (
        get_client_auth,
        get_control_tls,
        get_dashboard,
        get_data_plane,
    )

    try:
        control_tls = get_control_tls(manager_config)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    data_host, _ = get_data_plane(manager_config)
    try:
        get_dashboard(manager_config)  # the web console binds loopback only
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    problems = check_posture(
        provider.name,
        control_host=bind_host,
        control_tls=control_tls is not None,
        data_host=data_host,
        client_auth=get_client_auth(manager_config),
    )
    if problems:
        for problem in problems:
            typer.echo(f"error: {problem}", err=True)
        raise typer.Exit(1)

    # The catalog (profiles + allow/default) is validated here, so a missing `catalog:`
    # block or a typo in a profile stops the manager at once instead of reaching a
    # worker minutes later.
    try:
        get_catalog(manager_config)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    return {
        "config_file": config_file,
        "token": token,
        "provider": provider,
        "bind_host": bind_host,
        "control_tls": control_tls is not None,
        "bind_port": port or int(control_plane.get("port", 7014)),
        "heartbeat_interval": int(heartbeat_cfg.get("interval_s", 10)),
        "heartbeat_timeout": int(heartbeat_cfg.get("timeout_s", 30)),
        "forget_after": int(heartbeat_cfg.get("forget_after_seconds", 3600)),  # 1 hour
    }


def _no_token_message(provider) -> str:
    """Startup notice when no reusable join token is configured, per provider posture."""
    from oumigo.service.manager.auth import NO_TOKEN_WAN_NOTICE, NO_TOKEN_WARNING

    return NO_TOKEN_WAN_NOTICE if provider.is_wan else NO_TOKEN_WARNING


_MANAGER_ENV_OPTION = typer.Option(
    Path(".env"),
    "--env-file",
    help="Load KEY=VALUE vars (e.g. MG_ADMIN_USER / MG_ADMIN_PASSWORD_HASH) from this "
    "file. Existing environment variables win. Skipped if absent; the web console's Create "
    "Admin Account page writes here.",
)


def _load_manager_env(env_file: Path) -> Path:
    """Load the manager's env file; returns its absolute path (it may not exist yet)."""
    from oumigo.common.env import load_env_file
    from oumigo.common.envnames import MANAGER, refuse_retired

    env_file = env_file.expanduser().absolute()
    load_env_file(env_file)
    refuse_retired(MANAGER)
    return env_file


@manager_app.command("admin-password")
def manager_admin_password() -> None:
    """Make the password hash for the web console's admin account.

    Reads the password twice (never from the command line) and prints the
    MG_ADMIN_PASSWORD_HASH line to put into the manager's .env, next to
    MG_ADMIN_USER=<login>. Then restart the manager.
    """
    import getpass

    from oumigo.service.manager.admin_account import (
        ENV_ADMIN_PASSWORD_HASH,
        hash_password,
        password_problems,
    )

    password = getpass.getpass("New admin password: ")
    problems = password_problems(password)
    if problems:
        for problem in problems:
            typer.echo(f"error: {problem}", err=True)
        raise typer.Exit(1)
    if getpass.getpass("Again: ") != password:
        typer.echo("error: the two passwords differ", err=True)
        raise typer.Exit(1)
    typer.echo(f"{ENV_ADMIN_PASSWORD_HASH}={hash_password(password)}")


@manager_app.command("serve")
def manager_serve(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config-file",
        "-c",
        envvar=en.MG_CONFIG_FILE,
        help="Path to the manager config file (overrides the search path).",
    ),
    token_file: Optional[Path] = typer.Option(
        None, "--token-file", help="Path to a file containing the shared bearer token."
    ),
    host: Optional[str] = typer.Option(None, "--host", help="Control-plane bind host (overrides config)."),
    port: Optional[int] = typer.Option(None, "--port", help="Control-plane bind port (overrides config)."),
    no_mdns: bool = typer.Option(False, "--no-mdns", help="Do not advertise the manager over mDNS."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log server output at INFO."),
    env_file: Path = _MANAGER_ENV_OPTION,
) -> None:
    """Run the control-plane server in the foreground (no console).

    This is what systemd should run, and what `manager run` spawns on a TTY.
    """
    from oumigo.service.manager.control.server import run_server

    env_file = _load_manager_env(env_file)
    rt = _resolve_manager_runtime(config_file, token_file, host, port)
    if rt["token"] is None:
        typer.echo(_no_token_message(rt["provider"]), err=True)

    run_server(
        host=rt["bind_host"],
        port=rt["bind_port"],
        token=rt["token"],
        provider_name=rt["provider"].name,
        heartbeat_interval=rt["heartbeat_interval"],
        heartbeat_timeout=rt["heartbeat_timeout"],
        forget_after=rt["forget_after"],
        advertise=not no_mdns and not rt["provider"].is_wan,  # CUSTOM: workers use an explicit URL
        verbose=verbose,
        config_file=rt["config_file"],
        env_file=env_file,
    )


@manager_app.command("run")
def manager_run(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config-file",
        "-c",
        envvar=en.MG_CONFIG_FILE,
        help="Path to the manager config file (overrides the search path).",
    ),
    token_file: Optional[Path] = typer.Option(
        None, "--token-file", help="Path to a file containing the shared bearer token."
    ),
    host: Optional[str] = typer.Option(None, "--host", help="Control-plane bind host (overrides config)."),
    port: Optional[int] = typer.Option(None, "--port", help="Control-plane bind port (overrides config)."),
    no_mdns: str = typer.Option(
        "NO",
        "--no-mdns",
        metavar="[YES|NO]",
        callback=_parse_yes_no,
        help="Do not advertise the manager over mDNS. [YES|NO], default NO.",
    ),
    verbose: str = typer.Option(
        "NO",
        "--verbose",
        "-v",
        metavar="[YES|NO]",
        callback=_parse_yes_no,
        help="Stream server logs to the console. [YES|NO], default NO.",
    ),
    env_file: Path = _MANAGER_ENV_OPTION,
) -> None:
    """Run the manager.

    On a TTY: spawn the control-plane server as a child process and attach an
    interactive console over HTTP (quit stops the child). Headless (no TTY, e.g.
    systemd): run the server directly. Serves the control plane and the data-plane
    router (client inference calls forwarded to healthy worker vLLMs) together.
    """
    env_file = _load_manager_env(env_file)
    rt = _resolve_manager_runtime(config_file, token_file, host, port)

    if not sys.stdin.isatty():
        # Headless: run the server directly in this process.
        from oumigo.service.manager.control.server import run_server

        if rt["token"] is None:
            typer.echo(_no_token_message(rt["provider"]), err=True)
        run_server(
            host=rt["bind_host"],
            port=rt["bind_port"],
            token=rt["token"],
            provider_name=rt["provider"].name,
            heartbeat_interval=rt["heartbeat_interval"],
            heartbeat_timeout=rt["heartbeat_timeout"],
            forget_after=rt["forget_after"],
            advertise=not no_mdns and not rt["provider"].is_wan,
            verbose=verbose,
            config_file=rt["config_file"],
            env_file=env_file,
        )
        return

    # TTY: spawn the server as a child and attach the console over HTTP.
    from oumigo.service.manager.launcher import run_with_child_server

    forward = ["--host", rt["bind_host"], "--port", str(rt["bind_port"]),
               "--env-file", str(env_file)]
    if rt["config_file"] is not None:
        forward += ["--config-file", str(rt["config_file"])]
    if no_mdns:
        forward.append("--no-mdns")
    if verbose:
        forward.append("--verbose")

    run_with_child_server(
        host=rt["bind_host"],
        port=rt["bind_port"],
        token=rt["token"],
        forward_args=forward,
        verbose=verbose,
        tls=rt["control_tls"],
    )


def _auth_store(config_file: Optional[Path]):
    """Open the manager's auth state (the same file the running manager reads)."""
    from oumigo.common.state import default_state_dir
    from oumigo.config import resolve_config_file
    from oumigo.service.manager.credentials import AuthStore
    from oumigo.service.manager.settings import get_auth_state_dir, load_manager_yaml

    config = load_manager_yaml(resolve_config_file(config_file))
    return AuthStore.at(get_auth_state_dir(config) or default_state_dir())


def _parse_duration(text: str) -> float | None:
    """'90s' / '30m' / '1h' / '2d' -> seconds; '0' / 'none' / 'never' -> None (no expiry)."""
    value = text.strip().lower()
    if value in ("0", "none", "never", ""):
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        if value[-1] in units:
            return float(value[:-1]) * units[value[-1]]
        return float(value)
    except ValueError:
        raise typer.BadParameter(f"not a duration: {text!r} (e.g. 30m, 1h, 2d, never)") from None


_CONFIG_OPTION = typer.Option(
    None, "--config-file", "-c", envvar=en.MG_CONFIG_FILE,
    help="Manager config file (locates auth.state_dir).",
)


@token_app.command("create")
def token_create(
    ttl: str = typer.Option("1h", "--ttl", help="Lifetime: 30m, 1h, 2d, or 'never'."),
    uses: int = typer.Option(1, "--uses", help="Registrations allowed; 0 = unlimited."),
    label: Optional[str] = typer.Option(None, "--label", help="Free-form note (e.g. the pod name)."),
    config_file: Optional[Path] = _CONFIG_OPTION,
) -> None:
    """Mint a join token for a new worker (set it as WK_JOIN_TOKEN on the worker)."""
    token_id, token = _auth_store(config_file).create_join_token(
        ttl_s=_parse_duration(ttl), uses=uses or None, label=label
    )
    typer.echo(f"id: {token_id}", err=True)
    typer.echo(token)  # stdout carries only the token, so it can be captured by scripts


@token_app.command("list")
def token_list(config_file: Optional[Path] = _CONFIG_OPTION) -> None:
    """List minted join tokens (the tokens themselves are not stored)."""
    tokens = _auth_store(config_file).list_join_tokens()
    if not tokens:
        typer.echo("no join tokens minted.")
        return
    typer.echo(f"{'ID':<10}{'STATUS':<10}{'USES LEFT':<11}{'EXPIRES (UTC)':<21}LABEL")
    for t in tokens:
        status = "revoked" if t.revoked else ("usable" if t.usable else "spent")
        expires = (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t.expires_at))
                   if t.expires_at else "never")
        uses_left = "unlimited" if t.uses_left is None else str(t.uses_left)
        typer.echo(f"{t.id:<10}{status:<10}{uses_left:<11}{expires:<21}{t.label or ''}")


@token_app.command("revoke")
def token_revoke(token_id: str, config_file: Optional[Path] = _CONFIG_OPTION) -> None:
    """Revoke an unused join token. Workers that already joined keep their node tokens."""
    if not _auth_store(config_file).revoke_join_token(token_id):
        typer.echo(f"error: no join token with id {token_id}", err=True)
        raise typer.Exit(1)
    typer.echo(f"revoked join token {token_id}")


@node_app.command("revoke")
def node_revoke(node_id: str, config_file: Optional[Path] = _CONFIG_OPTION) -> None:
    """Revoke a worker: its node token stops working and the manager drops it and closes
    its tunnels within seconds. It cannot re-register under this node_id."""
    _auth_store(config_file).revoke_node(node_id)
    typer.echo(f"revoked node {node_id}")


@node_app.command("close")
def node_close(
    node: str = typer.Argument(..., help="The worker's node id (HASH ID) or name (Worker#3)."),
    config_file: Optional[Path] = _CONFIG_OPTION,
) -> None:
    """Close a worker: it stops taking new requests, finishes the ones in progress, then
    shuts down. The running manager acts on it within seconds."""
    node_id = _resolve_node(node, config_file)
    _auth_store(config_file).request_close(node_id, "cli")
    typer.echo(f"close requested for node {node_id}; the manager drains and stops it")


def _resolve_node(node: str, config_file: Optional[Path]) -> str:
    """The node id for `node` (a node id or a Worker#N name), checked against the running
    manager's workers. If the manager can't be reached, only a node id is accepted."""
    import re

    import httpx

    from oumigo.common.http import local_control_url, loopback_verify
    from oumigo.config import resolve_config_file
    from oumigo.service.manager.settings import get_control_tls, load_manager_yaml

    config = load_manager_yaml(resolve_config_file(config_file))
    port = int((config.get("control_plane") or {}).get("port", 7014))
    url = local_control_url(port, get_control_tls(config) is not None)
    try:
        resp = httpx.get(f"{url}/workers", timeout=5.0, verify=loopback_verify(url))
        resp.raise_for_status()
        workers = resp.json().get("workers", [])
    except (httpx.HTTPError, ValueError) as exc:
        if re.fullmatch(r"[0-9a-f]{16}", node):
            typer.echo(f"warning: manager at {url} not reachable ({exc}); requesting anyway",
                       err=True)
            return node
        typer.echo(f"error: manager at {url} not reachable ({exc}); give the node id "
                   "instead of a name", err=True)
        raise typer.Exit(1)
    for row in workers:
        if node == row.get("node_id") or node.lower() == str(row.get("name", "")).lower():
            return row["node_id"]
    typer.echo(f"error: no registered worker {node!r}", err=True)
    raise typer.Exit(1)


@node_app.command("unrevoke")
def node_unrevoke(node_id: str, config_file: Optional[Path] = _CONFIG_OPTION) -> None:
    """Lift a node revocation: its node token (derived, not stored) is accepted again."""
    if not _auth_store(config_file).unrevoke_node(node_id):
        typer.echo(f"error: node {node_id} is not revoked", err=True)
        raise typer.Exit(1)
    typer.echo(f"node {node_id} may register again")


@apikey_app.command("create")
def apikey_create(
    label: Optional[str] = typer.Option(None, "--label", help="Who holds it (e.g. kari-dev)."),
    config_file: Optional[Path] = _CONFIG_OPTION,
) -> None:
    """Create a client API key (clients send it as `Authorization: Bearer <key>`)."""
    key_id, key = _auth_store(config_file).create_api_key(label=label)
    typer.echo(f"id: {key_id}", err=True)
    typer.echo(key)  # stdout carries only the key, so it can be captured by scripts


@apikey_app.command("list")
def apikey_list(config_file: Optional[Path] = _CONFIG_OPTION) -> None:
    """List client API keys (the keys themselves are not stored)."""
    keys = _auth_store(config_file).list_api_keys()
    if not keys:
        typer.echo("no client API keys.")
        return
    typer.echo(f"{'ID':<10}{'STATUS':<10}{'CREATED (UTC)':<21}LABEL")
    for k in keys:
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(k.created_at))
        typer.echo(f"{k.id:<10}{'revoked' if k.revoked else 'active':<10}{created:<21}{k.label or ''}")


@apikey_app.command("revoke")
def apikey_revoke(key_id: str, config_file: Optional[Path] = _CONFIG_OPTION) -> None:
    """Revoke a client API key. The running manager refuses it from the next request on."""
    if not _auth_store(config_file).revoke_api_key(key_id):
        typer.echo(f"error: no client API key with id {key_id}", err=True)
        raise typer.Exit(1)
    typer.echo(f"revoked client API key {key_id}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
