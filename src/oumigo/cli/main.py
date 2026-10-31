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

app = typer.Typer(help="OumiGo — run and manage vLLM replica fleets.", no_args_is_help=True)

worker_app = typer.Typer(help="Worker-node coordinator (L1).", no_args_is_help=True)
manager_app = typer.Typer(help="Manager node: control plane + router (L3).", no_args_is_help=True)
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
        envvar="OUMIGO_MANAGER_URL",
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
        envvar="OUMIGO_TRANSPORT",
        help="How the router reaches this worker: 'direct' (router dials this host; LAN) or "
        "'tunnel' (this worker dials out to the manager; no inbound port needed). "
        "Default: direct.",
    ),
    discover_timeout: Optional[float] = typer.Option(
        None,
        "--discover-timeout",
        envvar="OUMIGO_DISCOVER_TIMEOUT",
        help="Seconds to wait for mDNS discovery of the manager (when no --manager-url). "
        "Default: 60.",
    ),
    env_file: Optional[Path] = typer.Option(
        Path(".env"),
        "--env-file",
        help="Load KEY=VALUE vars from this file into the environment (inherited by the "
        "backend). Existing environment variables win. Skipped if the file is absent.",
    ),
    backend: str = typer.Option(
        "vllm",
        "--backend",
        help="Inference backend to supervise: 'vllm' (default) or 'transformer' "
        "(HF-transformers server; max_concurrent_requests forced to 1).",
    ),
) -> None:
    """Load the env file, resolve identity, find the manager, register, and supervise the backend.

    The env file is applied before anything reads the environment, so the negotiable
    vars (MODEL_NAME / MAX_MODEL_LEN / MODEL_STORAGE_LOCATION / HF_HOME / HF_TOKEN, plus
    VLLM_*) reach the backend child process the coordinator spawns.
    """
    import logging

    from oumigo.common.env import load_env_file
    from oumigo.service.manager.auth import resolve_manager_token  # shared bearer token
    from oumigo.service.worker.coordinator import BACKENDS, run_worker

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if backend not in BACKENDS:
        typer.echo(f"error: unknown --backend {backend!r}; choose {' or '.join(BACKENDS)}.", err=True)
        raise typer.Exit(2)

    if env_file is not None:
        load_env_file(env_file)
    # typer resolves `envvar=` while parsing, before the env file is loaded: fill the
    # options a flag or the real environment left unset from the env file's values.
    manager_url = manager_url or os.environ.get("OUMIGO_MANAGER_URL", "").strip() or None
    transport = transport or os.environ.get("OUMIGO_TRANSPORT", "").strip() or "direct"
    if discover_timeout is None:
        try:
            discover_timeout = float(os.environ.get("OUMIGO_DISCOVER_TIMEOUT") or 60.0)
        except ValueError:
            typer.echo("error: OUMIGO_DISCOVER_TIMEOUT must be a number of seconds.", err=True)
            raise typer.Exit(2)

    if transport not in ("direct", "tunnel"):
        typer.echo(f"error: unknown --transport {transport!r}; choose direct or tunnel.", err=True)
        raise typer.Exit(2)

    # Join token: $OUMIGO_JOIN_TOKEN, else the v1 names ($OUMIGO_MANAGER_TOKEN / --token-file).
    token = os.environ.get("OUMIGO_JOIN_TOKEN", "").strip() or None
    if token is None:
        try:
            token = resolve_manager_token(token_file)
        except OSError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1)
    if token is None:
        typer.echo("info: no join token set; registering without authentication.", err=True)

    run_worker(
        manager_url=manager_url,
        token=token,
        state_dir=state_dir,
        discover_timeout=discover_timeout,
        backend=backend,
        transport=transport,
    )


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
    from oumigo.service.manager.auth import resolve_manager_token
    from oumigo.service.manager.settings import build_node_spec, get_provider_name, load_manager_yaml
    from oumigo.providers import create_provider

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
    dash_enabled, dash_host, _ = get_dashboard(manager_config)
    problems = check_posture(
        provider.name,
        control_host=bind_host,
        control_tls=control_tls is not None,
        data_host=data_host,
        client_auth=get_client_auth(manager_config),
        dashboard_enabled=dash_enabled,
        dashboard_host=dash_host,
    )
    if problems:
        for problem in problems:
            typer.echo(f"error: {problem}", err=True)
        raise typer.Exit(1)

    # The catalog (profiles + allow/default) is validated here, so a typo in a profile
    # stops the manager at once instead of reaching a worker minutes later.
    try:
        node_spec = build_node_spec(manager_config)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    return {
        "config_file": config_file,
        "token": token,
        "provider": provider,
        "node_spec": node_spec,
        "bind_host": bind_host,
        "control_tls": control_tls is not None,
        "bind_port": port or int(control_plane.get("port", 7014)),
        "heartbeat_interval": int(heartbeat_cfg.get("interval_s", 10)),
        "heartbeat_timeout": int(heartbeat_cfg.get("timeout_s", 30)),
        "forget_after": int(heartbeat_cfg.get("forget_after_seconds", 3600)),  # 1 hour
    }


def _no_token_message(provider) -> str:
    """Startup notice when no reusable join token is configured, per provider posture."""
    from oumigo.service.manager.auth import NO_TOKEN_WARNING, NO_TOKEN_WAN_NOTICE

    return NO_TOKEN_WAN_NOTICE if provider.is_wan else NO_TOKEN_WARNING


@manager_app.command("serve")
def manager_serve(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config-file",
        "-c",
        envvar="OUMIGO_CONFIG_FILE",
        help="Path to the manager config file (overrides the search path).",
    ),
    token_file: Optional[Path] = typer.Option(
        None, "--token-file", help="Path to a file containing the shared bearer token."
    ),
    host: Optional[str] = typer.Option(None, "--host", help="Control-plane bind host (overrides config)."),
    port: Optional[int] = typer.Option(None, "--port", help="Control-plane bind port (overrides config)."),
    no_mdns: bool = typer.Option(False, "--no-mdns", help="Do not advertise the manager over mDNS."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log server output at INFO."),
) -> None:
    """Run the control-plane server in the foreground (no console).

    This is what systemd should run, and what `manager run` spawns on a TTY.
    """
    from oumigo.service.manager.control.server import run_server

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
        node_spec=rt["node_spec"],
    )


@manager_app.command("run")
def manager_run(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config-file",
        "-c",
        envvar="OUMIGO_CONFIG_FILE",
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
) -> None:
    """Run the manager.

    On a TTY: spawn the control-plane server as a child process and attach an
    interactive console over HTTP (quit stops the child). Headless (no TTY, e.g.
    systemd): run the server directly. Serves the control plane and the data-plane
    router (client inference calls forwarded to healthy worker vLLMs) together.
    """
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
            node_spec=rt["node_spec"],
        )
        return

    # TTY: spawn the server as a child and attach the console over HTTP.
    from oumigo.service.manager.launcher import run_with_child_server

    forward = ["--host", rt["bind_host"], "--port", str(rt["bind_port"])]
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
    None, "--config-file", "-c", envvar="OUMIGO_CONFIG_FILE",
    help="Manager config file (locates auth.state_dir).",
)


@token_app.command("create")
def token_create(
    ttl: str = typer.Option("1h", "--ttl", help="Lifetime: 30m, 1h, 2d, or 'never'."),
    uses: int = typer.Option(1, "--uses", help="Registrations allowed; 0 = unlimited."),
    label: Optional[str] = typer.Option(None, "--label", help="Free-form note (e.g. the pod name)."),
    config_file: Optional[Path] = _CONFIG_OPTION,
) -> None:
    """Mint a join token for a new worker (set it as OUMIGO_JOIN_TOKEN on the worker)."""
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
