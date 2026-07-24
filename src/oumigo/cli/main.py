"""Console entrypoint: ``oumigo <subcommand>``.

One CLI with subcommands grouped by layer:
  * ``oumigo worker ...``  — worker-node coordinator (L1)
  * ``oumigo manager ...`` — manager: control plane + router (L3)
  * ``oumigo cluster ...`` — cluster-level operations
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from oumigo import __version__

app = typer.Typer(help="oumigo — run and manage vLLM replica fleets.", no_args_is_help=True)

worker_app = typer.Typer(help="Worker-node coordinator (L1).", no_args_is_help=True)
manager_app = typer.Typer(help="Manager node: control plane + router (L3).", no_args_is_help=True)
cluster_app = typer.Typer(help="Cluster-level operations.", no_args_is_help=True)

app.add_typer(worker_app, name="worker")
app.add_typer(manager_app, name="manager")
app.add_typer(cluster_app, name="cluster")


@app.command("version")
def version() -> None:
    """Print the oumigo version."""
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
        None, "--token-file", help="Path to a file containing the shared bearer token."
    ),
    state_dir: Optional[Path] = typer.Option(
        None, "--state-dir", help="Directory holding the persisted node identity."
    ),
    discover_timeout: float = typer.Option(
        60.0,
        "--discover-timeout",
        envvar="OUMIGO_DISCOVER_TIMEOUT",
        help="Seconds to wait for mDNS discovery of the manager (when no --manager-url).",
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
    vars (MODEL_NAME / MAX_MODEL_LEN / HF_HOME / HF_TOKEN, plus VLLM_*) reach the
    backend child process the coordinator spawns.
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

    try:
        token = resolve_manager_token(token_file)
    except OSError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    if token is None:
        typer.echo("info: no token set; registering without authentication.", err=True)

    run_worker(
        manager_url=manager_url,
        token=token,
        state_dir=state_dir,
        discover_timeout=discover_timeout,
        backend=backend,
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
    return {
        "config_file": config_file,
        "token": token,
        "provider": provider,
        "node_spec": build_node_spec(manager_config),
        "bind_host": host or control_plane.get("host", "0.0.0.0"),
        "bind_port": port or int(control_plane.get("port", 7014)),
        "heartbeat_interval": int(heartbeat_cfg.get("interval_s", 10)),
        "heartbeat_timeout": int(heartbeat_cfg.get("timeout_s", 30)),
        "forget_after": int(heartbeat_cfg.get("forget_after_seconds", 3600)),  # 1 hour
    }


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
    from oumigo.service.manager.auth import NO_TOKEN_WARNING
    from oumigo.service.manager.control.server import run_server

    rt = _resolve_manager_runtime(config_file, token_file, host, port)
    if rt["token"] is None:
        typer.echo(NO_TOKEN_WARNING, err=True)

    run_server(
        host=rt["bind_host"],
        port=rt["bind_port"],
        token=rt["token"],
        provider_name=rt["provider"].name,
        heartbeat_interval=rt["heartbeat_interval"],
        heartbeat_timeout=rt["heartbeat_timeout"],
        forget_after=rt["forget_after"],
        advertise=not no_mdns,
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
    from oumigo.service.manager.auth import NO_TOKEN_WARNING

    rt = _resolve_manager_runtime(config_file, token_file, host, port)

    if not sys.stdin.isatty():
        # Headless: run the server directly in this process.
        from oumigo.service.manager.control.server import run_server

        if rt["token"] is None:
            typer.echo(NO_TOKEN_WARNING, err=True)
        run_server(
            host=rt["bind_host"],
            port=rt["bind_port"],
            token=rt["token"],
            provider_name=rt["provider"].name,
            heartbeat_interval=rt["heartbeat_interval"],
            heartbeat_timeout=rt["heartbeat_timeout"],
            forget_after=rt["forget_after"],
            advertise=not no_mdns,
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
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
