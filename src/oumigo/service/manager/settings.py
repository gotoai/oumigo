"""Minimal manager-config loading (seed of the future ManagerSettings).

Reads the single manager config file (``manager.yaml`` by convention) as a plain
dict. This is a stepping stone: it will be replaced by a validated pydantic
``ManagerSettings`` (with pydantic-settings for env / .env). For now it exposes
only the few fields the manager currently uses.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from oumigo.config.catalog import FleetCatalog, Profile
from oumigo.config.spec import NodeSpec

DEFAULT_PROVIDER = "LAN"
DEFAULT_DATA_PLANE_HOST = "0.0.0.0"  # noqa: S104 - bind all interfaces so LAN clients reach the router
DEFAULT_DATA_PLANE_PORT = 7012
DEFAULT_VLLM_PORT = 7001

DEFAULT_DASHBOARD_HOST = "0.0.0.0"  # noqa: S104 - bind all interfaces so the LAN can view it
DEFAULT_DASHBOARD_PORT = 7080  # 7070 is commonly taken (AnyDesk); reporting plane uses 7080

DEFAULT_TRUSTED_PROXIES = ["127.0.0.1"]  # whose X-Forwarded-For is believed (nginx on the host)
DEFAULT_TUNNEL_POOL_SPARE = 1  # tunnel connections kept beyond a worker's capacity


def load_manager_yaml(config_file: Path | None) -> dict:
    """Load the manager config file, or ``{}`` if absent."""
    if config_file is None:
        return {}
    path = Path(config_file)
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_provider_name(config: dict) -> str:
    """The configured provider name, defaulting to LAN."""
    return str(config.get("provider", DEFAULT_PROVIDER))


def get_data_plane(config: dict) -> tuple[str, int]:
    """The data-plane (router) bind host/port from the `data_plane` block."""
    dp = config.get("data_plane") or {}
    return (
        str(dp.get("host", DEFAULT_DATA_PLANE_HOST)),
        int(dp.get("port", DEFAULT_DATA_PLANE_PORT)),
    )


def get_client_auth(config: dict) -> bool:
    """Whether the router requires a client API key (`data_plane.client_auth`).

    Default: on under provider CUSTOM, off under LAN. CUSTOM cannot turn it off — the
    manager can't see whether a proxy exposes the router (see `posture.check_posture`).
    """
    dp = config.get("data_plane") or {}
    value = dp.get("client_auth")
    if value is None:
        return get_provider_name(config).upper() == "CUSTOM"
    return bool(value)


def get_control_tls(config: dict) -> tuple[str, str] | None:
    """Built-in TLS for the control plane: (cert_file, key_file), or None (plain HTTP).

    Optional — the normal deployment terminates TLS in nginx and binds the control
    plane to loopback. Both files must be given together.
    """
    cp = config.get("control_plane") or {}
    cert, key = cp.get("tls_cert_file"), cp.get("tls_key_file")
    if not cert and not key:
        return None
    if not (cert and key):
        raise ValueError("control_plane.tls_cert_file and tls_key_file must be set together")
    return str(Path(cert).expanduser()), str(Path(key).expanduser())


def get_trusted_proxies(config: dict) -> list[str]:
    """Proxy addresses whose X-Forwarded-For the control plane believes (default: loopback)."""
    cp = config.get("control_plane") or {}
    proxies = cp.get("trusted_proxies")
    if proxies is None:
        return list(DEFAULT_TRUSTED_PROXIES)
    return [str(p) for p in proxies]


def get_tunnel(config: dict) -> tuple[bool, int]:
    """(enabled, pool_spare) from the `tunnel` block. On by default under every provider —
    each worker opts in with `OUMIGO_TRANSPORT=tunnel`; `enabled: false` refuses them."""
    tb = config.get("tunnel") or {}
    return bool(tb.get("enabled", True)), max(0, int(tb.get("pool_spare", DEFAULT_TUNNEL_POOL_SPARE)))


def get_worker_allowlist(config: dict) -> list[str] | None:
    """Worker source-IP allowlist (addresses / CIDR ranges), or None when switched off."""
    wl = config.get("worker_allowlist") or {}
    if not wl.get("enabled", False):
        return None
    return [str(a) for a in (wl.get("addresses") or [])]


def get_auth_state_dir(config: dict) -> Path | None:
    """Where the manager keeps its auth state (join tokens, signing secret, revocations).
    None means the default state directory."""
    auth = config.get("auth") or {}
    path = auth.get("state_dir")
    return Path(path).expanduser() if path else None


def get_agent_defaults(config: dict, profile: Profile | None = None) -> dict[str, float | None]:
    """Client-side agent defaults, served to clients: the profile's `agent:` if it has
    one, else the manager's `agent:` block. `profile` defaults to the fleet's default.

    These bind the *inference client* (`oumigo.api.agent`), not the manager or the
    workers — but they are fleet policy ("how long may one turn take against this
    model?"), so the fleet declares them and the router hands them out at
    `GET /agent-defaults`. A client that passes its own values explicitly ignores
    these; an unreachable manager just means no timeout, as before.

    Both are seconds; absent or `null` means no limit.
    """
    if profile is None:
        profile = get_catalog(config).default_profile
    if profile is not None and profile.agent is not None:
        return profile.agent.model_dump()
    return get_fleet_agent_defaults(config)


def get_fleet_agent_defaults(config: dict) -> dict[str, float | None]:
    """The manager's own `agent:` block: the fallback for profiles without `agent:`."""
    block = config.get("agent") or {}
    return {
        "turn_timeout": _opt_seconds(block.get("turn_timeout"), "agent.turn_timeout"),
        "stall_timeout": _opt_seconds(block.get("stall_timeout"), "agent.stall_timeout"),
    }


def _opt_seconds(value: object, field: str) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number of seconds, got {value!r}") from None
    if seconds <= 0:
        raise ValueError(f"{field} must be positive, got {seconds}")
    return seconds


def get_dashboard(config: dict) -> tuple[bool, str, int]:
    """The reporting-plane dashboard's enabled flag + bind host/port from `dashboard`.

    Bundled with the manager and on by default; `dashboard.enabled: false` is an
    escape hatch (e.g. a port conflict). Host/port default to 0.0.0.0:7080.
    """
    db = config.get("dashboard") or {}
    return (
        bool(db.get("enabled", True)),
        str(db.get("host", DEFAULT_DASHBOARD_HOST)),
        int(db.get("port", DEFAULT_DASHBOARD_PORT)),
    )


def get_catalog(config: dict) -> FleetCatalog:
    """The profiles this fleet allows and its default (`catalog:`, or a legacy `model:`).

    Raises `CatalogError` (a ValueError) for an invalid profile or `catalog:` block.
    """
    return FleetCatalog.from_config(config)


def get_vllm_port(config: dict) -> int:
    """The backend port workers prefer: `worker.vllm_port`, else a legacy `model.port`."""
    worker = config.get("worker") or {}
    if worker.get("vllm_port") is not None:
        return int(worker["vllm_port"])
    return int((config.get("model") or {}).get("port", DEFAULT_VLLM_PORT))


def build_node_spec(config: dict) -> NodeSpec | None:
    """The NodeSpec of the fleet's default profile, handed to workers that choose none.

    Returns None if the fleet has no default (no `catalog.default` and no legacy
    `model:` block): registration then returns no spec. Local-only concerns (cache
    dirs) are excluded; those come from each worker's own environment.
    """
    profile = get_catalog(config).default_profile
    return profile.node_spec(get_vllm_port(config)) if profile else None
