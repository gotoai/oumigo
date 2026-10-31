"""Network posture checks — what the configured provider allows the manager to expose.

`LAN` trusts the local network: anything may bind any interface, and worker auth may
be off (the v1 dev convenience). `CUSTOM` assumes workers on an untrusted wide-area
network, and the manager cannot see whether a TLS reverse proxy (nginx) sits in front
of it, so it insists on a shape that is safe either way:

- the **control plane** binds loopback (nginx terminates TLS and forwards to it), or
  serves TLS itself (`control_plane.tls_cert_file` / `tls_key_file`);
- the **router** binds loopback (it serves plain HTTP; nginx terminates TLS for remote
  clients) and requires client API keys (`data_plane.client_auth`), because a loopback
  bind does not prove that no proxy exposes it;
- the **dashboard** binds loopback: it has no auth, so view it through an SSH tunnel;
- worker auth is never disabled (see `oumigo.service.manager.auth`).

`check_posture` returns human-readable violations; the CLI refuses to start on any.
"""

from __future__ import annotations

import ipaddress

from oumigo.providers.static import CUSTOM


def is_loopback(host: str) -> bool:
    """True for a loopback bind address (127.0.0.0/8, ::1, ``localhost``)."""
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False  # a hostname or wildcard we can't prove is loopback


def check_posture(
    provider: str,
    *,
    control_host: str,
    control_tls: bool,
    data_host: str,
    client_auth: bool,
    dashboard_enabled: bool,
    dashboard_host: str,
) -> list[str]:
    """Return the posture violations for this bind layout (empty = OK)."""
    if provider != CUSTOM:
        return []
    problems: list[str] = []
    if not is_loopback(control_host) and not control_tls:
        problems.append(
            f"control_plane.host is {control_host!r}: under provider CUSTOM the control plane "
            "must bind 127.0.0.1 (behind a TLS reverse proxy such as nginx) or set "
            "control_plane.tls_cert_file / tls_key_file"
        )
    if not is_loopback(data_host):
        problems.append(
            f"data_plane.host is {data_host!r}: under provider CUSTOM the router must bind "
            "127.0.0.1 — it serves plain HTTP, so expose it through a TLS reverse proxy "
            "such as nginx"
        )
    if not client_auth:
        problems.append(
            "data_plane.client_auth is off: under provider CUSTOM the router always requires "
            "client API keys (create one with `oumigo manager apikey create`)"
        )
    if dashboard_enabled and not is_loopback(dashboard_host):
        problems.append(
            f"dashboard.host is {dashboard_host!r}: under provider CUSTOM the dashboard must "
            "bind 127.0.0.1 (view it through an SSH tunnel) or be disabled"
        )
    return problems
