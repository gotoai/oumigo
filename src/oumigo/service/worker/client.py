"""HTTP client for the worker -> manager control-plane calls.

TLS: an `https://` manager is verified against the system CA store, or against
`$OUMIGO_MANAGER_CA_FILE` (a PEM file holding the manager's self-signed certificate or
private CA) — see `configure_tls`. Verification is never disabled.
"""

from __future__ import annotations

import os
import ssl

import httpx

from oumigo.config.spec import NodeSpec
from oumigo.protocol.messages import (
    HeartbeatRequest,
    HeartbeatResponse,
    MetricsReport,
    RegisterRequest,
    RegisterResponse,
)


ENV_CA_FILE = "OUMIGO_MANAGER_CA_FILE"

_verify: ssl.SSLContext | bool = True  # set by configure_tls; shared by every call here


def configure_tls(ca_file: str | None = None) -> ssl.SSLContext:
    """Build (and adopt) the SSL context for talking to the manager.

    ``ca_file`` (default `$OUMIGO_MANAGER_CA_FILE`) adds a trusted PEM — the manager's
    self-signed certificate or private CA. Without it the system CA store is used, as
    for any public (e.g. Let's Encrypt) certificate behind nginx.
    """
    global _verify
    ca_file = ca_file or os.environ.get(ENV_CA_FILE) or None
    context = ssl.create_default_context(cafile=ca_file)
    _verify = context
    return context


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_spec(
    manager_url: str, token: str | None = None, timeout: float = 10.0
) -> tuple[NodeSpec | None, str | None]:
    """GET the fleet's vLLM spec (None when the manager has no model configured yet) and
    the manager's provider (`LAN` / `CUSTOM`; None from a pre-0.4 manager), so the worker
    can preflight its port and enforce the provider's posture before it registers."""
    resp = httpx.get(
        f"{manager_url.rstrip('/')}/spec",
        headers=_headers(token),
        timeout=timeout,
        verify=_verify,
    )
    resp.raise_for_status()
    body = resp.json()
    data = body.get("node_spec")
    spec = NodeSpec.model_validate(data) if data is not None else None
    return spec, body.get("provider")


def fetch_node_spec(
    manager_url: str, token: str | None = None, timeout: float = 10.0
) -> NodeSpec | None:
    """The fleet's vLLM spec only (see `fetch_spec`)."""
    return fetch_spec(manager_url, token, timeout)[0]


def register(
    manager_url: str, req: RegisterRequest, token: str | None = None, timeout: float = 10.0
) -> RegisterResponse:
    resp = httpx.post(
        f"{manager_url.rstrip('/')}/register",
        json=req.model_dump(mode="json"),
        headers=_headers(token),
        timeout=timeout,
        verify=_verify,
    )
    resp.raise_for_status()
    return RegisterResponse.model_validate(resp.json())


def heartbeat(
    manager_url: str, req: HeartbeatRequest, token: str | None = None, timeout: float = 10.0
) -> HeartbeatResponse:
    resp = httpx.post(
        f"{manager_url.rstrip('/')}/heartbeat",
        json=req.model_dump(mode="json"),
        headers=_headers(token),
        timeout=timeout,
        verify=_verify,
    )
    resp.raise_for_status()
    return HeartbeatResponse.model_validate(resp.json())


def send_metrics(
    manager_url: str, report: MetricsReport, token: str | None = None, timeout: float = 10.0
) -> None:
    """POST a buffered metrics batch. Raises on non-2xx so the caller can re-buffer."""
    resp = httpx.post(
        f"{manager_url.rstrip('/')}/metrics",
        json=report.model_dump(mode="json"),
        headers=_headers(token),
        timeout=timeout,
        verify=_verify,
    )
    resp.raise_for_status()
