"""Shared HTTP client helpers.

Thin wrappers over httpx so call sites stay uniform.
"""

from __future__ import annotations

from urllib.parse import urlparse

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def local_control_url(port: int, tls: bool) -> str:
    """URL for this host's own control plane (console, launcher, dashboard)."""
    return f"{'https' if tls else 'http'}://127.0.0.1:{port}"


def loopback_verify(url: str) -> bool:
    """httpx `verify` for a URL: False only for https to *this host's own* manager.

    A manager serving built-in TLS presents a certificate for its public name, which
    can never match 127.0.0.1; its console/launcher/dashboard talk to it over loopback,
    where there is no network to protect. Every other URL verifies normally — in
    particular, workers never disable verification.
    """
    parsed = urlparse(url)
    return not (parsed.scheme == "https" and parsed.hostname in LOOPBACK_HOSTS)
