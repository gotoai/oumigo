"""LAN discovery via mDNS / DNS-SD (zeroconf).

Shared by both roles: the manager *advertises* a service, the worker *browses*
for it. This is the LAN auto-fill for the manager URL — explicit URL/env still
takes precedence, and cloud environments use provisioning-time injection instead.
"""

from __future__ import annotations

import logging
import socket
import threading

from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf

log = logging.getLogger("oumigo.discovery")

SERVICE_TYPE = "_oumigo._tcp.local."
SERVICE_NAME = "oumigo-manager._oumigo._tcp.local."

DEFAULT_DISCOVER_TIMEOUT = 60.0  # seconds to browse before giving up
_RESOLVE_TIMEOUT_MS = 3000       # per-service info resolution, once one is seen


def get_lan_ip() -> str:
    """Best-effort LAN IP of this host (the outbound-interface address)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packets are actually sent; this just selects the outbound interface.
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def build_service_info(port: int, ip: str | None = None) -> ServiceInfo:
    """Build the mDNS ServiceInfo advertised for the manager control plane."""
    ip = ip or get_lan_ip()
    return ServiceInfo(
        SERVICE_TYPE,
        SERVICE_NAME,
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={"role": "manager"},
    )


def advertise_manager(port: int, ip: str | None = None) -> tuple[Zeroconf, ServiceInfo]:
    """Advertise the manager control plane on the LAN (blocking/sync). Returns (zc, info).

    Caller must later `zc.unregister_service(info); zc.close()` to stop advertising.
    The server uses the async path instead; this remains for sync callers/tests.
    """
    info = build_service_info(port, ip)
    zc = Zeroconf()
    zc.register_service(info)
    return zc, info


def discover_manager(timeout: float = DEFAULT_DISCOVER_TIMEOUT) -> str | None:
    """Browse the LAN for a manager and return its control-plane URL, or None.

    Waits up to `timeout` seconds for a manager to appear, returning as soon as one
    is found (so a manager that starts later is still picked up within the window).
    """
    zc = Zeroconf()
    found: dict[str, str] = {}
    event = threading.Event()

    class _Listener:
        def add_service(self, zeroconf: Zeroconf, type_: str, name: str) -> None:
            info = zeroconf.get_service_info(type_, name, timeout=_RESOLVE_TIMEOUT_MS)
            if info and info.addresses:
                ip = socket.inet_ntoa(info.addresses[0])
                found["url"] = f"http://{ip}:{info.port}"
                event.set()

        def update_service(self, *args: object) -> None:
            pass

        def remove_service(self, *args: object) -> None:
            pass

    ServiceBrowser(zc, SERVICE_TYPE, _Listener())
    try:
        event.wait(timeout)
        return found.get("url")
    finally:
        zc.close()
