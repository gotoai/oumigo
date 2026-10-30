"""Provider selection: map a configured provider name to a Provider instance."""

from __future__ import annotations

from oumigo.providers.base import Provider
from oumigo.providers.static import CUSTOM, LAN, StaticProvider

KNOWN_PROVIDERS = (LAN, CUSTOM)


def create_provider(name: str) -> Provider:
    """Instantiate the Provider selected by ``name`` (case-insensitive).

    ``LAN`` and ``CUSTOM`` are both the non-provisioning StaticProvider; CUSTOM covers
    workers on any vendor or site with no vendor-specific processing. Cloud backends
    that provision through a vendor API (e.g. ConoHa) will register here later.
    """
    key = name.strip().upper()
    if key in KNOWN_PROVIDERS:
        return StaticProvider(key)
    raise ValueError(f"unknown provider: {name!r} (known: {', '.join(KNOWN_PROVIDERS)})")
