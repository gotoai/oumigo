"""Provider selection: map a configured provider name to a Provider instance."""

from __future__ import annotations

from oumigo.providers.base import Provider
from oumigo.providers.static import StaticProvider

KNOWN_PROVIDERS = ("LAN",)


def create_provider(name: str) -> Provider:
    """Instantiate the Provider selected by ``name`` (case-insensitive).

    v1 supports only ``LAN`` (the StaticProvider). Cloud backends (e.g. ConoHa)
    will register here later.
    """
    key = name.strip().lower()
    if key == "lan":
        return StaticProvider()
    raise ValueError(f"unknown provider: {name!r} (known: {', '.join(KNOWN_PROVIDERS)})")
