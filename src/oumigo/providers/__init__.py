"""L2 — cloud / provisioning adapter.

A minimal, lifecycle-shaped `Provider` protocol plus a real `StaticProvider` for
LAN/manual hosts. Cloud backends (e.g. ConoHa, which is OpenStack-based) are
future implementations of the same protocol — extracted on the *second*
implementation, not designed up front.
"""

from oumigo.providers.base import Provider
from oumigo.providers.factory import KNOWN_PROVIDERS, create_provider

__all__ = ["KNOWN_PROVIDERS", "Provider", "create_provider"]
