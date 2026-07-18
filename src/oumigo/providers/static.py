"""StaticProvider — the real first implementation, for LAN/manual hosts.

Nodes are declared in config (e.g. your two LAN machines). `provision` returns a
known host and ensures its agent is up; `terminate` stops the agent; `list` reads
the declared set. Validates the Provider protocol's shape before any cloud
backend exists — ConoHa later is just implementation #2.
"""

from __future__ import annotations

# class StaticProvider:
#     """Provider backed by a static list of hosts from config."""
