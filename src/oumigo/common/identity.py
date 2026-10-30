"""Deterministic worker node identity.

A worker's `node_id` is a stable hash of its *host id* and its *actual* vLLM port:
``sha256("oumigo:<host_id>:<port>")`` truncated to 16 hex chars. The host id is a
random UUID the worker persists in its state directory (or `$OUMIGO_HOST_ID`), so
identity is:

- **stable across restarts and address changes** — a pod or laptop whose IP moves
  (or a tunnel worker, whose address means nothing to the manager) re-registers as
  itself;
- **unique per worker on a host** — a homogeneous fleet shares one configured port,
  so two workers on one box negotiate *different* actual ports (7001, 7002, …);
  folding the real port in keeps their ids distinct.

Before 0.4.0 the hash used the LAN IP instead of the host id, so an upgraded worker
gets a new node_id once.
"""

from __future__ import annotations

import hashlib

NODE_ID_HEX_LEN = 16  # 64 bits of SHA-256; ample for a fleet, short enough to read


def compute_node_id(host_id: str, port: int) -> str:
    """Return the 16-char hex node id for the worker on ``host_id`` at ``port``."""
    digest = hashlib.sha256(f"oumigo:{host_id}:{port}".encode()).hexdigest()
    return digest[:NODE_ID_HEX_LEN]
