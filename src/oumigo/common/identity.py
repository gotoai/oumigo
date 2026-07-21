"""Deterministic worker node identity.

A worker's `node_id` is a stable hash of its LAN address and its *actual* vLLM
port: ``sha256("oumigo:<ip>:<port>")`` truncated to 16 hex chars. Deriving it
(rather than minting a UUID) makes identity:

- **stable across restarts** — same host + same port -> same id, so a returning
  worker re-registers as itself with no persisted state;
- **unique per worker on a host** — a homogeneous fleet shares one configured
  port, so two workers on one box negotiate *different* actual ports (7001, 7002,
  …); folding the real port in keeps their ids distinct.

Both worker and manager derive it identically, so it never has to be trusted over
the wire — the manager can recompute and validate it from (address, port).
"""

from __future__ import annotations

import hashlib

NODE_ID_HEX_LEN = 16  # 64 bits of SHA-256; ample for a LAN fleet, short enough to read


def compute_node_id(address: str, port: int) -> str:
    """Return the 16-char hex node id for a worker at ``address:port``."""
    digest = hashlib.sha256(f"oumigo:{address}:{port}".encode()).hexdigest()
    return digest[:NODE_ID_HEX_LEN]
