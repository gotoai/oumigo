"""Worker node identity: a deterministic hash of address + actual vLLM port.

The `node_id` is derived (`oumigo.common.identity.compute_node_id`), not minted —
same host + same port always yields the same id, so restarts don't create phantom
nodes and no id has to be persisted. What *is* persisted is `incarnation`: a
per-node counter bumped each start so the manager can tell "same node, restarted"
apart from stale duplicates. It is keyed by node_id because one host may run
several workers (different negotiated ports -> different ids).

The port is known only after the coordinator preflights a free one, so identity is
resolved *after* port selection (not at process start).

Persist location: /var/lib/oumigo when writable (systemd service), else the XDG
state dir (~/.local/state/oumigo). Overridable via an explicit state_dir.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from oumigo.common.identity import compute_node_id

STATE_FILE = "incarnations.json"


def default_state_dir() -> Path:
    """Prefer /var/lib/oumigo (service state); fall back to $XDG_STATE_HOME/oumigo."""
    var = Path("/var/lib/oumigo")
    try:
        var.mkdir(parents=True, exist_ok=True)
        if os.access(var, os.W_OK):
            return var
    except (PermissionError, OSError):
        pass
    state_home = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local/state")
    return Path(state_home) / "oumigo"


def resolve_worker_identity(
    address: str, port: int, state_dir: Path | None = None
) -> tuple[str, int, Path]:
    """Return (node_id, incarnation, path) for a worker at ``address:port``.

    `node_id` is derived from (address, port); `incarnation` is read for that id,
    bumped, and persisted. First run for an id yields incarnation 1.
    """
    node_id = compute_node_id(address, port)
    directory = Path(state_dir) if state_dir else default_state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / STATE_FILE

    incarnations: dict[str, int] = {}
    if path.is_file():
        try:
            incarnations = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            incarnations = {}  # corrupt state is not worth failing a worker start over

    incarnation = int(incarnations.get(node_id, 0)) + 1
    incarnations[node_id] = incarnation
    _atomic_write(path, json.dumps(incarnations))
    return node_id, incarnation, path


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)  # atomic rename; no half-written identity on crash
