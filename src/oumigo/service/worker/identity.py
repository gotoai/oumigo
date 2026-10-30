"""Worker node identity: a deterministic hash of the host id + actual vLLM port.

The `node_id` is derived (`oumigo.common.identity.compute_node_id`) from a *host id*
and the port. The host id is a random UUID minted once and persisted here, or taken
from `$OUMIGO_HOST_ID` — useful on pods whose disk is wiped on restart (e.g. a RunPod
container disk) that should still come back as the same node. Same host + same port
always yields the same id, so restarts don't create phantom nodes, and an address
change (DHCP, NAT, tunnel) doesn't either.

Also persisted is `incarnation`: a per-node counter bumped each start so the manager
can tell "same node, restarted" apart from stale duplicates. It is keyed by node_id
because one host may run several workers (different negotiated ports -> different ids).

The port is known only after the coordinator preflights a free one, so identity is
resolved *after* port selection (not at process start).

Persist location: /var/lib/oumigo when writable (systemd service), else the XDG
state dir (~/.local/state/oumigo). Overridable via an explicit state_dir.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from oumigo.common.identity import compute_node_id

STATE_FILE = "incarnations.json"
HOST_ID_FILE = "host_id"
ENV_HOST_ID = "OUMIGO_HOST_ID"


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


def resolve_host_id(directory: Path) -> str:
    """The host id: `$OUMIGO_HOST_ID` if set, else the persisted UUID (minted on first use)."""
    env = os.environ.get(ENV_HOST_ID, "").strip()
    if env:
        return env
    path = directory / HOST_ID_FILE
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    host_id = str(uuid.uuid4())
    _atomic_write(path, host_id + "\n")
    return host_id


def resolve_worker_identity(port: int, state_dir: Path | None = None) -> tuple[str, int, Path]:
    """Return (node_id, incarnation, path) for this host's worker at ``port``.

    `node_id` is derived from (host id, port); `incarnation` is read for that id,
    bumped, and persisted. First run for an id yields incarnation 1.
    """
    directory = Path(state_dir) if state_dir else default_state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    node_id = compute_node_id(resolve_host_id(directory), port)
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
