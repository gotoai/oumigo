"""Worker node identity: a self-generated, persisted UUID.

The worker mints its own UUID on first run, persists it, and presents it on every
registration — so restarts don't create phantom nodes. Identity never expires on
a timer; only the *session* (heartbeat) does. Each start bumps `incarnation` so
the manager can tell "same node, restarted" apart from stale duplicates.

Persist location: /var/lib/oumigo when writable (systemd service), else the XDG
state dir (~/.local/state/oumigo). Overridable via an explicit state_dir.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

STATE_FILE = "node_id.json"


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


def resolve_node_identity(state_dir: Path | None = None) -> tuple[str, int, Path]:
    """Return (node_id, incarnation, path). Generates+persists on first run."""
    directory = Path(state_dir) if state_dir else default_state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / STATE_FILE

    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        node_id = data["node_id"]
        incarnation = int(data.get("incarnation", 0)) + 1
    else:
        node_id = str(uuid.uuid4())
        incarnation = 1
        data = {}

    data.update({"node_id": node_id, "incarnation": incarnation})
    _atomic_write(path, json.dumps(data))
    return node_id, incarnation, path


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)  # atomic rename; no half-written identity on crash
