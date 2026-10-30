"""Where oumigo persists local state (worker identity, manager auth state)."""

from __future__ import annotations

import os
from pathlib import Path


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
