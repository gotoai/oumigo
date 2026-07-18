"""Config-directory resolution.

Resolves *which directory* holds the config, in precedence order:

    --config <dir>  >  $OUMIGO_CONFIG_DIR  >  ~/.config/oumigo  >  /etc/oumigo

First existing directory wins (not a merge). This only *locates* the directory;
YAML parsing and value-level precedence (CLI > env > file > default) live
elsewhere. Not finding a directory is not an error here — callers decide whether
the resulting config is sufficient.
"""

from __future__ import annotations

import os
from pathlib import Path


def config_search_path(explicit: Path | str | None = None) -> list[Path]:
    """Ordered candidate config directories, highest precedence first."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("OUMIGO_CONFIG_DIR")
    if env:
        candidates.append(Path(env).expanduser())
    xdg = os.environ.get("XDG_CONFIG_HOME")
    user_config = Path(xdg) if xdg else Path.home() / ".config"
    candidates.append(user_config / "oumigo")
    candidates.append(Path("/etc/oumigo"))
    return candidates


def resolve_config_dir(explicit: Path | str | None = None) -> Path | None:
    """First existing directory in the search path, or None if none exist."""
    for candidate in config_search_path(explicit):
        if candidate.is_dir():
            return candidate
    return None
