"""Config-file resolution.

Resolves *which file* holds the manager config, in precedence order:

    --config-file <path>  >  $OUMIGO_CONFIG_FILE  >  ~/.config/oumigo/manager.yaml
                                                   >  /etc/oumigo/manager.yaml

First existing file wins. There is a single config file (no config directory) —
model settings live inside it. This only *locates* the file; YAML parsing and
value-level precedence (CLI > env > file > default) live elsewhere. Not finding a
file is not an error here — callers decide whether the resulting config is
sufficient.
"""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_FILENAME = "manager.yaml"


def config_file_search_path(explicit: Path | str | None = None) -> list[Path]:
    """Ordered candidate config files, highest precedence first."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("OUMIGO_CONFIG_FILE")
    if env:
        candidates.append(Path(env).expanduser())
    xdg = os.environ.get("XDG_CONFIG_HOME")
    user_config = Path(xdg) if xdg else Path.home() / ".config"
    candidates.append(user_config / "oumigo" / CONFIG_FILENAME)
    candidates.append(Path("/etc/oumigo") / CONFIG_FILENAME)
    return candidates


def resolve_config_file(explicit: Path | str | None = None) -> Path | None:
    """First existing file in the search path, or None if none exist."""
    for candidate in config_file_search_path(explicit):
        if candidate.is_file():
            return candidate
    return None
