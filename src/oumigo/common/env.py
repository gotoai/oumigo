"""Minimal .env loader (stdlib only — no python-dotenv dependency).

Reads ``KEY=VALUE`` lines from a .env file into ``os.environ`` so the values are
inherited by child processes (notably the vLLM server the worker spawns). By
default it does NOT override variables already present in the real environment —
an explicit shell/systemd export wins over the file, matching dotenv convention.

Deliberately small: `KEY=VALUE`, `#` comments, blank lines, an optional leading
`export `, and surrounding single/double quotes. No variable interpolation, no
`~` expansion (use absolute paths or `$HOME` via the shell).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("oumigo")


def load_env_file(path: Path | str = ".env", *, override: bool = False) -> int:
    """Load ``KEY=VALUE`` pairs from `path` into ``os.environ``.

    Returns the number of variables applied. A missing file is a no-op (0).
    Existing environment variables are preserved unless ``override=True``.
    """
    p = Path(path)
    if not p.is_file():
        return 0

    applied = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue  # not a KEY=VALUE line
        if not override and key in os.environ:
            continue  # explicit environment wins over the file
        os.environ[key] = _unquote(value.strip())
        applied += 1

    if applied:
        log.info("loaded %d environment variable(s) from %s", applied, p)
    return applied


def _unquote(value: str) -> str:
    """Strip a single matching pair of surrounding quotes, if present."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value
