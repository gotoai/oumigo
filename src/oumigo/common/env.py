"""Minimal .env loader (stdlib only — no python-dotenv dependency).

Reads ``KEY=VALUE`` lines from a .env file into ``os.environ`` so the values are
inherited by child processes (notably the vLLM server the worker spawns). By
default it does NOT override variables already present in the real environment —
an explicit shell/systemd export wins over the file, matching dotenv convention.

Deliberately small: `KEY=VALUE`, `#` comments, blank lines, an optional leading
`export `, and surrounding single/double quotes. No variable interpolation, no
`~` expansion (use absolute paths or `$HOME` via the shell).

`set_env_values` writes keys into a .env file (the console's Create Admin Account page),
keeping every other line as it is.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger("oumigo")


def load_env_file(path: Path | str = ".env", *, override: bool = False) -> int:
    """Load ``KEY=VALUE`` pairs from `path` into ``os.environ``.

    Returns the number of variables applied. A missing file is a no-op (0).
    Existing environment variables are preserved unless ``override=True``.
    """
    p = Path(path)
    applied = 0
    for key, value in read_env_file(p).items():
        if not override and key in os.environ:
            continue  # explicit environment wins over the file
        os.environ[key] = value
        applied += 1

    if applied:
        log.info("loaded %d environment variable(s) from %s", applied, p)
    return applied


def read_env_file(path: Path | str) -> dict[str, str]:
    """The ``KEY=VALUE`` pairs in `path` (later lines win), without applying them. A
    missing file reads as empty."""
    p = Path(path)
    if not p.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(raw)
        if parsed is not None:
            values[parsed[0]] = parsed[1]
    return values


def set_env_values(path: Path | str, values: dict[str, str]) -> None:
    """Set `values` in the .env file at `path`: replace each key's existing line (the
    first; later duplicates are removed) or append it, keeping every other line and
    comment. Atomic (a temporary file in the same directory, then a rename) and mode
    0600, as the file may hold secrets. A missing file is created."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
    out: list[str] = []
    written: set[str] = set()
    for raw in lines:
        parsed = _parse_line(raw)
        key = parsed[0] if parsed is not None else None
        if key in values:
            if key not in written:
                out.append(f"{key}={values[key]}")
                written.add(key)
            continue
        out.append(raw)
    out += [f"{key}={value}" for key, value in values.items() if key not in written]

    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _parse_line(raw: str) -> tuple[str, str] | None:
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    key, sep, value = line.partition("=")
    key = key.strip()
    if not sep or not key:
        return None  # not a KEY=VALUE line
    return key, _unquote(value.strip())


def _unquote(value: str) -> str:
    """Strip a single matching pair of surrounding quotes, if present."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value
