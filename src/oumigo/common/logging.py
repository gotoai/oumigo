"""Logging setup shared across the manager process."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Loggers whose INFO output is "server output" the user may want to see or hide.
_APP_LOGGERS = ("oumigo", "uvicorn", "uvicorn.error", "httpx")


def configure_logging(verbose: bool) -> None:
    """Install a root handler and set per-library verbosity."""
    logging.basicConfig(level=logging.INFO, format=_FORMAT)
    logging.getLogger("zeroconf").setLevel(logging.WARNING)  # always quiet (multicast noise)
    set_verbosity(verbose)


def set_verbosity(verbose: bool) -> None:
    """Show (verbose) or suppress (quiet) server/library INFO logs — safe to call at runtime."""
    level = logging.INFO if verbose else logging.WARNING
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(level)
