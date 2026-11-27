"""Environment variable names: the naming contract.

oumigo reads its own settings only from names with a side's prefix:

- ``MG_*`` — read by the **manager** (`oumigo manager ...`, including its web console);
- ``WK_*`` — read by a **worker** (`oumigo worker run`).

Variables owned by other software keep their own names (``HF_TOKEN``, ``HF_HOME``,
``VLLM_*``, ``CUDA_VISIBLE_DEVICES``, ``XDG_*``, ``RUNPOD_POD_ID`` ...). See
docs/environmental_variables.md.

Names used before 0.5.8 are refused with a message naming the new one, rather than
ignored: a setting that is silently dropped (e.g. the worker's model profile) would make
the process run with something else than its operator chose.
"""

from __future__ import annotations

import os

# --- manager -------------------------------------------------------------------------
MG_CONFIG_FILE = "MG_CONFIG_FILE"  # manager.yaml path
MG_JOIN_TOKEN = "MG_JOIN_TOKEN"  # the reusable join token workers may present
MG_ADMIN_USER = "MG_ADMIN_USER"  # web console admin login
MG_ADMIN_PASSWORD_HASH = "MG_ADMIN_PASSWORD_HASH"  # its scrypt hash

# --- worker --------------------------------------------------------------------------
WK_MANAGER_URL = "WK_MANAGER_URL"
WK_JOIN_TOKEN = "WK_JOIN_TOKEN"
WK_TRANSPORT = "WK_TRANSPORT"
WK_MANAGER_CA_FILE = "WK_MANAGER_CA_FILE"
WK_HOST_ID = "WK_HOST_ID"
WK_MODEL_PROFILE = "WK_MODEL_PROFILE"
WK_GPU_ID = "WK_GPU_ID"
WK_DISCOVER_TIMEOUT = "WK_DISCOVER_TIMEOUT"
WK_DRAIN_TIMEOUT = "WK_DRAIN_TIMEOUT"
WK_METRICS_MAX_KEEPING_AGE = "WK_METRICS_MAX_KEEPING_AGE"
WK_MODEL_STORAGE_LOCATION = "WK_MODEL_STORAGE_LOCATION"

MANAGER = "manager"
WORKER = "worker"

# Names oumigo no longer reads, per side, with what to set instead.
RETIRED: dict[str, dict[str, str]] = {
    MANAGER: {
        "OUMIGO_MANAGER_TOKEN": f"renamed to {MG_JOIN_TOKEN}",
        "OUMIGO_CONFIG_FILE": f"renamed to {MG_CONFIG_FILE}",
        "OUMIGO_ADMIN_USER": f"renamed to {MG_ADMIN_USER}",
        "OUMIGO_ADMIN_PASSWORD_HASH": f"renamed to {MG_ADMIN_PASSWORD_HASH}",
    },
    WORKER: {
        "OUMIGO_MANAGER_URL": f"renamed to {WK_MANAGER_URL}",
        "OUMIGO_JOIN_TOKEN": f"renamed to {WK_JOIN_TOKEN}",
        "OUMIGO_MANAGER_TOKEN": f"a worker's join token is now {WK_JOIN_TOKEN}",
        "OUMIGO_TRANSPORT": f"renamed to {WK_TRANSPORT}",
        "OUMIGO_MANAGER_CA_FILE": f"renamed to {WK_MANAGER_CA_FILE}",
        "OUMIGO_HOST_ID": f"renamed to {WK_HOST_ID}",
        "OUMIGO_MODEL_PROFILE": f"renamed to {WK_MODEL_PROFILE}",
        "OUMIGO_GPU_ID": f"renamed to {WK_GPU_ID}",
        "OUMIGO_DISCOVER_TIMEOUT": f"renamed to {WK_DISCOVER_TIMEOUT}",
        "OUMIGO_DRAIN_TIMEOUT": f"renamed to {WK_DRAIN_TIMEOUT}",
        "MODEL_STORAGE_LOCATION": f"renamed to {WK_MODEL_STORAGE_LOCATION}",
        "MODEL_NAME": f"removed in 0.5.5; choose a model with {WK_MODEL_PROFILE}",
        "MAX_MODEL_LEN": f"removed in 0.5.5; choose a profile with that context length "
                         f"({WK_MODEL_PROFILE})",
    },
}


def retired_in_env(side: str, environ: dict[str, str] | None = None) -> list[str]:
    """One line per retired name that is set (non-empty) for `side`."""
    environ = os.environ if environ is None else environ
    return [f"{name} is no longer read: {what}"
            for name, what in RETIRED[side].items() if environ.get(name, "").strip()]


def refuse_retired(side: str, environ: dict[str, str] | None = None) -> None:
    """Exit with every retired name that is set, so all are fixed in one go."""
    problems = retired_in_env(side, environ)
    if problems:
        raise SystemExit(
            "\n".join(problems)
            + f"\nUpdate the {side}'s environment / .env (see docs/environmental_variables.md).")
