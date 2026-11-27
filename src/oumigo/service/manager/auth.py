"""Manager-side bearer-token resolution (v1).

Shared-secret auth: the manager holds the token it expects workers to present,
and every worker presents the same value.

v1 policy is intentionally **fail-open**: if no token is configured (neither
``MG_JOIN_TOKEN`` nor a token file), authentication is *disabled* and a
warning is emitted. This is a development convenience; production deployments
should configure a token. Later phases will likely make this fail-closed.
"""

from __future__ import annotations

import os
from pathlib import Path

from oumigo.common.envnames import MG_JOIN_TOKEN

ENV_TOKEN = MG_JOIN_TOKEN

NO_TOKEN_WARNING = (
    "Warning: MG_JOIN_TOKEN is not set, worker node authentication will "
    "be automatically disabled. Please refer to OUMIGO relevant documents for "
    "more details."
)

NO_TOKEN_WAN_NOTICE = (
    "Notice: no reusable join token (MG_JOIN_TOKEN / auth.token_file) is set. "
    "Under provider CUSTOM worker authentication stays ENABLED: workers join with "
    "per-pod tokens from `oumigo manager token create`."
)


def resolve_manager_token(token_file: Path | None = None) -> str | None:
    """Return the shared bearer token, or ``None`` if auth should be disabled.

    Precedence: ``$MG_JOIN_TOKEN`` (value) > ``token_file`` (contents) >
    ``None``. A ``token_file`` that was explicitly requested but does not exist
    is an error (the caller clearly intended auth), raised as ``FileNotFoundError``.
    """
    env_token = os.environ.get(ENV_TOKEN)
    if env_token and env_token.strip():
        return env_token.strip()
    return read_token_file(token_file)


def read_token_file(token_file: Path | None) -> str | None:
    """The token in `token_file` (stripped), None for no file or an empty one. A file
    that was asked for but doesn't exist is a `FileNotFoundError`."""
    if token_file is not None:
        path = Path(token_file)
        if not path.is_file():
            raise FileNotFoundError(f"token file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text

    return None
