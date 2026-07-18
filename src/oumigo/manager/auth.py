"""Manager-side bearer-token resolution (v1).

Shared-secret auth: the manager holds the token it expects workers to present,
and every worker presents the same value.

v1 policy is intentionally **fail-open**: if no token is configured (neither
``OUMIGO_MANAGER_TOKEN`` nor a token file), authentication is *disabled* and a
warning is emitted. This is a development convenience; production deployments
should configure a token. Later phases will likely make this fail-closed.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_TOKEN = "OUMIGO_MANAGER_TOKEN"

NO_TOKEN_WARNING = (
    "Warning: OUMIGO_MANAGER_TOKEN is not set, worker node authentication will "
    "be automatically disabled. Please refer to OUMIGO relevant documents for "
    "more details."
)


def resolve_manager_token(token_file: Path | None = None) -> str | None:
    """Return the shared bearer token, or ``None`` if auth should be disabled.

    Precedence: ``$OUMIGO_MANAGER_TOKEN`` (value) > ``token_file`` (contents) >
    ``None``. A ``token_file`` that was explicitly requested but does not exist
    is an error (the caller clearly intended auth), raised as ``FileNotFoundError``.
    """
    env_token = os.environ.get(ENV_TOKEN)
    if env_token and env_token.strip():
        return env_token.strip()

    if token_file is not None:
        path = Path(token_file)
        if not path.is_file():
            raise FileNotFoundError(f"token file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text

    return None
