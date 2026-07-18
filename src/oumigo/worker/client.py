"""HTTP client for the worker -> manager control-plane calls."""

from __future__ import annotations

import httpx

from oumigo.protocol.messages import (
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterRequest,
    RegisterResponse,
)


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def register(
    manager_url: str, req: RegisterRequest, token: str | None = None, timeout: float = 10.0
) -> RegisterResponse:
    resp = httpx.post(
        f"{manager_url.rstrip('/')}/register",
        json=req.model_dump(mode="json"),
        headers=_headers(token),
        timeout=timeout,
    )
    resp.raise_for_status()
    return RegisterResponse.model_validate(resp.json())


def heartbeat(
    manager_url: str, req: HeartbeatRequest, token: str | None = None, timeout: float = 10.0
) -> HeartbeatResponse:
    resp = httpx.post(
        f"{manager_url.rstrip('/')}/heartbeat",
        json=req.model_dump(mode="json"),
        headers=_headers(token),
        timeout=timeout,
    )
    resp.raise_for_status()
    return HeartbeatResponse.model_validate(resp.json())
