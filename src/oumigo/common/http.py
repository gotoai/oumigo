"""Shared async HTTP client helpers (manager -> agent, router -> vLLM).

Thin wrappers over httpx with sane timeouts/retries so call sites stay uniform.
"""

from __future__ import annotations

# def make_client(...) -> httpx.AsyncClient: ...
