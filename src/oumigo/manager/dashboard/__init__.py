"""Reporting plane — performance & diagnostics dashboard (V1.0).

A read-only *consumer* of what the control and data planes already collect, not a
new source of truth. Runs as its own process (fault isolation); pulls ``worker:``
and ``gpu:`` metrics from the control plane over HTTP and serves a web dashboard,
rolling the raw 5s points up to minute buckets on the fly (no materialized rollup
tables in V1.0). vLLM counters are excluded until the safe-transform layer exists.

Bundled with the manager: `run_server` spawns it automatically as a child process
(disable via `dashboard.enabled: false` in the manager config). The
``python -m oumigo.manager.dashboard`` entrypoint is what the manager spawns and is
also usable standalone for development (defaults to port 7080).
"""

from oumigo.manager.dashboard.server import create_dashboard_app, run_dashboard

__all__ = ["create_dashboard_app", "run_dashboard"]
