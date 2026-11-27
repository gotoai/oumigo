"""The OumiGo console — the manager's web UI: Workers (with Close) and GPU Utilization.

Runs as its own process (fault isolation). It reads the fleet and its ``worker:`` /
``gpu:`` metrics from the control plane over HTTP; its only write is a close request in
the manager's state-dir auth store, which the running manager acts on. vLLM counters are
excluded from the charts until the safe-transform layer exists.

Bundled with the manager: `run_server` spawns it automatically as a child process
(disable via `dashboard.enabled: false` in the manager config). The
``python -m oumigo.service.manager.dashboard`` entrypoint is what the manager spawns and is
also usable standalone for development (defaults to port 7080).
"""

from oumigo.service.manager.dashboard.server import create_dashboard_app, run_dashboard

__all__ = ["create_dashboard_app", "run_dashboard"]
