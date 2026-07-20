"""Run the reporting-plane dashboard: ``python -m oumigo.manager.dashboard``.

Normally the manager spawns this automatically (see ``control.server.run_server``);
this entrypoint is what it spawns, and is also usable standalone for development.
A separate process by design (own plane, fault-isolated). Point it at the manager
control plane; it pulls ``worker:``/``gpu:`` metrics and serves the web UI.
"""

from __future__ import annotations

import argparse

from oumigo.manager.dashboard.server import run_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m oumigo.manager.dashboard",
        description="oumigo reporting-plane dashboard (V1.0).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Web bind host.")  # noqa: S104
    parser.add_argument("--port", type=int, default=7080, help="Web bind port (default 7080).")
    parser.add_argument(
        "--control-url",
        default="http://127.0.0.1:7014",
        help="Manager control-plane base URL (default matches the control-plane port).",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="Seconds between metric pulls."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Log at INFO.")
    args = parser.parse_args()

    run_dashboard(
        args.host,
        args.port,
        args.control_url,
        poll_interval_s=args.poll_interval,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
