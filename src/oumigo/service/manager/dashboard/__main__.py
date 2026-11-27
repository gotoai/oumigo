"""Run the console: ``python -m oumigo.service.manager.dashboard``.

Normally the manager spawns this automatically (see ``control.server.run_server``) with
its state dir and env file; this entrypoint is what it spawns, and is also usable
standalone for development. A separate process by design (fault-isolated).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oumigo.service.manager.dashboard.server import run_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m oumigo.service.manager.dashboard",
        description="OumiGo console (the manager's web UI)."
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Web bind host: a loopback address (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7080, help="Web bind port (default 7080).")
    parser.add_argument(
        "--control-url",
        default="http://127.0.0.1:7014",
        help="Manager control-plane base URL (default matches the control-plane port).",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="Seconds between metric pulls."
    )
    parser.add_argument("--state-dir", type=Path, default=None,
                        help="The manager's auth state dir (close requests, session key).")
    parser.add_argument("--env-file", type=Path, default=None,
                        help="The manager's .env file (Create Admin Account writes here).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log at INFO.")
    args = parser.parse_args()

    run_dashboard(
        args.host,
        args.port,
        args.control_url,
        poll_interval_s=args.poll_interval,
        verbose=args.verbose,
        state_dir=args.state_dir,
        env_file=args.env_file,
    )


if __name__ == "__main__":
    main()
