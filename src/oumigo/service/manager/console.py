"""Interactive manager console — an HTTP client.

Attaches to a running control-plane server over HTTP (`/status`, `/workers`).
`oumigo manager run` (on a TTY) spawns the server as a child process and runs this
console against it; `verbose` toggles the child server's log verbosity via signals.
"""

from __future__ import annotations

import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from oumigo import __version__

try:
    # Importing readline transparently upgrades input() with line editing and
    # arrow-key history (like bash). Absent it, arrow keys emit raw escape codes.
    import readline
except ImportError:  # pragma: no cover - readline is stdlib on Linux (mandated platform)
    readline = None  # type: ignore[assignment]

PROMPT = "oumigo> "
HISTORY_LENGTH = 200  # commands retained in the console history cache


class ManagerConsole:
    """REPL that talks to the control-plane server over HTTP."""

    def __init__(self, base_url: str, server_pid: int | None = None, verbose: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.server_pid = server_pid
        self.verbose = verbose
        self._running = False

    def run(self) -> None:
        self._running = True
        self._init_history()
        self._banner()
        try:
            while self._running:
                try:
                    line = input(PROMPT).strip()
                except EOFError:  # Ctrl-D → clean exit
                    print()
                    self._quit()
                    break
                except KeyboardInterrupt:  # Ctrl-C cancels the line, not the process
                    print()
                    continue
                if line:
                    self._dispatch(line)
        finally:
            self._save_history()

    # --- command history (bash-like arrow-key navigation) --------------------

    @staticmethod
    def _history_path() -> Path:
        """User-level history file, alongside other oumigo state (XDG state dir)."""
        state_home = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local/state")
        return Path(state_home) / "oumigo" / "console_history"

    def _init_history(self) -> None:
        if readline is None:
            return
        path = self._history_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                readline.read_history_file(str(path))
        except OSError:
            pass  # history is a convenience; never block the console on it
        readline.set_history_length(HISTORY_LENGTH)

    def _save_history(self) -> None:
        if readline is None:
            return
        try:
            readline.set_history_length(HISTORY_LENGTH)
            readline.write_history_file(str(self._history_path()))
        except OSError:
            pass

    def _dispatch(self, line: str) -> None:
        cmd, *args = line.split()
        handler = self._commands.get(cmd)
        if handler is None:
            print(f"unknown command: {cmd!r} (try 'help')")
            return
        handler(args)

    @property
    def _commands(self):
        return {
            "help": self._help,
            "status": self._status,
            "workers": self._workers,
            "metrics": self._metrics,
            "verbose": self._verbose,
            "version": lambda args: print(__version__),
            "quit": lambda args: self._quit(),
            "exit": lambda args: self._quit(),
        }

    def _get(self, path: str):
        try:
            resp = httpx.get(self.base_url + path, timeout=3.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - surface any client/server error to the REPL
            print(f"(server unreachable: {exc})")
            return None

    def _banner(self) -> None:
        print(f"oumigo manager {__version__}")
        print(f"attached to control plane: {self.base_url}")
        print("type 'help' for commands, 'quit' to exit.")

    def _help(self, args: list[str]) -> None:
        print("commands:")
        print("  help              show this help")
        print("  status            show manager status")
        print("  workers           list registered workers (id, ip, port, state, uptime)")
        print("  metrics           show each worker's latest received grid metrics")
        print("  verbose [on|off]  stream server logs to the console (toggle)")
        print("  version           print the oumigo version")
        print("  quit | exit       stop the manager (server + console)")

    def _status(self, args: list[str]) -> None:
        data = self._get("/status")
        if data is None:
            return
        print("manager status:")
        print(f"  control plane : {self.base_url}")
        print(f"  provider      : {data.get('provider', '?')}")
        print(f"  auth          : {'enabled' if data.get('auth') else 'disabled (no token)'}")
        print(f"  verbose       : {'on' if self.verbose else 'off'}")
        print(f"  workers       : {data.get('workers', 0)} registered")

    # vLLM stamps this the moment a worker (re)enters SERVING; float UTC epoch seconds.
    _VLLM_START_METRIC = "vllm:start_timestamp"

    def _workers(self, args: list[str]) -> None:
        data = self._get("/workers")
        if data is None:
            return
        workers = data.get("workers", [])
        if not workers:
            print("no workers registered yet.")
            return

        starts = self._vllm_starts()  # node_id -> vllm:start_timestamp (float epoch)
        now = time.time()
        rows = []
        for r in sorted(workers, key=lambda r: r.get("seq", 0)):
            start = starts.get(r["node_id"])
            rows.append((
                r.get("name", f"Worker#{r.get('seq', 0)}"),
                r["node_id"],
                r.get("address") or "-",
                str(r["port"]) if r.get("port") is not None else "-",
                (r.get("capabilities") or {}).get("gpu") or "-",
                r.get("backend") or "-",
                r["state"],
                r.get("model") or "-",
                self._fmt_utc(start),
                f"{now - r['last_seen']:.0f}s ago",
            ))

        headers = ("WORKER", "HASH ID", "IP", "PORT", "GPU#0", "BACKEND", "STATE", "MODEL_NAME", "START (UTC)", "LAST SEEN")
        widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print("  " + fmt.format(*headers))
        for row in rows:
            print("  " + fmt.format(*row))

    def _vllm_starts(self) -> dict[str, float]:
        """Map node_id -> vLLM start epoch from the latest grid slot, for the uptime column."""
        data = self._get("/metrics/latest")
        if data is None:
            return {}
        out: dict[str, float] = {}
        for record in data.get("nodes", []):
            value = record.get("metrics", {}).get(self._VLLM_START_METRIC)
            if value is not None:
                out[record["node_id"]] = value
        return out

    @staticmethod
    def _fmt_utc(epoch: float | None) -> str:
        """Format a float UTC epoch as 'YYYY-MM-DD HH:MM:SS'; '-' when unknown."""
        if not epoch:
            return "-"
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _metrics(self, args: list[str]) -> None:
        data = self._get("/metrics/latest")
        if data is None:
            return
        nodes = data.get("nodes", [])
        if not nodes:
            print("no metrics received yet.")
            return
        for record in nodes:
            metrics = record.get("metrics", {})
            print(f"  {record['node_id']}  @ {record['timestamp']} UTC  ({len(metrics)} metrics)")
            for name, value in metrics.items():
                # start-timestamps are float UTC epoch seconds — show full, not 1.8e+09.
                shown = f"{value:.3f}" if name.endswith("_timestamp") else f"{value:g}"
                print(f"      {name} = {shown}")

    def _verbose(self, args: list[str]) -> None:
        if args and args[0].lower() in ("on", "off"):
            self.verbose = args[0].lower() == "on"
        else:
            self.verbose = not self.verbose
        if self.server_pid:
            try:
                os.kill(self.server_pid, signal.SIGUSR1 if self.verbose else signal.SIGUSR2)
            except ProcessLookupError:
                print("(server process not found)")
        print(f"verbose logging {'on' if self.verbose else 'off'}")

    def _quit(self) -> None:
        print("stopping manager. bye.")
        self._running = False
