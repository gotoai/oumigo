"""Metric mirror — the reporting plane's incremental pull from the control plane.

The dashboard runs in its own process, so it can't touch the control plane's
in-memory store directly; it pulls raw points over HTTP (``GET /metrics/since``)
into a local, time-bounded buffer that the on-the-fly rollup reads. V1.0: purely
in-memory, horizon = this process's lifetime (seeded by the first pull, which can
only reach back as far as the control plane still retains).

Only ``worker:`` and ``gpu:`` domains are pulled — vLLM counters are excluded until
the safe-transform layer exists.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

log = logging.getLogger("oumigo.service.manager.dashboard")

TS_FORMAT = "%Y-%m-%d %H:%M:%S"


class MetricMirror:
    """A dict-keyed, time-bounded mirror of the control plane's recent gauge points.

    Keyed by ``(node_id, metric, timestamp)`` so a re-pulled slot upserts rather than
    duplicates. Pulls use a watermark with a small overlap window so a lagging
    worker's slots (stamped below another worker's max) aren't skipped.
    """

    def __init__(
        self,
        control_url: str,
        *,
        prefixes: tuple[str, ...] = ("worker:", "gpu:"),
        retention_min: int = 90,
        overlap_s: int = 60,
    ) -> None:
        self.control_url = control_url.rstrip("/")
        self.prefixes = prefixes
        self.retention = timedelta(minutes=retention_min)
        self.overlap = timedelta(seconds=overlap_s)
        self._rows: dict[tuple[str, str, str], float] = {}
        self._max_ts: str = ""
        self.node_info: dict[str, dict] = {}  # node_id -> {"seq", "name"} from /workers
        self.last_ok: datetime | None = None
        self.last_error: str | None = None

    def _after(self, now: datetime) -> str:
        """The `after` bound for the next pull: watermark minus overlap, or window start."""
        if not self._max_ts:
            return (now - self.retention).strftime(TS_FORMAT)
        seen = datetime.strptime(self._max_ts, TS_FORMAT).replace(tzinfo=timezone.utc)
        return (seen - self.overlap).strftime(TS_FORMAT)

    async def refresh(self, client: httpx.AsyncClient) -> None:
        """Pull new points since the watermark, upsert them, and prune the window."""
        now = datetime.now(timezone.utc)
        params = {"after": self._after(now), "prefix": ",".join(self.prefixes)}
        try:
            resp = await client.get(
                f"{self.control_url}/metrics/since", params=params, timeout=5.0
            )
            resp.raise_for_status()
            points = resp.json().get("points", [])
        except (httpx.HTTPError, ValueError) as exc:  # network down / bad JSON
            self.last_error = str(exc)
            log.warning("metrics pull from %s failed: %s", self.control_url, exc)
            return

        for p in points:
            ts = p["timestamp"]
            self._rows[(p["node_id"], p["metric"], ts)] = float(p["value"])
            if ts > self._max_ts:
                self._max_ts = ts
        self._prune(now)
        await self._refresh_names(client)
        self.last_ok = now
        self.last_error = None
        log.debug("pulled %d points; buffer now %d rows", len(points), len(self._rows))

    async def _refresh_names(self, client: httpx.AsyncClient) -> None:
        """Pull the manager's node_id -> Worker#N mapping from `/workers`.

        Best-effort: on failure keep the last-known names so the chart stays labeled.
        """
        try:
            resp = await client.get(f"{self.control_url}/workers", timeout=5.0)
            resp.raise_for_status()
            workers = resp.json().get("workers", [])
        except (httpx.HTTPError, ValueError):
            return
        self.node_info = {
            n["node_id"]: {"seq": n.get("seq", 0), "name": n.get("name") or n["node_id"][:8]}
            for n in workers
            if "node_id" in n
        }

    def _prune(self, now: datetime) -> None:
        cutoff = (now - self.retention).strftime(TS_FORMAT)
        stale = [k for k in self._rows if k[2] < cutoff]  # k[2] is the timestamp
        for k in stale:
            del self._rows[k]

    def snapshot(self) -> list[dict]:
        """Rows as dicts for the aggregator. Safe to iterate — single-threaded loop."""
        return [
            {"node_id": n, "metric": m, "timestamp": t, "value": v}
            for (n, m, t), v in self._rows.items()
        ]

    def __len__(self) -> int:
        return len(self._rows)
