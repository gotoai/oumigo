"""In-memory sqlite metrics store (v1).

Holds the manager's historical time-series: one long/narrow fact table,
``metric_fact(node_id, metric, timestamp, value)``, capped at the most recent
``retention_s`` (24 h by default). Ingest is an **upsert** by
``(node_id, metric, timestamp)`` so a retried/overlapping worker batch rewrites
the same grid slots rather than duplicating them (see docs/metrics.md).

In-memory means the store is lost on a manager restart — consistent with the
registry, which is likewise rebuilt from re-registrations. Swap ``:memory:`` for a
file path (WAL) if restart durability is ever needed; the schema is unchanged.

Thread-safe defensively (a single shared connection + lock), matching the
registry: the control-plane server is async-single-thread today, but nothing here
assumes that.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from oumigo.protocol.messages import MetricPoint

TS_FORMAT = "%Y-%m-%d %H:%M:%S"  # UTC; matches the worker's grid-slot stamp

RETENTION_S = 24 * 60 * 60  # keep at most 24 hours of data points


class MetricStore:
    """A capped, upsert-keyed fact table backed by in-memory sqlite."""

    def __init__(self, retention_s: float = RETENTION_S) -> None:
        self.retention_s = retention_s
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_fact (
                    node_id   TEXT NOT NULL,
                    metric    TEXT NOT NULL,
                    timestamp TEXT NOT NULL,   -- 'YYYY-MM-DD HH:MM:SS' UTC, grid slot
                    value     REAL NOT NULL,
                    PRIMARY KEY (node_id, metric, timestamp)
                )
                """
            )
            # Range pruning is by timestamp, which is last in the PK; give it its own index.
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_metric_fact_ts ON metric_fact(timestamp)"
            )

    # --- ingest --------------------------------------------------------------

    def ingest(
        self, node_id: str, points: Iterable[MetricPoint], now: datetime | None = None
    ) -> int:
        """Upsert a batch and prune anything older than the retention window.

        Returns the number of rows written. `now` is injectable for testing.
        """
        rows = [(node_id, p.metric, p.timestamp, p.value) for p in points]
        with self._lock, self._conn:
            if rows:
                self._conn.executemany(
                    "INSERT INTO metric_fact (node_id, metric, timestamp, value) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(node_id, metric, timestamp) DO UPDATE SET value = excluded.value",
                    rows,
                )
            self._prune_locked(now)
        return len(rows)

    def _prune_locked(self, now: datetime | None = None) -> None:
        cutoff = self._cutoff(now)
        self._conn.execute("DELETE FROM metric_fact WHERE timestamp < ?", (cutoff,))

    def _cutoff(self, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        # Lexical string compare is chronological for the fixed-width UTC format.
        return (now - timedelta(seconds=self.retention_s)).strftime(TS_FORMAT)

    # --- queries -------------------------------------------------------------

    def latest_per_node(self) -> list[dict]:
        """For each node, the most recent grid slot and the metrics at that slot.

        Shape: ``[{"node_id", "timestamp", "metrics": {name: value, ...}}, ...]``,
        ordered by node_id.
        """
        with self._lock:
            latest = dict(
                self._conn.execute(
                    "SELECT node_id, MAX(timestamp) FROM metric_fact GROUP BY node_id"
                ).fetchall()
            )
            out: list[dict] = []
            for node_id, ts in latest.items():
                metrics = dict(
                    self._conn.execute(
                        "SELECT metric, value FROM metric_fact "
                        "WHERE node_id = ? AND timestamp = ? ORDER BY metric",
                        (node_id, ts),
                    ).fetchall()
                )
                out.append({"node_id": node_id, "timestamp": ts, "metrics": metrics})
        out.sort(key=lambda r: r["node_id"])
        return out

    def since(self, after: str, prefixes: Iterable[str] | None = None) -> list[dict]:
        """Raw points with ``timestamp > after``, optionally filtered by metric prefix.

        Powers the reporting plane's incremental pull (a separate process that can't
        share this in-memory connection). ``after`` is a grid-slot timestamp string
        (``''`` returns the whole retained window); ``prefixes`` keeps only metrics
        starting with any of them (e.g. ``('worker:', 'gpu:')``). Rows ascend by
        timestamp so the caller can advance its watermark to the last one seen.
        """
        query = "SELECT node_id, metric, timestamp, value FROM metric_fact WHERE timestamp > ?"
        params: list = [after]
        prefixes = list(prefixes or ())
        if prefixes:
            # Our prefixes are fixed literals with no LIKE wildcards ('_'/'%'), so a
            # plain LIKE is safe here; '%' is the intended "rest of the name" match.
            query += " AND (" + " OR ".join(["metric LIKE ?"] * len(prefixes)) + ")"
            params += [f"{p}%" for p in prefixes]
        query += " ORDER BY timestamp"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [{"node_id": n, "metric": m, "timestamp": t, "value": v} for n, m, t, v in rows]

    def count(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM metric_fact").fetchone()
        return int(n)
