"""Durable GPU-utilization rollups for the console's long charts.

The manager keeps raw 5 s samples in memory for 24 hours, and the console's mirror keeps
90 minutes; both are lost on a restart. The long charts therefore read from rollups the
console builds as samples arrive and keeps in a small SQLite file in the manager's state
directory:

| tier  | grid     | window  |
|-------|----------|---------|
| `48h` | 5 min    | 48 h    |
| `10d` | 30 min   | 10 days |
| `45d` | 2 h      | 45 days |

Each bucket holds the sum and count of a node's 5 s values (the mean across its GPUs per
slot), so a point is the **mean of every sample in its bucket**; the newest bucket is
shown while it fills. Buckets align to the UTC grid. A per-node watermark (the last slot
added) makes ingesting idempotent: the mirror re-pulls overlapping slots, and after a
console restart it re-seeds 90 minutes, but a slot is only ever added once. Workers
report their slots in order, so a slot at or before the watermark was already counted.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from oumigo.service.manager.dashboard import tokens as tk

ROLLUP_DB_FILE = "console_rollups.db"
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
_GPU_UTIL_RE = re.compile(r"^gpu:#(\d+)_util_pct$")
_UNNAMED_SEQ = 10**9  # sorts nodes without a Worker#N after named ones


@dataclass(frozen=True)
class Tier:
    key: str
    grid_s: int
    window_s: int


TIERS = {
    "48h": Tier("48h", 5 * 60, 48 * 3600),
    "10d": Tier("10d", 30 * 60, 10 * 86400),
    "45d": Tier("45d", 2 * 3600, 45 * 86400),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bucket (
    tier TEXT NOT NULL,
    node_id TEXT NOT NULL,
    start INTEGER NOT NULL,          -- bucket start, UTC epoch seconds
    total REAL NOT NULL,
    samples INTEGER NOT NULL,
    PRIMARY KEY (tier, node_id, start)
);
CREATE TABLE IF NOT EXISTS watermark (
    node_id TEXT PRIMARY KEY,
    slot INTEGER NOT NULL            -- the last 5 s slot added, UTC epoch seconds
);
CREATE TABLE IF NOT EXISTS token_bucket (
    tier TEXT NOT NULL,
    node_id TEXT NOT NULL,
    kind TEXT NOT NULL,              -- generated | input
    start INTEGER NOT NULL,
    tokens REAL NOT NULL,            -- tokens counted in the bucket
    PRIMARY KEY (tier, node_id, kind, start)
);
CREATE TABLE IF NOT EXISTS counter_mark (
    node_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    slot INTEGER NOT NULL,           -- the last sample counted
    value REAL NOT NULL,             -- the counter's value then
    PRIMARY KEY (node_id, kind)
);
"""


def _epoch(ts: str) -> int:
    return int(datetime.strptime(ts, TS_FORMAT).replace(tzinfo=timezone.utc).timestamp())


class RollupStore:
    """The tiers in SQLite (``path=None``: in memory, for tests and a console started
    without a state directory)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        if path is not None:
            path.chmod(0o600)

    @classmethod
    def at(cls, state_dir: Path) -> RollupStore:
        return cls(Path(state_dir) / ROLLUP_DB_FILE)

    # --- ingest --------------------------------------------------------------------

    def ingest(self, rows: list[dict], now: datetime | None = None) -> int:
        """Add the 5 s slots in `rows` (mirror points) not added before; returns how many
        (node, slot) values were added. Also drops buckets older than their tier."""
        slots: dict[tuple[str, int], list[float]] = {}
        for r in rows:
            if _GPU_UTIL_RE.match(r["metric"]):
                slots.setdefault((r["node_id"], _epoch(r["timestamp"])), []).append(
                    float(r["value"]))
        with self._lock:
            marks = dict(self._conn.execute("SELECT node_id, slot FROM watermark"))
            added = 0
            newest: dict[str, int] = {}
            for (node_id, slot), values in sorted(slots.items(), key=lambda kv: kv[0][1]):
                if slot <= marks.get(node_id, -1):
                    continue
                value = sum(values) / len(values)  # the mean across the node's GPUs
                for tier in TIERS.values():
                    start = slot - slot % tier.grid_s
                    self._conn.execute(
                        "INSERT INTO bucket (tier, node_id, start, total, samples)"
                        " VALUES (?, ?, ?, ?, 1) ON CONFLICT (tier, node_id, start) DO UPDATE"
                        " SET total = total + excluded.total, samples = samples + 1",
                        (tier.key, node_id, start, value))
                newest[node_id] = max(slot, newest.get(node_id, slot))
                added += 1
            for node_id, slot in newest.items():
                self._conn.execute(
                    "INSERT INTO watermark (node_id, slot) VALUES (?, ?) ON CONFLICT (node_id)"
                    " DO UPDATE SET slot = excluded.slot", (node_id, slot))
            self._ingest_counters_locked(rows)
            self._prune_locked((now or datetime.now(timezone.utc)).timestamp())
            self._conn.commit()
        return added

    def _ingest_counters_locked(self, rows: list[dict]) -> None:
        """Token counters: add each new sample's delta (see `tokens.delta`) to the
        token buckets of every tier. The first sample of a node only sets its mark."""
        samples: dict[tuple[str, str], dict[int, float]] = {}
        for r in rows:
            kind = tk.COUNTER_KIND.get(r["metric"])
            if kind is not None:
                samples.setdefault((r["node_id"], kind), {})[_epoch(r["timestamp"])] = float(
                    r["value"])
        marks = {(n, k): (slot, value) for n, k, slot, value in self._conn.execute(
            "SELECT node_id, kind, slot, value FROM counter_mark")}
        for (node_id, kind), by_slot in samples.items():
            mark = marks.get((node_id, kind))
            for slot in sorted(by_slot):
                if mark is not None and slot <= mark[0]:
                    continue
                value = by_slot[slot]
                tokens = tk.delta(mark[1] if mark else None, value)
                if tokens:
                    for tier in TIERS.values():
                        self._conn.execute(
                            "INSERT INTO token_bucket (tier, node_id, kind, start, tokens)"
                            " VALUES (?, ?, ?, ?, ?) ON CONFLICT (tier, node_id, kind, start)"
                            " DO UPDATE SET tokens = tokens + excluded.tokens",
                            (tier.key, node_id, kind, slot - slot % tier.grid_s, tokens))
                mark = (slot, value)
            if mark is not None:
                self._conn.execute(
                    "INSERT INTO counter_mark (node_id, kind, slot, value) VALUES (?, ?, ?, ?)"
                    " ON CONFLICT (node_id, kind) DO UPDATE SET slot = excluded.slot,"
                    " value = excluded.value", (node_id, kind, mark[0], mark[1]))

    def _prune_locked(self, now_epoch: float) -> None:
        for tier in TIERS.values():
            oldest = math.floor(now_epoch / tier.grid_s) * tier.grid_s - tier.window_s
            self._conn.execute("DELETE FROM bucket WHERE tier = ? AND start <= ?",
                               (tier.key, oldest))
        for tier in TIERS.values():
            oldest = math.floor(now_epoch / tier.grid_s) * tier.grid_s - tier.window_s
            self._conn.execute("DELETE FROM token_bucket WHERE tier = ? AND start <= ?",
                               (tier.key, oldest))
        longest = max(t.window_s for t in TIERS.values())
        self._conn.execute("DELETE FROM watermark WHERE slot < ?", (now_epoch - longest,))
        self._conn.execute("DELETE FROM counter_mark WHERE slot < ?", (now_epoch - longest,))

    # --- query ---------------------------------------------------------------------

    def series(self, tier_key: str, now: datetime,
               node_info: dict[str, dict] | None = None) -> dict:
        """The tier's window as a chart: one label per bucket (oldest first, the current,
        still-filling bucket last) and one series per node (bucket means, None where it
        reported nothing), labeled and ordered like the 60-minute chart."""
        tier = TIERS[tier_key]
        node_info = node_info or {}
        end = math.floor(now.timestamp() / tier.grid_s) * tier.grid_s
        count = tier.window_s // tier.grid_s
        starts = [end - tier.grid_s * (count - 1 - i) for i in range(count)]
        index = {s: i for i, s in enumerate(starts)}
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_id, start, total, samples FROM bucket WHERE tier = ? AND start >= ?",
                (tier.key, starts[0])).fetchall()
        per_node: dict[str, list[float | None]] = {}
        for node_id, start, total, samples in rows:
            if start in index:
                per_node.setdefault(node_id, [None] * count)[index[start]] = round(
                    total / samples, 2)

        def sort_key(node_id: str) -> tuple[int, str]:
            return ((node_info.get(node_id) or {}).get("seq") or _UNNAMED_SEQ, node_id)

        label_format = "%H:%M" if tier.window_s <= 86400 else "%m-%d %H:%M"
        return {
            "generated_at": now.astimezone(timezone.utc).strftime(TS_FORMAT),
            "window_s": tier.window_s,
            "grid_s": tier.grid_s,
            "unit": "%",
            "labels": [datetime.fromtimestamp(s, tz=timezone.utc).strftime(label_format)
                       for s in starts],
            "series": [
                {"node_id": node_id,
                 "label": (node_info.get(node_id) or {}).get("name") or node_id[:8],
                 "data": per_node[node_id]}
                for node_id in sorted(per_node, key=sort_key)
            ],
        }

    def token_series(self, tier_key: str, now: datetime, mode: str,
                     node_info: dict[str, dict] | None = None) -> dict:
        """The tier's window of token amounts: rate (MT / min; the current bucket divided
        by the time elapsed in it) or cumulative (MT, from the window's left edge)."""
        tier = TIERS[tier_key]
        now_epoch = now.timestamp()
        end = math.floor(now_epoch / tier.grid_s) * tier.grid_s
        count = tier.window_s // tier.grid_s
        starts = [end - tier.grid_s * (count - 1 - i) for i in range(count)]
        index = {s: i for i, s in enumerate(starts)}
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_id, kind, start, tokens FROM token_bucket"
                " WHERE tier = ? AND start >= ?", (tier.key, starts[0])).fetchall()
        per_series: dict[tuple[str, str], list[float | None]] = {}
        for node_id, kind, start, tokens in rows:
            if start in index:
                per_series.setdefault((node_id, kind), [None] * count)[index[start]] = tokens
        for key, values in per_series.items():
            if mode == tk.RATE:
                per_series[key] = [
                    None if v is None else round(
                        v / tk.MILLION / (max(min(now_epoch - s, tier.grid_s), 5.0) / 60), 6)
                    for v, s in zip(values, starts)]
            else:
                per_series[key] = [None if v is None else round(v / tk.MILLION, 6)
                                   for v in tk.cumulative(values)]
        label_format = "%H:%M" if tier.window_s <= 86400 else "%m-%d %H:%M"
        return {
            "generated_at": now.astimezone(timezone.utc).strftime(TS_FORMAT),
            "window_s": tier.window_s,
            "grid_s": tier.grid_s,
            "unit": "MT / min" if mode == tk.RATE else "MT",
            "labels": [datetime.fromtimestamp(s, tz=timezone.utc).strftime(label_format)
                       for s in starts],
            "series": tk.assemble(per_series, node_info),
        }
