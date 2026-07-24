"""Unit tests for the manager's in-memory sqlite metrics store."""

from __future__ import annotations

from datetime import datetime, timezone

from oumigo.service.manager.control.store import MetricStore
from oumigo.protocol.messages import MetricPoint


def _pt(ts: str, metric: str, value: float) -> MetricPoint:
    return MetricPoint(timestamp=ts, metric=metric, value=value)


def test_ingest_and_latest_per_node() -> None:
    store = MetricStore()
    store.ingest("nodeA", [_pt("2026-07-19 00:00:05", "worker:cpu_util_pct", 10.0)])
    store.ingest("nodeA", [_pt("2026-07-19 00:00:10", "worker:cpu_util_pct", 20.0)])
    store.ingest("nodeB", [_pt("2026-07-19 00:00:05", "worker:cpu_util_pct", 5.0)])

    latest = store.latest_per_node()
    assert [r["node_id"] for r in latest] == ["nodeA", "nodeB"]  # ordered by node_id
    # nodeA's newest slot wins.
    assert latest[0]["timestamp"] == "2026-07-19 00:00:10"
    assert latest[0]["metrics"]["worker:cpu_util_pct"] == 20.0
    assert latest[1]["timestamp"] == "2026-07-19 00:00:05"


def test_ingest_is_idempotent_upsert() -> None:
    store = MetricStore()
    key = ("2026-07-19 00:00:05", "worker:cpu_util_pct")
    store.ingest("nodeA", [_pt(*key, 10.0)])
    store.ingest("nodeA", [_pt(*key, 42.0)])  # same (node, metric, ts) -> overwrite
    assert store.count() == 1
    assert store.latest_per_node()[0]["metrics"]["worker:cpu_util_pct"] == 42.0


def test_retention_prunes_older_than_24h() -> None:
    store = MetricStore()
    now = datetime(2026, 7, 20, 0, 0, 0, tzinfo=timezone.utc)
    # One point ~25 h old (should be pruned), one ~1 h old (should stay).
    store.ingest("nodeA", [_pt("2026-07-18 23:00:00", "worker:cpu_util_pct", 1.0)], now=now)
    store.ingest("nodeA", [_pt("2026-07-19 23:00:00", "worker:cpu_util_pct", 2.0)], now=now)
    assert store.count() == 1
    latest = store.latest_per_node()
    assert latest[0]["timestamp"] == "2026-07-19 23:00:00"


def test_latest_per_node_returns_all_metrics_of_the_slot() -> None:
    store = MetricStore()
    ts = "2026-07-19 00:00:05"
    store.ingest(
        "nodeA",
        [
            _pt(ts, "worker:cpu_util_pct", 42.0),
            _pt(ts, "worker:mem_used_bytes", 1000.0),
            _pt(ts, "worker:mem_used_pct", 3.0),
        ],
    )
    metrics = store.latest_per_node()[0]["metrics"]
    assert set(metrics) == {"worker:cpu_util_pct", "worker:mem_used_bytes", "worker:mem_used_pct"}


def test_empty_store_has_no_nodes() -> None:
    assert MetricStore().latest_per_node() == []
