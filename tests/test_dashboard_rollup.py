"""The console's durable GPU-utilization rollups (48 h / 10 d / 45 d charts)."""

from __future__ import annotations

from datetime import datetime, timezone

from oumigo.service.manager.dashboard.rollup import TIERS, RollupStore

T0 = int(datetime(2026, 11, 27, 0, 0, tzinfo=timezone.utc).timestamp())  # a 2 h boundary


def _ts(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _slots(node: str, start: int, values: list[float], gpus: int = 1) -> list[dict]:
    """One 5 s slot per value; with gpus=2 the second GPU reads value + 20."""
    rows = []
    for i, v in enumerate(values):
        for g in range(gpus):
            rows.append({"node_id": node, "metric": f"gpu:#{g}_util_pct",
                         "timestamp": _ts(start + 5 * i), "value": v + 20 * g})
    rows.append({"node_id": node, "metric": "worker:cpu_pct", "timestamp": _ts(start),
                 "value": 99.0})  # not a GPU utilization metric: ignored
    return rows


def _at(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def test_a_point_is_the_mean_of_its_bucket_across_gpus() -> None:
    store = RollupStore()
    # 5 min of samples: 0, 10, ..., the second GPU 20 higher -> slot means 10, 20, ...
    store.ingest(_slots("A", T0, [10.0 * (i % 7) for i in range(60)], gpus=2), now=_at(T0 + 299))
    out = store.series("48h", _at(T0 + 299))
    expected = round(sum(10.0 * (i % 7) + 10 for i in range(60)) / 60, 2)
    assert out["series"][0]["data"][-1] == expected   # the current bucket, filling
    assert out["series"][0]["data"][-2] is None        # nothing before
    assert len(out["labels"]) == 48 * 12 and out["grid_s"] == 300


def test_reingesting_overlapping_slots_does_not_count_twice(tmp_path) -> None:
    store = RollupStore.at(tmp_path)
    rows = _slots("A", T0, [50.0] * 24) + _slots("A", T0 + 120, [10.0] * 12)
    store.ingest(rows[:25], now=_at(T0 + 200))
    store.ingest(rows, now=_at(T0 + 200))   # the mirror's overlap re-pull
    reopened = RollupStore.at(tmp_path)     # a console restart re-seeds its 90 minutes
    assert reopened.ingest(rows, now=_at(T0 + 200)) == 0
    point = reopened.series("48h", _at(T0 + 200))["series"][0]["data"][-1]
    assert point == round((24 * 50 + 12 * 10) / 36, 2)


def test_each_tier_has_its_grid_and_window() -> None:
    store = RollupStore()
    store.ingest(_slots("A", T0, [40.0]), now=_at(T0 + 1))
    for key, buckets in (("48h", 576), ("10d", 480), ("45d", 540)):
        out = store.series(key, _at(T0 + 1))
        assert len(out["labels"]) == buckets and out["grid_s"] == TIERS[key].grid_s
        assert out["series"][0]["data"][-1] == 40.0
    assert store.series("45d", _at(T0 + 1))["labels"][-1] == "11-27 00:00"  # UTC


def test_old_buckets_leave_the_window_and_are_pruned() -> None:
    store = RollupStore()
    store.ingest(_slots("A", T0, [30.0]), now=_at(T0 + 1))
    later = T0 + 49 * 3600
    store.ingest(_slots("A", later, [60.0]), now=_at(later + 1))
    data = store.series("48h", _at(later + 1))["series"][0]["data"]
    assert data[-1] == 60.0 and 30.0 not in data            # 49 h ago: gone from 48 h
    assert 30.0 in store.series("10d", _at(later + 1))["series"][0]["data"]
    with store._lock:
        tiers = {t for (t,) in store._conn.execute(
            "SELECT DISTINCT tier FROM bucket WHERE start = ?", (T0,))}
    assert tiers == {"10d", "45d"}                          # pruned from the 48 h tier


def test_series_are_labeled_and_ordered_by_worker_number() -> None:
    store = RollupStore()
    store.ingest(_slots("node-b", T0, [10.0]) + _slots("node-a", T0, [20.0]) +
                 _slots("gone", T0, [30.0]), now=_at(T0 + 1))
    info = {"node-a": {"seq": 2, "name": "Worker#2"}, "node-b": {"seq": 1, "name": "Worker#1"}}
    labels = [s["label"] for s in store.series("10d", _at(T0 + 1), info)["series"]]
    assert labels == ["Worker#1", "Worker#2", "gone"]       # unknown nodes last, by id
