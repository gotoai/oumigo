"""Tests for the reporting plane's per-node GPU-utilization series (raw 5s grid)."""

from __future__ import annotations

from datetime import datetime, timezone

from oumigo.manager.dashboard.aggregate import _trailing_moving_average, gpu_util_series


def _row(node_id: str, metric: str, ts: str, value: float) -> dict:
    return {"node_id": node_id, "metric": metric, "timestamp": ts, "value": value}


def _gpu_rows(node_id: str, start_epoch: int, values: list[float]) -> list[dict]:
    """One gpu:#0_util_pct point per consecutive 5s slot from start_epoch."""
    rows = []
    for i, v in enumerate(values):
        ts = datetime.fromtimestamp(start_epoch + 5 * i, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        rows.append(_row(node_id, "gpu:#0_util_pct", ts, v))
    return rows


def test_spatial_mean_across_gpus_at_each_5s_slot() -> None:
    now = datetime(2026, 7, 20, 12, 0, 7, tzinfo=timezone.utc)  # grid-aligns to 12:00:05
    rows = [
        _row("A", "gpu:#0_util_pct", "2026-07-20 12:00:00", 10.0),
        _row("A", "gpu:#1_util_pct", "2026-07-20 12:00:00", 30.0),  # slot mean 20
        _row("A", "gpu:#0_util_pct", "2026-07-20 12:00:05", 40.0),
        _row("A", "gpu:#1_util_pct", "2026-07-20 12:00:05", 60.0),  # slot mean 50
    ]
    out = gpu_util_series(rows, now, window_s=15, grid_s=5)

    assert out["unit"] == "%" and out["grid_s"] == 5
    # window_s=15, grid_s=5 -> slots 11:59:55, 12:00:00, 12:00:05
    assert out["labels"] == ["11:59:55", "12:00:00", "12:00:05"]
    # only 2 present samples in any trailing window here (< MA_MIN_SAMPLES=4) -> gaps
    assert out["series"][0]["data"] == [None, None, None]


def test_series_values_are_the_trailing_moving_average() -> None:
    # 6 consecutive slots ending at the grid slot for `now`; window fills to 6.
    now = datetime(2026, 7, 20, 12, 0, 2, tzinfo=timezone.utc)  # grid slot 12:00:00
    start = int(datetime(2026, 7, 20, 11, 59, 35, tzinfo=timezone.utc).timestamp())
    rows = _gpu_rows("A", start, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    out = gpu_util_series(rows, now, window_s=30, grid_s=5)  # 6 slots
    data = out["series"][0]["data"]
    # slots 0..2 have <4 samples -> None; slot3=mean(0,10,20,30)=15; slot4=20; slot5=25
    assert data == [None, None, None, 15.0, 20.0, 25.0]


def test_missing_slots_are_gaps_not_zero() -> None:
    now = datetime(2026, 7, 20, 12, 0, 7, tzinfo=timezone.utc)
    rows = [_row("A", "gpu:#0_util_pct", "2026-07-20 12:00:00", 50.0)]
    out = gpu_util_series(rows, now, window_s=15, grid_s=5)
    assert out["series"][0]["data"] == [None, None, None]  # single sample < min window


def test_non_gpu_util_metrics_ignored() -> None:
    now = datetime(2026, 7, 20, 12, 0, 22, tzinfo=timezone.utc)
    # 4 present gpu slots so the MA emits a value; cpu/vram rows must not count.
    start = int(datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp())
    rows = _gpu_rows("A", start, [42.0, 42.0, 42.0, 42.0])
    rows += [
        _row("A", "worker:cpu_util_pct", "2026-07-20 12:00:15", 99.0),
        _row("A", "gpu:#0_vram_used_pct", "2026-07-20 12:00:15", 80.0),
    ]
    out = gpu_util_series(rows, now, window_s=30, grid_s=5)
    assert out["series"][0]["data"][-1] == 42.0  # only gpu:#0_util_pct contributed


def test_trailing_moving_average_window_and_min_samples() -> None:
    # window=6, min_samples=4: leading 3 are gaps, then trailing 6-slot mean.
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    ma = _trailing_moving_average(data)
    assert ma[0] is None and ma[1] is None and ma[2] is None
    assert ma[3] == 25.0  # mean(10,20,30,40)
    assert ma[5] == 35.0  # mean(10..60)
    assert ma[6] == 45.0  # mean(20..70) -> trailing 6 slots


def test_moving_average_counts_only_present_samples_in_window() -> None:
    # Gaps count against the sample total; a window needs >=4 present values.
    data = [None, None, None, 10.0, 20.0, 30.0, 40.0, None]
    ma = _trailing_moving_average(data)
    assert ma[5] is None  # window slots 0..5 -> present {10,20,30} = 3 (< 4)
    assert ma[6] == 25.0  # window slots 1..6 -> present {10,20,30,40} = 4
    assert ma[7] == 25.0  # window slots 2..7 -> present {10,20,30,40} (slot7 is a gap)


def test_default_window_is_60min_at_5s() -> None:
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    out = gpu_util_series([], now)
    assert len(out["labels"]) == 720  # 3600s / 5s


def test_labels_and_order_use_manager_worker_names() -> None:
    now = datetime(2026, 7, 20, 12, 0, 7, tzinfo=timezone.utc)
    rows = [
        _row("uuid-x", "gpu:#0_util_pct", "2026-07-20 12:00:05", 10.0),
        _row("uuid-y", "gpu:#0_util_pct", "2026-07-20 12:00:05", 20.0),
    ]
    node_info = {
        "uuid-x": {"seq": 2, "name": "Worker#2"},
        "uuid-y": {"seq": 1, "name": "Worker#1"},
    }
    out = gpu_util_series(rows, now, node_info=node_info, window_s=15, grid_s=5)
    # ordered by seq (Worker#1 first -> gets series color slot 1), labeled by name
    assert [s["label"] for s in out["series"]] == ["Worker#1", "Worker#2"]
    assert out["series"][0]["node_id"] == "uuid-y"


def test_unnamed_node_falls_back_to_short_uuid() -> None:
    now = datetime(2026, 7, 20, 12, 0, 7, tzinfo=timezone.utc)
    rows = [_row("abcdef0123456789", "gpu:#0_util_pct", "2026-07-20 12:00:05", 5.0)]
    out = gpu_util_series(rows, now, node_info={}, window_s=15, grid_s=5)
    assert out["series"][0]["label"] == "abcdef01"
