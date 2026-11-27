"""Token charts: counter deltas, rates, running totals, and their rollups."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from oumigo.service.manager.dashboard import tokens as tk
from oumigo.service.manager.dashboard.rollup import RollupStore

T0 = int(datetime(2026, 11, 27, 0, 0, tzinfo=timezone.utc).timestamp())  # a 2 h boundary
GEN, INP = tk.KINDS["generated"], tk.KINDS["input"]


def _ts(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _counter(node: str, metric: str, start: int, values: list[float]) -> list[dict]:
    return [{"node_id": node, "metric": metric, "timestamp": _ts(start + 5 * i), "value": v}
            for i, v in enumerate(values)]


def _at(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _by_label(out: dict) -> dict[str, list]:
    return {s["label"]: s["data"] for s in out["series"]}


def test_delta_treats_a_drop_as_a_restart() -> None:
    assert tk.delta(None, 500) == 0          # the first sample only sets the baseline
    assert tk.delta(1000, 1600) == 600
    assert tk.delta(1600, 40) == 40          # vLLM restarted: 40 since the restart


def test_live_rate_is_mt_per_minute_with_the_moving_average() -> None:
    # 600 generated tokens every 5 s = 7200 / min = 0.0072 MT / min.
    rows = _counter("A", GEN, T0, [600.0 * i for i in range(30)])
    out = tk.live_series(rows, _at(T0 + 145), tk.RATE, {"A": {"seq": 1, "name": "Worker#1"}})
    data = _by_label(out)["Worker#1 · generated"]
    assert out["unit"] == "MT / min" and len(out["labels"]) == 720
    assert data[-1] == pytest.approx(0.0072)
    assert _by_label(out)["All workers · generated"][-1] == pytest.approx(0.0072)


def test_live_cumulative_runs_from_the_left_edge_across_a_restart() -> None:
    values = [1000.0, 1600.0, 2200.0, 100.0, 400.0]  # vLLM restarted before the 4th sample
    rows = _counter("A", INP, T0, values)
    out = tk.live_series(rows, _at(T0 + 20), tk.CUMULATIVE)
    data = [v for v in _by_label(out)["A · input"] if v is not None]
    # 600 + 600 + 100 (since the restart) + 300 = 1600 tokens
    assert data == pytest.approx([0.0006, 0.0012, 0.0013, 0.0016])
    assert out["unit"] == "MT"


def test_the_total_sums_the_workers_present() -> None:
    rows = (_counter("A", GEN, T0, [0, 1000, 2000]) + _counter("B", GEN, T0, [0, 500, 1000]))
    out = tk.live_series(rows, _at(T0 + 10), tk.CUMULATIVE)
    labels = _by_label(out)
    assert labels["All workers · generated"][-1] == pytest.approx(0.003)
    assert set(labels) == {"A · generated", "B · generated", "All workers · generated"}


def test_rollup_buckets_count_each_sample_once_and_survive_a_restart(tmp_path) -> None:
    rows = _counter("A", GEN, T0, [6000.0 * i for i in range(60)])  # 5 min, 354 000 tokens
    store = RollupStore.at(tmp_path)
    store.ingest(rows[:30], now=_at(T0 + 299))
    store.ingest(rows, now=_at(T0 + 299))             # overlap re-pull
    reopened = RollupStore.at(tmp_path)                # console restart, mirror re-seed
    reopened.ingest(rows, now=_at(T0 + 299))
    later = _at(T0 + 300 + 299)                        # the next bucket is current now
    cum = _by_label(reopened.token_series("48h", later, tk.CUMULATIVE))["A · generated"]
    assert cum[-2] == pytest.approx(0.354)
    rate = _by_label(reopened.token_series("48h", later, tk.RATE))["A · generated"]
    assert rate[-2] == pytest.approx(0.354 / 5)        # a full 5 min bucket: MT / min


def test_rollup_rate_of_the_current_bucket_uses_the_time_elapsed() -> None:
    store = RollupStore()
    store.ingest(_counter("A", INP, T0, [0.0, 60_000.0]), now=_at(T0 + 5))
    now = _at(T0 + 60)                                  # 60 s into the 2 h bucket
    rate = _by_label(store.token_series("45d", now, tk.RATE))["A · input"]
    assert rate[-1] == pytest.approx(0.06)              # 60 000 tokens in 1 minute
    assert len(rate) == 540
