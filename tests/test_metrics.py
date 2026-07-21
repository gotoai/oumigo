"""Unit tests for worker metrics: pure parsers, buffer retention, and reporting."""

from __future__ import annotations

import pytest

from oumigo.protocol.messages import MetricsReport
from oumigo.worker.metrics import (
    M_CPU_UTIL,
    M_MEM_TOTAL,
    M_MEM_USED,
    M_MEM_USED_PCT,
    MetricRow,
    MetricsBuffer,
    MetricsCollector,
    _cpu_util_pct,
    _histogram_aggregates,
    _parse_cpu_times,
    _parse_meminfo,
    _parse_prometheus,
    _parse_smi_csv,
    collect_vllm_metrics,
    grid_timestamp,
)

# --- parsers --------------------------------------------------------------------


def test_parse_cpu_times_sums_and_idle() -> None:
    # cpu  user nice system idle iowait irq softirq steal ...
    idle, total = _parse_cpu_times("cpu  100 0 50 800 40 0 10 0\n")
    assert idle == 800 + 40  # idle + iowait
    assert total == 100 + 0 + 50 + 800 + 40 + 0 + 10 + 0


def test_cpu_util_pct_is_a_rate_between_snapshots() -> None:
    # 100 total jiffies elapsed, 25 of them idle -> 75% busy.
    assert _cpu_util_pct((900, 1000), (925, 1100)) == 75.0
    # No elapsed time -> guard against div-by-zero.
    assert _cpu_util_pct((900, 1000), (900, 1000)) == 0.0


def test_parse_meminfo_bytes_and_pct() -> None:
    text = "MemTotal:       1000 kB\nMemAvailable:    250 kB\nMemFree:  100 kB\n"
    total, used, pct = _parse_meminfo(text)
    assert total == 1000 * 1024
    assert used == 750 * 1024  # total - available
    assert pct == 75.0


def test_grid_timestamp_is_utc_second_resolution() -> None:
    # 2021-01-01 00:00:05 UTC
    assert grid_timestamp(1609459205) == "2021-01-01 00:00:05"


# --- buffer ---------------------------------------------------------------------


def _rows(epoch: int, n: int = 1) -> list[MetricRow]:
    return [MetricRow("uuid", epoch, grid_timestamp(epoch), f"m{i}", float(i)) for i in range(n)]


def test_buffer_drain_empties_and_returns_all() -> None:
    buf = MetricsBuffer()
    buf.append(_rows(100, n=3))
    drained = buf.drain()
    assert len(drained) == 3
    assert len(buf) == 0


def test_buffer_restore_prepends_older_batch() -> None:
    buf = MetricsBuffer()
    buf.append(_rows(200))       # a sample arrives during a failed report
    buf.restore(_rows(100))      # the failed (older) batch comes back
    snap = buf.snapshot()
    assert [r.grid_epoch for r in snap] == [100, 200]  # ascending order preserved


def test_buffer_evicts_oldest_5min_chunk_on_overflow() -> None:
    # capacity 30 min, evict 5 min chunks; one row per 5 min slot.
    buf = MetricsBuffer(capacity_s=1800, evict_chunk_s=300)
    for k in range(7):  # slots at 0,300,...,1800 -> span 1800 hits capacity
        buf.append(_rows(k * 300))
    epochs = [r.grid_epoch for r in buf.snapshot()]
    # The oldest 5-min chunk (epoch 0) is dropped so span stays under capacity.
    assert 0 not in epochs
    assert epochs[0] == 300
    assert buf.snapshot()[-1].grid_epoch - buf.snapshot()[0].grid_epoch < 1800


# --- reporter -------------------------------------------------------------------


def test_flush_success_clears_buffer() -> None:
    sent: list[MetricsReport] = []
    col = MetricsCollector("http://m", None, "uuid", send=lambda url, rep, tok: sent.append(rep))
    col._buffer.append(_rows(100, n=2))
    col._flush_once()
    assert len(sent) == 1
    assert sent[0].node_id == "uuid"
    assert len(sent[0].points) == 2
    assert len(col._buffer) == 0


def test_flush_failure_keeps_buffer_for_retry() -> None:
    def boom(url: str, rep: MetricsReport, tok: str | None) -> None:
        raise RuntimeError("manager down")

    col = MetricsCollector("http://m", None, "uuid", send=boom)
    col._buffer.append(_rows(100, n=2))
    col._flush_once()
    assert len(col._buffer) == 2  # restored, not lost


# --- GPU (nvidia-smi CSV parser) ------------------------------------------------


def test_parse_smi_csv_two_gpus() -> None:
    # index, util, mem.used(MiB), mem.total(MiB), temp, power.draw
    csv = "0, 37, 500, 16384, 53, 24.9\n1, 0, 100, 16384, 40, [Not Supported]\n"
    out = _parse_smi_csv(csv)
    assert out["gpu:#0_util_pct"] == 37.0
    assert out["gpu:#0_vram_used_bytes"] == 500 * 1024 * 1024
    assert out["gpu:#0_vram_total_bytes"] == 16384 * 1024 * 1024
    assert out["gpu:#0_vram_used_pct"] == round(100 * 500 / 16384, 2)
    assert out["gpu:#0_temp_c"] == 53.0
    assert out["gpu:#0_power_w"] == 24.9
    # Second GPU: unsupported power reading is dropped, the rest still present.
    assert "gpu:#1_power_w" not in out
    assert out["gpu:#1_util_pct"] == 0.0


# --- vLLM (Prometheus text parser) ----------------------------------------------

_VLLM_SAMPLE = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="acme/x"} 3.0
vllm:num_requests_waiting{model_name="acme/x"} 1.0
vllm:kv_cache_usage_perc{model_name="acme/x"} 0.42
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{model_name="acme/x"} 12345.0
vllm:request_success_total{model_name="acme/x",finished_reason="stop"} 10.0
vllm:request_success_total{model_name="acme/x",finished_reason="length"} 2.0
# TYPE vllm:time_to_first_token_seconds histogram   (SELECTED -> summarized)
vllm:time_to_first_token_seconds_bucket{model_name="acme/x",le="0.01"} 0.0
vllm:time_to_first_token_seconds_bucket{model_name="acme/x",le="0.05"} 2.0
vllm:time_to_first_token_seconds_bucket{model_name="acme/x",le="0.1"} 6.0
vllm:time_to_first_token_seconds_bucket{model_name="acme/x",le="0.5"} 9.0
vllm:time_to_first_token_seconds_bucket{model_name="acme/x",le="1.0"} 10.0
vllm:time_to_first_token_seconds_bucket{model_name="acme/x",le="+Inf"} 10.0
vllm:time_to_first_token_seconds_sum{model_name="acme/x"} 2.5
vllm:time_to_first_token_seconds_count{model_name="acme/x"} 10.0
vllm:time_to_first_token_seconds_created{model_name="acme/x"} 1784520000.0
# TYPE vllm:inter_token_latency_seconds histogram   (NOT selected -> dropped)
vllm:inter_token_latency_seconds_bucket{model_name="acme/x",le="0.1"} 3.0
vllm:inter_token_latency_seconds_bucket{model_name="acme/x",le="+Inf"} 4.0
vllm:inter_token_latency_seconds_sum{model_name="acme/x"} 0.3
vllm:inter_token_latency_seconds_count{model_name="acme/x"} 4.0
vllm:engine_sleep_state{model_name="acme/x",state="awake"} 1.0
vllm:engine_sleep_state{model_name="acme/x",state="discard_all"} 0.0
vllm:estimated_flops_per_gpu_total{model_name="acme/x"} 1.2e9
vllm:cache_config_info{block_size="16"} 1.0
"""


def test_parse_prometheus_gauges_counters_labels_histograms() -> None:
    out = _parse_prometheus(_VLLM_SAMPLE)
    # gauges (model_name label folded away)
    assert out["vllm:num_requests_running"] == 3.0
    assert out["vllm:kv_cache_usage_perc"] == 0.42
    # counter total kept as-is
    assert out["vllm:prompt_tokens_total"] == 12345.0
    # label-split counter -> one key per finish reason
    assert out["vllm:request_success_total_stop"] == 10.0
    assert out["vllm:request_success_total_length"] == 2.0
    # selected histogram -> transformed into a distribution summary (no raw sum/buckets)
    ttft = "vllm:time_to_first_token_seconds"
    assert out[f"{ttft}_count"] == 10.0
    assert out[f"{ttft}_mean"] == 0.25
    assert out[f"{ttft}_min"] == 0.05
    assert out[f"{ttft}_max"] == 1.0
    assert out[f"{ttft}_tile25"] == pytest.approx(0.05625)
    assert out[f"{ttft}_median"] == pytest.approx(0.0875)
    assert out[f"{ttft}_tile75"] == pytest.approx(0.3)
    assert f"{ttft}_sum" not in out
    assert not any(k.endswith("_bucket") for k in out)
    # prometheus `_created` timestamps are dropped, not passed through as a metric
    assert not any(k.endswith("_created") for k in out)
    # non-selected histogram fully dropped
    assert not any(k.startswith("vllm:inter_token_latency_seconds") for k in out)
    # explicitly excluded from V1: engine_sleep_state + estimated_* dropped
    assert not any(k.startswith("vllm:engine_sleep_state") for k in out)
    assert not any(k.startswith("vllm:estimated_") for k in out)
    # _info gauge skipped
    assert not any(k.endswith("_info") for k in out)


def test_histogram_aggregates_five_number_summary() -> None:
    buckets = {"0.01": 0.0, "0.05": 2.0, "0.1": 6.0, "0.5": 9.0, "1.0": 10.0, "+Inf": 10.0}
    out = _histogram_aggregates("m", buckets, hsum=2.5, hcount=10.0)
    assert out["m_count"] == 10.0
    assert out["m_mean"] == 0.25
    assert out["m_min"] == 0.05  # first populated bucket edge
    assert out["m_max"] == 1.0  # last populated finite bucket edge
    assert out["m_tile25"] == pytest.approx(0.05625)
    assert out["m_median"] == pytest.approx(0.0875)
    assert out["m_tile75"] == pytest.approx(0.3)


def test_histogram_aggregates_no_observations_only_count() -> None:
    out = _histogram_aggregates("m", {"+Inf": 0.0}, hsum=0.0, hcount=0.0)
    assert out == {"m_count": 0.0}


def test_collect_vllm_metrics_no_url_is_empty() -> None:
    assert collect_vllm_metrics(None) == {}


def test_sample_emits_worker_start_and_vllm_start_on_serving() -> None:
    import time

    from oumigo.worker.metrics import M_VLLM_START, M_WORKER_START, MetricsCollector

    c = MetricsCollector(
        "http://m", None, "node-x", vllm_url=None, worker_start=1_784_520_000.0
    )
    c._sample_at(int(time.time() // 5 * 5))
    rows = {r.metric: r.value for r in c._buffer.snapshot()}
    assert rows[M_WORKER_START] == 1_784_520_000.0  # constant float epoch
    assert M_VLLM_START not in rows  # not SERVING yet -> absent

    c.mark_serving()
    c._sample_at(int(time.time() // 5 * 5) + 5)
    rows2 = {r.metric: r.value for r in c._buffer.snapshot()}
    assert M_VLLM_START in rows2 and rows2[M_VLLM_START] > 0

    c.clear_serving()  # simulate a crash leaving SERVING
    slot = int(time.time() // 5 * 5) + 10
    c._sample_at(slot)
    after_clear = {r.metric for r in c._buffer.snapshot() if r.grid_epoch == slot}
    assert M_WORKER_START in after_clear  # worker stamp is always present
    assert M_VLLM_START not in after_clear  # vllm stamp gone until it serves again


def test_request_rate_tracker_derives_per_minute_over_window() -> None:
    from oumigo.worker.metrics import _RequestRateTracker

    t = _RequestRateTracker()
    # No history yet, and while the window is shorter than WINDOW_MIN_S (60s): a gap.
    assert t.update(0, 100.0) is None
    assert t.update(55, 111.0) is None  # only 55s of history so far
    # At 60s the window qualifies: Δcount=120-100=20 over 60s -> 20/min.
    assert t.update(60, 120.0) == 20.0
    # At 120s the oldest retained reference is still t=0 (within the 180s max window):
    # Δcount=240-100=140 over 120s -> 70/min.
    assert t.update(120, 240.0) == 70.0


def test_request_rate_tracker_caps_window_at_180s() -> None:
    from oumigo.worker.metrics import _RequestRateTracker

    t = _RequestRateTracker()
    assert t.update(0, 0.0) is None
    # The t=0 sample has aged past WINDOW_MAX_S, so the reference is the next-oldest
    # retained sample (t=5), giving a ~180s window rather than 185s.
    t.update(5, 10.0)
    # 300 completions from t=5's count of 10 -> Δ=300 over (185-5)=180s -> 100/min.
    assert t.update(185, 310.0) == 100.0


def test_request_rate_tracker_skips_counter_reset() -> None:
    from oumigo.worker.metrics import _RequestRateTracker

    t = _RequestRateTracker()
    t.update(0, 500.0)
    # vLLM restarted: the cumulative counter dropped. A negative delta is a gap, not a
    # bogus negative rate.
    assert t.update(60, 3.0) is None


def test_request_rate_tracker_gap_when_counter_absent() -> None:
    from oumigo.worker.metrics import _RequestRateTracker

    t = _RequestRateTracker()
    t.update(0, 100.0)
    assert t.update(60, None) is None  # vLLM down / scrape miss -> no update, no row
    # A later present read still measures from the retained t=0 sample.
    assert t.update(120, 220.0) == 60.0


def test_collect_host_metrics_shape() -> None:
    # On the Linux CI host this reads real /proc; assert the always-on keys appear.
    from oumigo.worker.metrics import _CpuSampler, collect_host_metrics

    cpu = _CpuSampler()
    cpu.prime()
    metrics = collect_host_metrics(cpu)
    # mem is always present on Linux; cpu_util may be None on the very first read only
    # if /proc/stat was unreadable — priming above makes it available here.
    assert M_MEM_TOTAL in metrics and M_MEM_USED in metrics and M_MEM_USED_PCT in metrics
    assert metrics[M_MEM_TOTAL] > 0
    assert 0.0 <= metrics[M_MEM_USED_PCT] <= 100.0
    assert M_CPU_UTIL in metrics
