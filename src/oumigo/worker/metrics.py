"""Worker-side metrics collection: grid-aligned sampling + buffered reporting.

Two background threads owned by the coordinator (the coordinator is a synchronous
thread-driven loop, so the collector matches that model rather than asyncio):

- **Sampler** wakes on an exact wall-clock grid (every ``grid_s`` seconds, aligned
  to ``:00, :05, :10, …`` UTC), reads the metric set, and appends one row per
  metric to an in-memory buffer. Each row is a data point of
  ``(node_id, timestamp, metric, value)`` — timestamp stamped with the *grid slot*
  (``YYYY-MM-DD HH:MM:SS`` UTC), not the actual read time.
- **Reporter** wakes every ``report_s`` seconds and ships the whole buffer to the
  manager's ``POST /metrics``. On success the sent rows are gone (the buffer was
  drained). On failure the rows are restored so the next cycle retries — the buffer
  keeps up to ``capacity_s`` of history (default 30 min), dropping the oldest
  ``evict_chunk_s`` span (default 5 min) whenever a new sample would overflow it.

Three metric sources, each degrading to *nothing* (a gap) when unavailable:

- **host** (`worker:*`) — stdlib ``/proc`` reads, always on.
- **GPU** (`gpu:#N_*`) — ``pynvml`` when importable, else an ``nvidia-smi``
  subprocess, else empty (CPU-only node).
- **vLLM** (`vllm:*`) — scraped from the local vLLM ``/metrics`` endpoint; empty
  until vLLM is serving.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from oumigo.protocol.messages import MetricPoint, MetricsReport
from oumigo.worker import client

log = logging.getLogger("oumigo.worker.metrics")

# --- metric names (worker: domain — see docs/metrics.md naming protocol) --------

M_CPU_CORES = "worker:cpu_cores"
M_CPU_UTIL = "worker:cpu_util_pct"
M_MEM_TOTAL = "worker:mem_total_bytes"
M_MEM_USED = "worker:mem_used_bytes"
M_MEM_USED_PCT = "worker:mem_used_pct"

TS_FORMAT = "%Y-%m-%d %H:%M:%S"  # UTC, grid-aligned


def grid_timestamp(grid_epoch: int) -> str:
    """Format a grid-slot epoch as the wire timestamp (UTC, second resolution)."""
    return datetime.fromtimestamp(grid_epoch, tz=timezone.utc).strftime(TS_FORMAT)


# --- host metric readers (pure parsers + thin /proc wrappers) -------------------


def _read(path: str) -> str:
    with open(path, encoding="ascii") as f:
        return f.read()


def _parse_cpu_times(stat_text: str) -> tuple[int, int]:
    """Return (idle_jiffies, total_jiffies) from the aggregate `cpu` line of /proc/stat."""
    first = stat_text.splitlines()[0]
    vals = [int(x) for x in first.split()[1:]]  # drop the leading 'cpu' label
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    return idle, sum(vals)


def _cpu_util_pct(prev: tuple[int, int], cur: tuple[int, int]) -> float:
    """Busy % over the interval between two /proc/stat snapshots."""
    didle = cur[0] - prev[0]
    dtotal = cur[1] - prev[1]
    if dtotal <= 0:
        return 0.0
    return round(100.0 * (1.0 - didle / dtotal), 2)


def _parse_meminfo(text: str) -> tuple[int, int, float]:
    """Return (total_bytes, used_bytes, used_pct) from /proc/meminfo (kB -> bytes)."""
    info: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            try:
                info[key] = int(parts[0]) * 1024  # values are in kB
            except ValueError:
                continue
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", info.get("MemFree", 0))
    used = max(0, total - avail)
    pct = round(100.0 * used / total, 2) if total else 0.0
    return total, used, pct


class _CpuSampler:
    """Holds the previous /proc/stat snapshot so utilization is a rate, not a level."""

    def __init__(self) -> None:
        self._prev: tuple[int, int] | None = None

    def prime(self) -> None:
        """Take a baseline read so the first grid sample already has a delta."""
        try:
            self._prev = _parse_cpu_times(_read("/proc/stat"))
        except OSError:
            self._prev = None

    def utilization(self) -> float | None:
        """Busy % since the last call, or None if unreadable / no baseline yet."""
        try:
            cur = _parse_cpu_times(_read("/proc/stat"))
        except OSError:
            return None
        prev, self._prev = self._prev, cur
        return None if prev is None else _cpu_util_pct(prev, cur)


def collect_host_metrics(cpu: _CpuSampler) -> dict[str, float]:
    """Sample the always-on host metrics. Missing readings are omitted (become gaps)."""
    out: dict[str, float] = {}

    cores = os.cpu_count()
    if cores:
        out[M_CPU_CORES] = float(cores)

    util = cpu.utilization()
    if util is not None:
        out[M_CPU_UTIL] = util

    try:
        total, used, pct = _parse_meminfo(_read("/proc/meminfo"))
    except OSError:
        pass
    else:
        out[M_MEM_TOTAL] = float(total)
        out[M_MEM_USED] = float(used)
        out[M_MEM_USED_PCT] = pct

    return out


# --- GPU metrics (pynvml, nvidia-smi fallback) ----------------------------------


def _gpu_emit(
    out: dict[str, float],
    index: int,
    *,
    util: float | None = None,
    vram_used: float | None = None,
    vram_total: float | None = None,
    temp: float | None = None,
    power: float | None = None,
) -> None:
    """Write the per-GPU keys (`gpu:#N_*`), skipping any reading that came back None."""
    p = f"gpu:#{index}_"
    if util is not None:
        out[p + "util_pct"] = float(util)
    if vram_used is not None:
        out[p + "vram_used_bytes"] = float(vram_used)
    if vram_total is not None:
        out[p + "vram_total_bytes"] = float(vram_total)
    if vram_used is not None and vram_total:
        out[p + "vram_used_pct"] = round(100.0 * vram_used / vram_total, 2)
    if temp is not None:
        out[p + "temp_c"] = float(temp)
    if power is not None:
        out[p + "power_w"] = round(float(power), 2)


class _GpuSampler:
    """Reads per-GPU utilization/VRAM/temp/power. NVML preferred, nvidia-smi fallback.

    ``start`` picks a backend once (NVML init is not free); ``sample`` is called on
    the grid; ``close`` releases NVML. A node with no GPU / no tooling stays silent.
    """

    def __init__(self) -> None:
        self._mode: str | None = None
        self._nvml = None
        self._handles: list = []

    def start(self) -> None:
        try:
            import pynvml  # type: ignore[import-untyped]  # optional, GPU-only dep, no stubs
        except ImportError:
            pynvml = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(pynvml.nvmlDeviceGetCount())
                ]
                self._nvml = pynvml
                self._mode = "pynvml"
                log.info("GPU metrics via NVML (%d device(s))", len(self._handles))
                return
            except Exception as exc:  # noqa: BLE001 - NVML absent/unusable; try the fallback
                log.debug("NVML unavailable (%s); trying nvidia-smi", exc)
        if shutil.which("nvidia-smi"):
            self._mode = "smi"
            log.info("GPU metrics via nvidia-smi subprocess")
        else:
            log.info("no GPU telemetry available (no NVML, no nvidia-smi)")

    def close(self) -> None:
        if self._mode == "pynvml" and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - shutdown is best-effort
                pass

    def sample(self) -> dict[str, float]:
        if self._mode == "pynvml":
            return self._sample_nvml()
        if self._mode == "smi":
            return self._sample_smi()
        return {}

    def _sample_nvml(self) -> dict[str, float]:
        out: dict[str, float] = {}
        n = self._nvml
        assert n is not None
        for i, h in enumerate(self._handles):
            util = vram_used = vram_total = temp = power = None
            try:
                util = n.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:  # noqa: BLE001 - a single unreadable metric must not sink the rest
                pass
            try:
                mem = n.nvmlDeviceGetMemoryInfo(h)
                vram_used, vram_total = mem.used, mem.total
            except Exception:  # noqa: BLE001
                pass
            try:
                temp = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
            except Exception:  # noqa: BLE001
                pass
            try:
                power = n.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
            except Exception:  # noqa: BLE001
                pass
            _gpu_emit(out, i, util=util, vram_used=vram_used, vram_total=vram_total,
                      temp=temp, power=power)
        return out

    def _sample_smi(self) -> dict[str, float]:
        query = "index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        return _parse_smi_csv(proc.stdout)


def _smi_float(token: str) -> float | None:
    """nvidia-smi emits 'N/A' / '[Not Supported]' for missing readings -> None."""
    try:
        return float(token.strip())
    except ValueError:
        return None


def _parse_smi_csv(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != 6:
            continue
        idx = _smi_float(cells[0])
        if idx is None:
            continue
        used_mib = _smi_float(cells[2])
        total_mib = _smi_float(cells[3])
        _gpu_emit(
            out,
            int(idx),
            util=_smi_float(cells[1]),
            vram_used=used_mib * 1024 * 1024 if used_mib is not None else None,
            vram_total=total_mib * 1024 * 1024 if total_mib is not None else None,
            temp=_smi_float(cells[4]),
            power=_smi_float(cells[5]),
        )
    return out


# --- vLLM metrics (scrape the local /metrics, parse Prometheus text) -------------

_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
# Constant / high-cardinality labels folded away: the fleet is one model, so
# `model_name` is noise; `le` marks histogram buckets we don't store.
_NOISE_LABELS = frozenset({"model_name", "model", "engine"})


def _sanitize(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()


def _parse_prometheus(text: str, prefix: str = "vllm:") -> dict[str, float]:
    """Flatten Prometheus text into ``{storage_key: value}`` for `prefix` metrics.

    Histograms keep only their `_sum`/`_count` series (buckets dropped); `_info`
    gauges are skipped; remaining non-noise labels are appended to the key so
    label-split series (e.g. `request_success_total` by finish reason) stay distinct.
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, _, rest = line.partition("{")
            labels_part, _, tail = rest.partition("}")
        else:
            name, _, tail = line.partition(" ")
            labels_part = ""
        name = name.strip()
        if not name.startswith(prefix) or name.endswith("_bucket") or name.endswith("_info"):
            continue
        tokens = tail.split()
        if not tokens:
            continue
        try:
            value = float(tokens[0])
        except ValueError:
            continue
        labels = {k: v for k, v in _LABEL_RE.findall(labels_part) if k not in _NOISE_LABELS}
        if labels:
            suffix = "_".join(_sanitize(labels[k]) for k in sorted(labels))
            key = f"{name}_{suffix}"
        else:
            key = name
        out[key] = value  # model_name collapse => last wins (single-model fleet)
    return out


def collect_vllm_metrics(vllm_url: str | None, timeout: float = 2.0) -> dict[str, float]:
    """Scrape and flatten vLLM's `/metrics`. Empty while vLLM is down / unreachable."""
    if not vllm_url:
        return {}
    try:
        resp = httpx.get(f"{vllm_url.rstrip('/')}/metrics", timeout=timeout)
    except httpx.HTTPError:
        return {}
    if resp.status_code != 200:
        return {}
    return _parse_prometheus(resp.text)


# --- buffer ---------------------------------------------------------------------


@dataclass(frozen=True)
class MetricRow:
    """One buffered data point: node UUID + grid timestamp + metric name + value."""

    node_id: str
    grid_epoch: int          # integer epoch of the grid slot; drives retention math
    timestamp: str           # 'YYYY-MM-DD HH:MM:SS' (UTC) — the reported form
    metric: str
    value: float


class MetricsBuffer:
    """Thread-safe, time-bounded store of metric rows (ascending by grid slot).

    Rows stay until a successful report drains them. If reporting keeps failing the
    buffer grows to at most ``capacity_s`` of wall-clock span; a new sample that
    would overflow that triggers eviction of the oldest ``evict_chunk_s`` span.
    """

    def __init__(self, capacity_s: float = 1800.0, evict_chunk_s: float = 300.0) -> None:
        self.capacity_s = capacity_s
        self.evict_chunk_s = evict_chunk_s
        self._rows: list[MetricRow] = []
        self._lock = threading.Lock()

    def append(self, rows: list[MetricRow]) -> None:
        """Add the newest grid slot's rows, evicting the oldest span if we overflow."""
        with self._lock:
            self._rows.extend(rows)
            self._evict_locked()

    def restore(self, rows: list[MetricRow]) -> None:
        """Put a failed batch back at the front (it predates anything sampled since)."""
        with self._lock:
            self._rows[:0] = rows
            self._evict_locked()

    def drain(self) -> list[MetricRow]:
        """Atomically take everything for a report attempt, leaving the buffer empty."""
        with self._lock:
            rows, self._rows = self._rows, []
            return rows

    def snapshot(self) -> list[MetricRow]:
        with self._lock:
            return list(self._rows)

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    def _evict_locked(self) -> None:
        # Rows are appended in grid order, so index 0 is the oldest, [-1] the newest.
        while self._rows and self._rows[-1].grid_epoch - self._rows[0].grid_epoch >= self.capacity_s:
            cutoff = self._rows[0].grid_epoch + self.evict_chunk_s
            drop = 0
            while drop < len(self._rows) and self._rows[drop].grid_epoch < cutoff:
                drop += 1
            if drop == 0:  # single slot wider than the chunk; nothing to trim, bail
                break
            log.warning(
                "metrics buffer at capacity (%.0fs); dropping oldest %d rows (%.0fs span)",
                self.capacity_s,
                drop,
                self.evict_chunk_s,
            )
            del self._rows[:drop]


# --- collector (owns the two threads) -------------------------------------------

# Report sender seam: (manager_url, report, token) -> None. Injectable for tests.
SendFn = Callable[[str, MetricsReport, "str | None"], None]


def _default_send(manager_url: str, report: MetricsReport, token: str | None) -> None:
    client.send_metrics(manager_url, report, token)


class MetricsCollector:
    """Runs the sampler + reporter threads for one worker node."""

    def __init__(
        self,
        manager_url: str,
        token: str | None,
        node_id: str,
        *,
        grid_s: float = 5.0,
        report_s: float = 30.0,
        capacity_s: float = 1800.0,
        evict_chunk_s: float = 300.0,
        vllm_url: str | None = None,
        send: SendFn = _default_send,
    ) -> None:
        self.manager_url = manager_url
        self.token = token
        self.node_id = node_id
        self.grid_s = grid_s
        self.report_s = report_s
        self.vllm_url = vllm_url
        self._send = send
        self._buffer = MetricsBuffer(capacity_s, evict_chunk_s)
        self._cpu = _CpuSampler()
        self._gpu = _GpuSampler()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._cpu.prime()
        self._gpu.start()
        self._threads = [
            threading.Thread(target=self._sample_loop, name="oumigo-metrics-sampler", daemon=True),
            threading.Thread(target=self._report_loop, name="oumigo-metrics-reporter", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self, flush: bool = True) -> None:
        """Signal both threads, join them, then best-effort ship whatever remains."""
        self._stop.set()
        for t in self._threads:
            t.join(timeout=self.grid_s + 2.0)
        self._threads = []
        self._gpu.close()
        if flush:
            self._flush_once()

    # --- sampler -------------------------------------------------------------

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            target = math.floor(now / self.grid_s) * self.grid_s + self.grid_s
            if self._stop.wait(target - now):  # interruptible sleep to the grid boundary
                return
            self._sample_at(int(target))

    def _sample_at(self, grid_epoch: int) -> None:
        ts = grid_timestamp(grid_epoch)
        metrics = collect_host_metrics(self._cpu)
        metrics.update(self._gpu.sample())                    # gpu:#N_* (empty if no GPU)
        metrics.update(collect_vllm_metrics(self.vllm_url))   # vllm:*   (empty until serving)
        rows = [
            MetricRow(self.node_id, grid_epoch, ts, name, value)
            for name, value in metrics.items()
        ]
        if rows:
            self._buffer.append(rows)

    # --- reporter ------------------------------------------------------------

    def _report_loop(self) -> None:
        while not self._stop.wait(self.report_s):  # first report after one interval
            self._flush_once()

    def _flush_once(self) -> None:
        batch = self._buffer.drain()
        if not batch:
            return
        report = MetricsReport(
            node_id=self.node_id,
            points=[MetricPoint(timestamp=r.timestamp, metric=r.metric, value=r.value) for r in batch],
        )
        try:
            self._send(self.manager_url, report, self.token)
        except Exception as exc:  # noqa: BLE001 - never let a report failure kill the thread
            log.warning("metrics report failed (%s); keeping %d points buffered", exc, len(batch))
            self._buffer.restore(batch)
        else:
            log.debug("metrics: reported %d points", len(batch))
