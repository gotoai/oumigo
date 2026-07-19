# Metrics & Dashboard — Data Model Design

> Status: design (pre-implementation). Captures the decisions from design
> discussion; implementation not started. vLLM functions are also not built yet,
> so vLLM-sourced metrics are a later phase.

## Purpose

Monitor worker-node health and performance from the manager: GPU utilization and
VRAM, node CPU and memory, error/restart counts, and (later) vLLM's own inference
metrics (throughput, latency, queue depth, KV-cache usage). Surface it in a
lightweight, built-in web dashboard.

## Goals / non-goals

**Goals**
- **Self-contained.** No external services or infra. Everything is Python
  standard library + GPU-core libraries (same family vLLM already needs) + one
  *vendored* JS chart asset. A vendored static file is not a runtime dependency.
- A simple **live** view (last minutes–hours) of the fleet.
- Cheap to run and reason about: low-frequency, low-cardinality data.

**Non-goals (v1)**
- No Prometheus/Grafana/TSDB dependency. (An optional read-only `/metrics`
  endpoint may be added later purely as an interop escape hatch — it adds no
  dependency — but it is not the product's monitoring path.)
- No long-term historian, no alerting engine, no downsampling.

## Why we store at all

vLLM's metrics API and `pynvml` return an **instantaneous snapshot** — current
gauges and monotonic counters. They keep no history. Two consequences:

1. **History does not exist upstream.** A chart is a series of samples we must
   accumulate ourselves.
2. **Rates need two samples.** vLLM exposes tokens/requests as counters;
   "tokens/sec" is `Δcounter / Δt` between consecutive samples, so we must retain
   at least the previous sample.

So the manager keeps its own time-series store. The good news: the data is tiny
(see below), so the "store" is a ring buffer, not a database.

## Two channels: heartbeat (real-time) vs metrics batch (historical)

Worker telemetry reaches the manager over **two deliberately separate channels**
with different latency and reliability semantics. Conflating them would either
bloat the heartbeat or delay liveness detection.

| | **Heartbeat** | **Metrics batch** |
|---|---|---|
| Cadence | every ~10 s | sample 5 s (grid), report 30 s |
| Buffering | **none** — sent directly | buffered locally, sent as a batch |
| Backfill / retry | **no** — a miss is a miss | yes — late batches backfill by grid slot |
| Latency | low (real-time) | delay-tolerant |
| Purpose | **liveness + real-time "now" monitor** | **historical time-series charts** |
| Loss semantics | missed beats → `LOST` (reaper) | missed batch → a gap in history |
| Endpoint | `POST /heartbeat` | `POST /metrics` |

- **The heartbeat is a special, unbuffered, real-time data point.** It is the
  low-latency channel: it drives the liveness reaper (`LOST`) and the dashboard's
  real-time "current status" view. It is **never buffered or backfilled** — a
  missed heartbeat simply counts toward the timeout — and it stays small.
- It **may carry a compact current snapshot** (current node `state`, optionally a
  few live gauges such as current GPU util) for the real-time indicator — kept
  minimal and distinct from the historical batch, which remains the source of
  truth for the time-series.
- **The metrics batch is delay-tolerant history.** Everything below about the
  grid, backfill, upsert, and storage applies to *this* channel only.

## Sizing (why this stays trivial)

Order-of-magnitude: ~10 nodes × ~20 metrics × 1 sample / 5 s. One hour at full
resolution ≈ 720 points/series → ~144k floats ≈ ~1 MB in memory. A full day at
5 s is ~17k points/series, which uPlot renders directly. This is
**low-frequency, low-cardinality**: no TSDB, no downsampling needed.

## Sampling protocol — grid-aligned

Workers sample on a **shared wall-clock grid** (e.g. every 5 s at `:00, :05, :10,
…`). Each sample is **stamped with its grid slot**, not the actual wall time of
the read (which lands a few ms late). The manager treats per-node timing jitter
as white noise and assumes samples are aligned to the promised grid. All series
are built on this grid.

Why grid alignment (beyond tidy timestamps): it gives every node a **shared time
axis**, which makes storage *columnar* (one integer grid-slot axis + parallel
value arrays) and makes multi-node rendering in uPlot trivial.

- **Mechanism:** sleep until the next grid boundary, sample, stamp with the grid
  slot, repeat.
- **Assumption:** node clocks are roughly NTP-synced (normal on cloud/LAN).
  Without NTP, nodes' grids drift apart in real time. Acceptable for utilization
  metrics; stated explicitly.
- **Gaps stay gaps.** A missed sample/slot is stored as nothing (a hole), never a
  carried-forward value. uPlot draws a break — which correctly reads as "no data
  here." (Liveness itself is decided on the separate real-time heartbeat channel,
  not inferred from metric gaps — see *Two channels*.)

## What is collected

All worker-side collection uses stdlib or GPU-core libraries:

| Metric group | Source | Notes |
|---|---|---|
| CPU util %, memory used/total | `/proc` (stdlib file reads) | zero dependency; Linux is mandated |
| GPU util %, VRAM used/total, temp, power | `pynvml` (NVML bindings) | GPU-core; `nvidia-smi` subprocess is a zero-Python-dep fallback |
| error count, restart count, uptime | the worker **coordinator** | it owns the vLLM restart policy, so it owns these counters |
| throughput, latency (TTFT/TPOT), queue, KV-cache | vLLM `/metrics` | **later phase**; "core vLLM stuff", not a third party |

Multi-GPU nodes: encode the GPU index in the metric name (e.g. `gpu.0.util_pct`,
`gpu.1.vram_used_bytes`) rather than adding a separate column — GPU is a
sub-dimension of the node.

## Transport — batched report, own cadence

- Sampling cadence (grid) and reporting cadence are **decoupled**: sample every
  5 s, **report every 30 s** as a batch (6 samples) to a manager HTTP endpoint
  (`POST /metrics`). This is the historical channel — the real-time heartbeat is
  the *other* channel (see *Two channels*).
- Auth: same shared bearer token as registration/heartbeat.

**Sample-batch payload (shape):**

```json
{
  "node_id": "…uuid…",
  "samples": [
    { "grid_ts": 1710000005,
      "metrics": { "cpu.util_pct": 42.1, "mem.used_bytes": 1234567,
                   "gpu.0.util_pct": 88, "gpu.0.vram_used_bytes": 987654,
                   "errors_total": 3, "restarts_total": 1 } },
    { "grid_ts": 1710000010, "metrics": { "…": 0 } }
  ]
}
```

`grid_ts` is integer epoch seconds aligned to the grid.

**Backfill instead of loss (unlocked by grid timestamps).** Because every sample
carries its grid slot, a *late* batch still lands in the correct slots. So on a
failed report the worker **keeps the buffer and retries** next cycle (bounded to a
cap). A transient network blip becomes *no data loss*; only a real outage (worker
down, or outage exceeding the cap) becomes a gap. This preserves "gaps stay gaps"
while recovering blips.

**Idempotent ingest.** The manager **upserts by `(node_id, metric, grid_ts)`**,
never appends. A retried/overlapping batch simply rewrites the same slots, so
backfill is safe.

## Storage — long/narrow, columnar on read

- **Model:** a single **long/narrow fact table** — `(grid_ts, node_id, metric,
  value)`. **Not** a star/snowflake schema: at this scale (essentially one
  dimension — the node) normalized dimensions add joins for no payoff. The few
  node attributes (GPU model, etc.) are denormalized or read from the registry.
- **Hot tier (v1):** in-memory, bounded. Either a ring buffer per
  `(node_id, metric)` (`deque(maxlen = window / grid)`) or an equivalent flat
  structure. Fixed memory, O(1) append, auto-evicts by age.
- **Rendering:** **pivot long → columnar on read** for uPlot (shared grid-slot x
  axis + one y array per node). Long-format is best for ingest/churn; columnar is
  best for the chart — convert at the boundary.
- **Retention / churn eviction:** prune **by time**. A stopped node emits no new
  rows and its old rows age out of the window on their own — churn self-heals. The
  only residual is GC'ing a node's now-empty container, done by the same retention
  sweep. No explicit per-node cleanup path needed.
- **Raw, derive on read:** store vLLM counters as-is; compute rates (tokens/s) at
  query time from adjacent samples, so we can re-derive over any window.

**Persistence (optional, later):** if history must survive a manager restart or
exceed the in-memory window, use **`sqlite3` (Python standard library)** — one
table `(grid_ts, node_id, metric, value)`, index on `(node_id, metric, grid_ts)`,
periodic `DELETE` for retention. Zero external deps; handles this volume trivially.
**Downsampling: skip** until retention reaches weeks/months.

## Concurrency — async tasks, not a process or a thread

Worker-side sampling and reporting run as **async tasks in the worker
coordinator** (making the coordinator `async`, matching the manager's model):
- **Not a separate process** — sampling `/proc` + `pynvml` is microseconds of
  work; a process only adds IPC for no parallelism/isolation benefit.
- **Not a raw thread** — the concurrency model is process-for-parallelism,
  async-single-thread within (see the manager control plane). NML/`/proc` reads
  don't meaningfully block the event loop.
- If vLLM supervision ever genuinely starves the sampler, split it into its own
  **process** (not a thread) — but do not pre-pay that.

Manager-side, ingest and the retention sweep are async tasks on the control-plane
event loop, like the heartbeat reaper.

## Visualization

- Served by the manager (async FastAPI): **one static HTML page + a JSON API**;
  the page **polls every few seconds** (SSE/WebSocket possible later — both built
  in — if push is wanted).
- **Vanilla JS, no framework/build step** (no React/Vue/npm) — a single static
  page keeps it compact and self-contained.
- **Chart engine: uPlot**, vendored and inlined (~40 KB, fastest/smallest for many
  live time-series). No CDN. (Chart.js is the simpler-API fallback; pure
  SVG/canvas is the zero-JS-dep purist option.)

## v1 scope

Two channels — **heartbeat (10 s, unbuffered, real-time/liveness)** and the
**metrics batch (historical)** — kept separate. Metrics: in-memory ring buffer
only · grid-aligned sampling (5 s) · batched report (30 s) with bounded
retry/backfill · manager upsert by `(node_id, metric, grid_ts)` · long/narrow
model, columnar on read · time-based retention · gaps as gaps · raw counters,
rates derived on read · uPlot dashboard via polled JSON. No sqlite, no
downsampling, no Prometheus.

## Deferred / open questions

- Deep data-model deep-dive (long vs columnar tradeoffs, exact table/index).
- `sqlite3` persistence tier + retention policy specifics.
- vLLM `/metrics` ingestion (waits on vLLM functions).
- Optional read-only Prometheus-format `/metrics` interop endpoint.
- Exact metric set, units, and per-GPU naming convention.
- Config surface: grid interval, report interval, retention window (mirrors the
  `heartbeat.*` config style).
