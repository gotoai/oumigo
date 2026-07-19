# Metrics & Dashboard — Data Model Design

> Status: **implemented** — worker collection (host + GPU + vLLM), the manager's
> in-memory sqlite ingest, and a console `metrics` command are all built. Captures
> the design decisions and reflects what exists. The uPlot dashboard is the main
> piece still outstanding (see *Deferred*).

## Purpose

Monitor worker-node health and performance from the manager: GPU utilization and
VRAM, node CPU and memory, error/restart counts, and (later) vLLM's own inference
metrics (throughput, latency, queue depth, KV-cache usage). Surface them in a
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

So both sides retain samples: the **worker** buffers them in memory between reports
(drain-on-success, retry-on-failure), and the **manager** keeps the time-series
store. The data is tiny (see below), so the manager store is a single **in-memory
sqlite** table, not a heavyweight TSDB.

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
axis**, which makes storage *columnar* (one shared grid-slot axis + parallel
value arrays) and makes multi-node rendering in uPlot trivial.

- **Mechanism:** sleep until the next grid boundary (`floor(now / grid) · grid +
  grid`), sample, and stamp with the grid slot — rendered on the wire as a
  `YYYY-MM-DD HH:MM:SS` **UTC** string. Repeat.
- **Assumption:** node clocks are roughly NTP-synced (normal on cloud/LAN).
  Without NTP, nodes' grids drift apart in real time. Acceptable for utilization
  metrics; stated explicitly.
- **Gaps stay gaps.** A missed sample/slot is stored as nothing (a hole), never a
  carried-forward value. uPlot draws a break — which correctly reads as "no data
  here." (Liveness itself is decided on the separate real-time heartbeat channel,
  not inferred from metric gaps — see *Two channels*.)

## What is collected

All worker-side collection uses stdlib, GPU-core libraries, or a local HTTP scrape:

| Metric group | Source | Notes |
|---|---|---|
| CPU util %, memory used/total | `/proc` (stdlib file reads) | zero dependency; Linux is mandated |
| GPU util %, VRAM used/total, temp, power | `pynvml` (NVML bindings) | GPU-core; `nvidia-smi` subprocess is a zero-Python-dep fallback |
| error count, restart count, uptime | the worker **coordinator** | it owns the vLLM restart policy, so it owns these counters |
| throughput, latency (TTFT/TPOT), queue, KV-cache | vLLM `/metrics` | scraped + flattened from the local vLLM endpoint; "core vLLM stuff", not a third party |

All metric keys follow a **`<domain>:<key>`** protocol — colon-separated, with a
snake_case key — matching vLLM's own `vllm:` convention. Domains: `worker:` (host +
coordinator counters), `gpu:` (per-GPU), `vllm:` (engine). Multi-GPU nodes encode
the GPU's sequence number in the key — prefixed with `#` — rather than adding a
separate column (e.g. `gpu:#0_util_pct`, `gpu:#1_vram_used_bytes`, where `#0` is the
first GPU) — GPU is a sub-dimension of the node.

## Metric catalog — the exact set

Three sources, all implemented: **host** (`worker:*`, stdlib `/proc`), **GPU**
(`gpu:#N_*`, NVML / nvidia-smi), and **vLLM engine** (`vllm:*`, scraped from the
local `/metrics`). Each degrades to nothing when unavailable — no GPU, or vLLM not
yet serving — so the set present at any grid slot reflects what the node can see.
Names below are the oumigo storage keys, in the `<domain>:<key>` protocol above;
per-GPU keys embed the index (e.g. `gpu:#0_util_pct`).

### Host / node metrics — always on

Source: `/proc` (stdlib file reads), zero dependency.

| oumigo metric | Type | Unit | Description |
|---|---|---|---|
| `worker:cpu_cores` | gauge | count | logical CPU cores on the node (constant; the denominator for util %) |
| `worker:cpu_util_pct` | gauge | 0–100 | CPU used % across all cores over the sample interval |
| `worker:mem_total_bytes` | gauge | bytes | total physical RAM |
| `worker:mem_used_bytes` | gauge | bytes | RAM in use |
| `worker:mem_used_pct` | gauge | 0–100 | `used / total × 100` |

Canonical storage is **bytes**; the dashboard renders **GB** (`bytes / 1e9`), so
the requested "Total GB / used GB / used %" are one conversion away without losing
precision in the store. `worker:cpu_util_pct` is a *rate*: `/proc/stat` reports
cumulative jiffies, so the worker computes % from two reads across the grid interval
— it is not an instantaneous read. `worker:mem_used_pct` is derivable but stored
directly for a zero-math live gauge.

### GPU metrics — per device, when present

Source: **NVML** (`pynvml`) when importable, else an **`nvidia-smi`** subprocess,
else silent (CPU-only node). `#N` is the GPU's sequence number; an individual
reading that a card doesn't support (e.g. power on some laptops) is skipped as a
gap, not fatal.

| oumigo metric | Type | Unit | Description |
|---|---|---|---|
| `gpu:#N_util_pct` | gauge | 0–100 | GPU utilization |
| `gpu:#N_vram_used_bytes` | gauge | bytes | VRAM in use |
| `gpu:#N_vram_total_bytes` | gauge | bytes | total VRAM |
| `gpu:#N_vram_used_pct` | gauge | 0–100 | `used / total × 100` (derived) |
| `gpu:#N_temp_c` | gauge | °C | core temperature |
| `gpu:#N_power_w` | gauge | watts | power draw |

### vLLM engine metrics — scraped from `/metrics`

The full V1 metric set vLLM exposes (authoritative source: `vllm/v1/metrics/
loggers.py`). Stored under the `vllm:` domain. Counters are emitted with a
`_total` suffix in the Prometheus text format; histograms expose `_bucket` /
`_sum` / `_count`. Every series carries a `model_name` label — constant for our
homogeneous fleet. Metrics marked **(situational)** only appear when that feature
is enabled (LoRA, multi-modal, speculative decode, external KV connector, KV-block
residency tracking).

**How the scraper flattens (as implemented):** it keeps `vllm:` series and forms
one storage key each — gauges and counter `…_total` pass through; histogram
`_bucket` lines are dropped and only `_sum` / `_count` kept; `_info` gauges are
skipped; the constant `model_name` label is folded away, and any remaining label's
value is appended to the key so splits stay distinct (e.g. a `finished_reason`
split → `vllm:request_success_total_stop`).

**Engine / scheduler state — Gauges**

| Metric | Description |
|---|---|
| `vllm:num_requests_running` | requests currently in execution batches |
| `vllm:num_requests_waiting` | requests queued awaiting scheduling |
| `vllm:kv_cache_usage_perc` | fraction of KV-cache blocks in use (0–1) |
| `vllm:num_requests_waiting_by_reason` | waiting requests broken out by reason *(situational)* |
| `vllm:engine_sleep_state` | engine awake/sleeping indicator *(situational)* |
| `vllm:lora_requests_info` | per-adapter running/waiting counts *(situational, info)* |
| `vllm:cache_config_info` | static cache-config info gauge *(info)* |

**Throughput & outcomes — Counters** (emitted `…_total`)

| Metric | Description |
|---|---|
| `vllm:prompt_tokens_total` | prefill tokens processed |
| `vllm:generation_tokens_total` | generation tokens produced |
| `vllm:num_preemptions_total` | cumulative scheduler preemptions |
| `vllm:request_success_total` | finished requests, labeled by `finished_reason` |
| `vllm:corrupted_requests_total` | corrupted requests *(situational)* |

**Cache effectiveness — Counters** *(situational)* (emitted `…_total`)

| Metric | Description |
|---|---|
| `vllm:prefix_cache_queries_total` | prefix-cache queried tokens |
| `vllm:prefix_cache_hits_total` | prefix-cache hit tokens |
| `vllm:prompt_tokens_cached_total` | cached prompt tokens (local + external) |
| `vllm:prompt_tokens_by_source_total` | prompt tokens by source |
| `vllm:external_prefix_cache_queries_total` | external (KV-connector) prefix-cache queries |
| `vllm:external_prefix_cache_hits_total` | external prefix-cache hits |
| `vllm:mm_cache_queries_total` | multi-modal cache queries |
| `vllm:mm_cache_hits_total` | multi-modal cache hits |

**Latency — Histograms** (seconds)

| Metric | Description |
|---|---|
| `vllm:time_to_first_token_seconds` | time to first token (TTFT) |
| `vllm:inter_token_latency_seconds` | inter-token latency |
| `vllm:request_time_per_output_token_seconds` | time per output token (TPOT), per request |
| `vllm:e2e_request_latency_seconds` | end-to-end request latency |
| `vllm:request_queue_time_seconds` | time in WAITING phase |
| `vllm:request_inference_time_seconds` | time in RUNNING phase |
| `vllm:request_prefill_time_seconds` | time in PREFILL phase |
| `vllm:request_decode_time_seconds` | time in DECODE phase |

**Request shape — Histograms**

| Metric | Description |
|---|---|
| `vllm:iteration_tokens_total` | tokens per engine step (a histogram despite the name) |
| `vllm:request_prompt_tokens` | input prompt token counts |
| `vllm:request_generation_tokens` | generation token counts |
| `vllm:request_max_num_generation_tokens` | max requested generation tokens |
| `vllm:request_params_n` | the `n` sampling parameter |
| `vllm:request_params_max_tokens` | the `max_tokens` parameter |
| `vllm:request_prefill_kv_computed_tokens` | new KV tokens computed during prefill *(situational)* |

**KV-block residency — Histograms** *(situational; seconds)*

| Metric | Description |
|---|---|
| `vllm:kv_block_lifetime_seconds` | block lifetime, allocation → eviction |
| `vllm:kv_block_idle_before_evict_seconds` | idle time before eviction |
| `vllm:kv_block_reuse_gap_seconds` | gap between consecutive block accesses |

**Feature-gated families** — present only when the corresponding feature/flag is
on; not part of the baseline fleet dashboard, listed for completeness:

| Metric | Type | Enabled by |
|---|---|---|
| `vllm:spec_decode_num_accepted_tokens_per_pos` | Counter | speculative decoding |
| `vllm:nixl_num_failed_notifications` | Counter | NIXL KV connector |
| `vllm:nixl_num_failed_transfers` | Counter | NIXL KV connector |
| `vllm:nixl_num_kv_expired_reqs` | Counter | NIXL KV connector |
| `vllm:nixl_bytes_transferred` | Histogram | NIXL KV connector |
| `vllm:nixl_num_descriptors` | Histogram | NIXL KV connector |
| `vllm:nixl_post_time_seconds` | Histogram | NIXL KV connector |
| `vllm:nixl_xfer_time_seconds` | Histogram | NIXL KV connector |
| `vllm:estimated_flops_per_gpu_total` | Counter | `--enable-mfu-metrics` |
| `vllm:estimated_read_bytes_per_gpu_total` | Counter | `--enable-mfu-metrics` |
| `vllm:estimated_write_bytes_per_gpu_total` | Counter | `--enable-mfu-metrics` |

**Deprecation / hidden metrics.** vLLM hides a deprecated metric one minor version
after deprecation; it can be re-enabled with
`--show-hidden-metrics-for-version=X.Y` and is removed one version later. The
scraper should tolerate a metric disappearing across a vLLM upgrade (store a gap,
per *gaps stay gaps*) rather than treating it as an error.

### Storing non-scalar metric types

The store is scalar — `(node_id, timestamp, metric, value)` — but vLLM emits counters
and histograms, so map each type in:

- **Gauges** → store the value directly.
- **Counters** → store the raw cumulative value; derive rates (tokens/s, req/s, hit
  ratio) on read from adjacent grid slots (matches *raw, derive on read*). A
  label-split counter such as `request_success_total{finished_reason=…}` expands to
  one key per label value: `vllm:request_success_total_<reason>` (e.g. `…_stop`).
- **Histograms** → do **not** store buckets in v1. Persist the two scalars a fleet
  dashboard actually uses — `_count` and `_sum` (mean = `Δsum / Δcount` on read). If
  per-node quantiles are wanted later, store a fixed set (p50/p95/p99) as their own
  keys. Full bucket retention is a deferred, opt-in concern.

## Transport — batched report, own cadence

- Sampling cadence (grid) and reporting cadence are **decoupled**: sample every
  5 s, **report every 30 s** as a batch (~6 grid slots × the metric set) to a
  manager HTTP endpoint (`POST /metrics`). This is the historical channel — the
  real-time heartbeat is the *other* channel (see *Two channels*).
- Auth: same shared bearer token as registration/heartbeat.

**Sample-batch payload (shape):** a flat list of `(timestamp, metric, value)` rows;
`node_id` is hoisted to the report level (it would repeat on every row otherwise).

```json
{
  "node_id": "…uuid…",
  "points": [
    { "timestamp": "2026-07-19 00:00:05", "metric": "worker:cpu_util_pct",  "value": 42.1 },
    { "timestamp": "2026-07-19 00:00:05", "metric": "worker:mem_used_bytes", "value": 1234567 },
    { "timestamp": "2026-07-19 00:00:05", "metric": "gpu:#0_util_pct",       "value": 88 },
    { "timestamp": "2026-07-19 00:00:10", "metric": "worker:cpu_util_pct",  "value": 39.7 }
  ]
}
```

`timestamp` is a `YYYY-MM-DD HH:MM:SS` **UTC** string aligned to the grid slot. The
manager re-attaches `node_id` to store each row as `(node_id, timestamp, metric,
value)` — the long/narrow model below.

**Backfill instead of loss.** The worker **drains** its buffer into each report and
clears it only on a 2xx; on failure it **restores the batch and retries** next
cycle. Because every row carries its grid-slot timestamp, a *late* batch still
lands in the correct slots. The buffer is bounded to **`capacity_s` (default 30
min)**: during a sustained outage it drops the **oldest `evict_chunk_s` (default
5 min)** span whenever a new sample would overflow. A transient blip → *no data
loss*; only an outage longer than the buffer → a bounded gap. This preserves "gaps
stay gaps" while recovering blips.

**Idempotent ingest.** The manager **upserts by `(node_id, metric, timestamp)`**,
never appends. A retried/overlapping batch simply rewrites the same slots, so
backfill is safe.

## Storage — long/narrow, in-memory sqlite

- **Model:** a single **long/narrow fact table** — `(node_id, timestamp, metric,
  value)`. **Not** a star/snowflake schema: at this scale (essentially one
  dimension — the node) normalized dimensions add joins for no payoff. The few
  node attributes (GPU model, etc.) are denormalized or read from the registry.
- **Store (v1): in-memory `sqlite3`** (Python standard library) — table
  **`metric_fact`** holding the row above, PK / upsert key `(node_id, metric,
  timestamp)`, plus a `timestamp` index for pruning. SQL gives the dashboard
  group-by and rollups for free. Zero external deps; handles this volume trivially.
  *(The worker's own send-buffer — a bounded in-memory list, drain-on-success — is
  a separate structure; see Transport.)*
- **Status: implemented.** `POST /metrics` upserts into `metric_fact`;
  `GET /metrics/latest` returns each node's most recent grid slot (powers the
  console `metrics` command).
- **Rendering:** **pivot long → columnar on read** for uPlot (shared grid-slot x
  axis + one y array per node). Long-format is best for ingest/churn; columnar is
  best for the chart — convert at the boundary.
- **Retention / churn eviction (24 h):** prune **by time** — each ingest runs
  `DELETE FROM metric_fact WHERE timestamp < now−24h`, so the table holds at most
  the last 24 hours. A stopped node's rows age out on their own — churn self-heals;
  that `DELETE` is the only cleanup path. (Pruning is inline on ingest today —
  cheap and indexed; a periodic sweep is an easy swap if the coupling is unwanted.)
- **Raw, derive on read:** store vLLM counters as-is; compute rates (tokens/s) at
  query time from adjacent samples, so we can re-derive over any window.

**Durability (optional, later):** in-memory sqlite is lost on a manager restart —
consistent with the registry, which is likewise rebuilt from scratch. If history
must survive a restart, switch the same schema to an **on-disk** sqlite file (WAL
mode) — no model change. **Downsampling: skip** until retention reaches
weeks/months.

## Concurrency — background threads on the worker, async on the manager

Worker-side sampling and reporting run as **two background threads owned by the
coordinator**, because the coordinator is a **synchronous, thread-driven loop**
(its state machine ticks on the heartbeat, using `threading.Event`) — not an
asyncio program. A sampler thread keeps the 5 s grid (host `/proc` + GPU NVML +
a vLLM `/metrics` scrape); a reporter thread ships the buffer every 30 s; they
share the buffer under a lock. The GPU sampler picks its backend once at start and
releases NVML on stop.
- **Not async** — the coordinator isn't an event loop, so async tasks would have
  nothing to run on. Threads match the code that exists.
- **Not a separate process** — sampling `/proc` is microseconds of work; a process
  only adds IPC for no parallelism/isolation benefit. (A blocking report POST is
  exactly why the reporter is its *own* thread — so it never delays a grid sample.)
- If vLLM supervision ever genuinely starves the sampler, split it into its own
  **process** — but do not pre-pay that.

Manager-side, the ingest endpoint runs as an async handler on the control-plane
event loop and prunes inline; the store guards its sqlite connection with a lock
(defensive, like the registry). The manager *is* an async FastAPI/uvicorn server.

## Inspection — the `metrics` console command

Until the dashboard lands, the manager console (`oumigo manager run`) reads the
store directly:

- **`GET /metrics/latest`** (server) → each node's most recent grid slot and the
  metrics at it: `[{node_id, timestamp, metrics: {name: value, …}}, …]`.
- **`metrics`** (console) → prints that per worker: the latest received grid
  timestamp (UTC) and each `metric = value`. No arguments today; a history/range
  view (`metrics <node>`, `--watch`) is a natural extension the store already
  supports.

The console is an HTTP client in the parent process while the store lives in the
server child, so inspection goes over HTTP, not shared memory.

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
**metrics batch (historical)** — kept separate. Metrics: three sources (host
`/proc` + GPU NVML/nvidia-smi + vLLM `/metrics`) · grid-aligned sampling (5 s) ·
batched report (30 s), drain-on-success / restore-and-retry on failure · worker
send-buffer bounded to 30 min, evicting the oldest 5 min on overflow · flat
`(node_id, timestamp, metric, value)` rows with `YYYY-MM-DD HH:MM:SS` UTC
timestamps · manager store = **in-memory sqlite** (`metric_fact`, 24 h), upsert by
`(node_id, metric, timestamp)` · time-based retention · gaps as gaps · raw
counters, rates derived on read · `metrics` console inspection. Worker
sampling/reporting on background threads. No downsampling, no Prometheus. The uPlot
dashboard is the remaining v1 piece.

## Deferred / open questions

- **uPlot dashboard** — the manager's static page + JSON API over the store.
  `GET /metrics/latest` exists; a windowed range/query endpoint is still needed.
- Optional **on-disk** sqlite for restart durability + retention-policy specifics.
- Retention is inline-on-ingest today; move to a periodic async sweep if the
  coupling ever becomes unwanted.
- Optional read-only Prometheus-format `/metrics` interop endpoint.
- Config surface: grid interval, report interval, and buffer capacity are
  coordinator kwargs today; plumb them through `manager.yaml` (mirrors the
  `heartbeat.*` config style).
- `pynvml` is an optional (unlisted) dependency — add it to the `worker` extra in
  `pyproject.toml` if NVML should be guaranteed rather than falling back to
  `nvidia-smi`.
