# Worker Node States — Lifecycle & Run-State Design

> Status: implemented (worker side). The state machine, vLLM supervision, and the
> protocol/manager wiring described here are built (`oumigo.worker.coordinator`,
> `oumigo.worker.supervisor`). Not yet wired: the manager-side *trigger* for STOP
> (the pull channel exists on the heartbeat ack, but nothing sets it yet) and
> vLLM-sourced load telemetry (run-state is derived best-effort from vLLM
> `/metrics`; richer metrics are a later phase).

## Purpose

Define the vocabulary the manager schedules against and the worker coordinator
reports. These names leak into logs, metrics, and the scheduler, so they are
pinned down before implementation.

## Two axes, not one

A worker is described by **two separate enums**, because "where is this node in
its lifecycle" and "is vLLM busy right now" are different questions with
different consumers.

| Axis | Enum | Consumer | Answers |
|---|---|---|---|
| **Lifecycle** | `NodeState` | manager scheduler, liveness reaper | Can this node be dispatched to at all? |
| **Run activity** | `RunState` | load/routing, drain logic | Is vLLM currently working? |

Overloading a single enum (the old `READY` vs `SERVING` split) conflated the two
and forced an awkward "first request ever" latch. Splitting them removes the
latch and gives each axis one clear meaning.

## `NodeState` — lifecycle

The coarse state the manager schedules against and the coordinator reports on the
heartbeat.

| State | Meaning | Reporter |
|---|---|---|
| `REGISTERING` | coordinator up, announcing itself to the manager; no vLLM yet | worker |
| `INITIALIZING` | registered, config received; vLLM booting / loading weights (minutes) | worker |
| `SERVING` | vLLM healthy and accepting requests (subsumes old `READY` + `SERVING`) | worker |
| `DRAINING` | stop requested; refusing new work, waiting for in-flight to finish | worker |
| `STOPPED` | cleanly shut down | worker |
| `FAILED` | terminal failure; restart policy exhausted | worker |
| `LOST` | manager stopped receiving heartbeats | manager-observed |

**Happy path:** `REGISTERING → INITIALIZING → SERVING → DRAINING → STOPPED`.

`FAILED` and `LOST` are off-path terminals. `INITIALIZING` is new (was missing);
it exists so the multi-minute model load is *visible* on the heartbeat rather
than being mistaken for a `LOST` silence.

## `RunState` — vLLM activity

| State | Meaning |
|---|---|
| `IDLE` | vLLM up, zero in-flight requests |
| `EXECUTING` | vLLM up, ≥1 request in flight |

**`RunState` is nested under `SERVING`, not orthogonal.** It only has meaning
while vLLM is up — i.e. `NodeState ∈ {SERVING, DRAINING}`. In every other
lifecycle state it is undefined (`None` on the wire). It is a sub-state of
SERVING, not a free cross-product.

`IDLE`/`EXECUTING` is a **coarse projection** of a continuous quantity: vLLM does
continuous batching, so real "busyness" is *N in-flight + queue depth + KV-cache
utilization*. The binary enum is deliberately coarse:

- **Lifecycle/drain use it as a binary** — DRAINING needs exactly "is anything in
  flight?" and nothing finer.
- **Scheduling wants the gradient** — which warm node gets the next request is a
  load decision, and that gradient lives in [metrics](metrics.md), **not** in
  this enum.

The enum also leaves room to grow a coarse backpressure signal later (e.g.
`OVERLOADED`) without touching the lifecycle axis.

## Workflow

1. Coordinator process starts → `NodeState = REGISTERING`.
2. It looks for the manager. If it cannot find one before timeout, it **quits**
   (never registered, so no state transition needed).
3. On successful registration — including receipt of the vLLM config —
   → `INITIALIZING`, and it starts the vLLM server with that config.
4. When vLLM is healthy → `SERVING` (run-state `IDLE`). The node enters SERVING
   the moment vLLM is ready; there is no "wait for the first request" step.
5. Run-state tracks activity continuously: `IDLE → EXECUTING` on first in-flight
   request, back to `IDLE` when in-flight drops to zero. `NodeState` stays
   `SERVING` throughout.
6. On `STOP` from the manager:
   - If run-state is `EXECUTING` → `DRAINING`: refuse new work, wait for
     run-state to fall `EXECUTING → IDLE`, then `STOPPED`.
   - If already `IDLE` (or still `INITIALIZING`/pre-traffic) → nothing to drain →
     straight to `STOPPED`.
7. After a clean shutdown → `STOPPED`. In a **LAN** provision the coordinator
   process stays up; in a **cloud** provision the instance is also shut down.
8. If vLLM fails, the coordinator applies its **restart policy** first:
   - attempt restart → re-enter `INITIALIZING` (a restart *is* a re-init);
   - only when the restart policy is **exhausted** → `FAILED` (terminal).

   Recovery therefore happens *before* `FAILED`, not after — `FAILED` stays
   genuinely terminal. The [`incarnation`](../src/oumigo/protocol/messages.py)
   field is bumped on each restart so the manager can tell a recovered node from
   a fresh one.
9. (passive) If the manager's reaper stops seeing heartbeats → `LOST`
   (manager-observed). `LOST` is not necessarily terminal for the worker: a
   transient blip can be recovered — a later heartbeat gets
   `HeartbeatResponse.known = False` and the worker loops back to `REGISTERING`.

## Transition map

```
REGISTERING ──register + config──▶ INITIALIZING ──vLLM healthy──▶ SERVING
                                     ▲                            (IDLE ⇄ EXECUTING)
                                     │ restart (policy)                 │
                                     │                                  │ STOP
             vLLM crash ────────────┤                                  ▼
                                     │ (policy exhausted)   DRAINING ──drained──▶ STOPPED
                                     ▼                     (wait EXECUTING→IDLE)
                                   FAILED                    │
                                                             └─ STOP while IDLE ─▶ STOPPED (no drain)

LOST ── manager-observed on heartbeat loss; worker may re-REGISTER on known=False
```

## Protocol impact

The heartbeat currently carries a single `state: NodeState`
([messages.py](../src/oumigo/protocol/messages.py)). Under this design it carries
**both axes**:

- `node_state: NodeState` — required.
- `run_state: RunState | None` — `None` unless `node_state ∈ {SERVING, DRAINING}`.

The old `HeartbeatRequest.state = READY` default is stale (no `READY` state
exists anymore) and should become a required field or default to `INITIALIZING`.

## Open questions

- Distinct `RESTARTING`/`RECOVERING` state vs. reusing `INITIALIZING` for crash
  recovery — currently reusing `INITIALIZING`, relying on `incarnation` to
  distinguish. Revisit if logs/metrics need to separate cold boot from recovery.
- Coarse backpressure state (`OVERLOADED`) on the `RunState` axis — deferred
  until there is a concrete routing consumer.
- Exact restart-policy shape (retry count, backoff) that governs the
  `INITIALIZING ⇄ crash → FAILED` loop.
