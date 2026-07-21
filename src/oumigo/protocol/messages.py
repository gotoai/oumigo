"""Control-plane message schemas exchanged between worker and manager.

Both sides import these so the wire contract cannot drift. If a message crosses
the network, its schema lives here.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from oumigo.config.spec import NodeSpec
from oumigo.protocol.states import NodeState, RunState


class NodeCapabilities(BaseModel):
    """What a worker can do. Placeholder — GPU details filled in later."""

    gpu: str | None = None
    vram_gb: float | None = None


class WorkerCommand(str, Enum):
    """Commands the manager can hand a worker on the heartbeat ack (pull channel)."""

    STOP = "stop"  # drain in-flight work, shut vLLM down cleanly, go STOPPED


class RegisterRequest(BaseModel):
    """Worker -> manager: announce identity and ask to join the fleet.

    `node_id` is the deterministic `sha256("oumigo:<address>:<vllm_port>")[:16]`
    the worker derives after preflighting its port, so it is reported together with
    that `vllm_port` (the manager can recompute and validate the pair).
    """

    node_id: str
    address: str                       # LAN-reachable address the worker advertises
    vllm_port: int | None = None       # actual (preflight-selected) vLLM port; part of node_id
    model: str | None = None           # effective model the worker serves (env-negotiable)
    incarnation: int = 0               # bumped each worker start; same identity
    state: NodeState = NodeState.REGISTERING
    capabilities: NodeCapabilities = Field(default_factory=NodeCapabilities)


class RegisterResponse(BaseModel):
    """Manager -> worker: accept the registration, hand back cadence and vLLM config.

    `node_spec` is the vLLM configuration the worker must run (homogeneous fleet:
    the manager derives it from its single model config). It is None only when the
    manager has no model configured yet — the worker then cannot start vLLM.
    """

    accepted: bool
    node_id: str
    heartbeat_interval_s: int = 10
    node_spec: NodeSpec | None = None
    message: str | None = None


class HeartbeatRequest(BaseModel):
    """Worker -> manager: liveness ping carrying both state axes.

    `run_state` is only meaningful while `node_state` is SERVING or DRAINING; it is
    None otherwise (see docs/worker-node-states.md).
    """

    node_id: str
    node_state: NodeState
    run_state: RunState | None = None
    vllm_port: int | None = None  # worker's actual serving port (may differ from model.port)
    max_concurrent_requests: int | None = None  # worker's negotiated in-flight cap (None = unchanged)


class HeartbeatResponse(BaseModel):
    """Manager -> worker: ack. `known=False` means re-register (manager forgot us).

    `command`, when set, is an out-of-band instruction (e.g. STOP) the worker acts
    on after this beat.
    """

    ok: bool
    known: bool = True
    command: WorkerCommand | None = None


class MetricPoint(BaseModel):
    """One metric data point in a batch. `node_id` is hoisted to the report level.

    `timestamp` is the grid-slot wall-clock in 'YYYY-MM-DD HH:MM:SS' (UTC) form, so
    the manager stores rows as (node_id, timestamp, metric, value) — the long/narrow
    fact model in docs/metrics.md.
    """

    timestamp: str
    metric: str
    value: float


class MetricsReport(BaseModel):
    """Worker -> manager: a buffered batch of grid-aligned samples (historical channel).

    Separate from the heartbeat: delay-tolerant, buffered, and retried on failure.
    """

    node_id: str
    points: list[MetricPoint]
