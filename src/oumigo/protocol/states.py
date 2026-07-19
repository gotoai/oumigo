"""Node lifecycle state machine — the vocabulary the manager schedules against.

These names leak into logs, metrics, and the scheduler, so get them right early.
See docs/worker-node-states.md for the full design (two axes: lifecycle + run).
"""

from __future__ import annotations

from enum import Enum


class NodeState(str, Enum):
    """Coarse lifecycle state — what the manager schedules against."""

    REGISTERING = "registering"    # coordinator up, announcing itself to the manager
    INITIALIZING = "initializing"  # registered + config received; vLLM booting / loading weights
    SERVING = "serving"            # vLLM healthy and accepting requests (subsumes old READY)
    DRAINING = "draining"          # stop requested; finishing in-flight work before shutdown
    STOPPED = "stopped"            # cleanly shut down
    FAILED = "failed"              # terminal failure; restart policy gave up (worker-reported)
    LOST = "lost"                  # manager stopped receiving heartbeats (manager-observed)


class RunState(str, Enum):
    """vLLM activity — only meaningful while NodeState is SERVING or DRAINING.

    A coarse projection of vLLM's continuous-batching load; the fine-grained
    gradient (in-flight count, queue depth, KV-cache use) lives in metrics.
    """

    IDLE = "idle"            # vLLM up, zero in-flight requests
    EXECUTING = "executing"  # vLLM up, >=1 request in flight
