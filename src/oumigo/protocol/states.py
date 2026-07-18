"""Node lifecycle state machine — the vocabulary the manager schedules against.

These names leak into logs, metrics, and the scheduler, so get them right early.
"""

from __future__ import annotations

from enum import Enum


class NodeState(str, Enum):
    REGISTERING = "registering"  # coordinator up, announcing itself to the manager
    READY = "ready"              # vLLM healthy, no traffic yet
    SERVING = "serving"          # vLLM healthy, taking requests
    DRAINING = "draining"        # finishing in-flight work before stop
    STOPPED = "stopped"          # cleanly shut down
    FAILED = "failed"            # terminal failure; restart policy gave up (worker-reported)
    LOST = "lost"                # manager stopped receiving heartbeats (manager-observed)
