"""Coordinator — the long-lived node agent.

Wraps a VLLMProcess with the control/reporting surface (register, heartbeat,
start/stop/restart) and owns the NodeState machine + restart policy. This is the
process that actually runs on a worker box; the manager drives it over HTTP.
"""

from __future__ import annotations

# class Coordinator:
#     """Supervises vLLM, reports state upward, executes manager commands."""
