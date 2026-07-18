"""Control- and data-plane message schemas exchanged over the network.

Control plane (manager <-> agent): register, heartbeat, start/stop/restart, status.
Data plane (client <-> router): inference request/response envelopes if/when the
router adds anything beyond transparent pass-through of the vLLM OpenAI API.

Define these as pydantic models so both sides validate identically.
"""

from __future__ import annotations

# class RegisterRequest(BaseModel): ...
# class Heartbeat(BaseModel): ...
# class NodeStatus(BaseModel): ...
