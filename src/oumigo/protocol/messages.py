"""Control-plane message schemas exchanged between worker and manager.

Both sides import these so the wire contract cannot drift. If a message crosses
the network, its schema lives here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from oumigo.protocol.states import NodeState


class NodeCapabilities(BaseModel):
    """What a worker can do. Placeholder — GPU details filled in later."""

    gpu: str | None = None
    vram_gb: float | None = None


class RegisterRequest(BaseModel):
    """Worker -> manager: announce identity and ask to join the fleet."""

    node_id: str
    address: str                       # LAN-reachable address the worker advertises
    incarnation: int = 0               # bumped each worker start; same identity
    state: NodeState = NodeState.REGISTERING
    capabilities: NodeCapabilities = Field(default_factory=NodeCapabilities)


class RegisterResponse(BaseModel):
    """Manager -> worker: accept the registration and hand back cadence."""

    accepted: bool
    node_id: str
    heartbeat_interval_s: int = 10
    message: str | None = None


class HeartbeatRequest(BaseModel):
    """Worker -> manager: liveness ping."""

    node_id: str
    state: NodeState = NodeState.READY


class HeartbeatResponse(BaseModel):
    """Manager -> worker: ack. `known=False` means re-register (manager forgot us)."""

    ok: bool
    known: bool = True
