"""Shared wire contract between the worker (L1) and the manager (L3).

Both sides import these schemas so control- and data-plane messages can never
drift. Rule of thumb: if a message crosses the network, its schema lives here.
"""

from oumigo.protocol.states import NodeState, RunState

__all__ = ["NodeState", "RunState"]
