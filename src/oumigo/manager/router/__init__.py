"""Data plane: health-aware router forwarding client API calls to worker vLLMs.

Start with round-robin / least-loaded over *healthy* replicas (health-awareness
is the only must-have; smart scheduling is a later knob). Must be async and must
not block on control-plane work.
"""
