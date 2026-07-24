"""Control plane: provisioning (via L2), L1 lifecycle, node registry, state.

Owns the desired-vs-actual reconciliation loop and the source of truth for what
nodes exist and what state they're in. Low-frequency and correctness-critical.
"""
