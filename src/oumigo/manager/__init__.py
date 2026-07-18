"""L3 — the manager node.

Two responsibilities with very different characteristics, kept logically separate:

  * control plane (`manager.control`): provisions via L2, drives L1 lifecycle,
    tracks node state, feeds the dashboard. Low-frequency, correctness-critical.
  * data plane (`manager.router`): forwards client inference calls to healthy
    workers. On the hot path of every request/token — async, must not block on
    control-plane work.

Start as one process with a clean internal boundary; the split lets the router
move to its own process/replica set later without surgery.
"""
