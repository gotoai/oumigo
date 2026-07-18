# oumigo

A vertical-integration toolkit for running and managing **vLLM replica fleets** — a
manager/driver node coordinating many independent vLLM worker replicas behind a
health-aware router, with a pluggable cloud-provisioning layer.

> Status: early skeleton. Nothing here is implemented yet.

## Architecture

Three layers (see the discussion notes that shaped this):

- **L1 — worker node** (`oumigo.agent`): a long-lived *coordinator* supervises a
  vLLM server as a child process, monitors health, takes start/stop/restart from
  the manager, and owns the node state machine + restart-with-give-up policy.
- **L2 — provisioning** (`oumigo.providers`): a minimal, lifecycle-shaped `Provider`
  protocol. Ships with a real `StaticProvider` (LAN/manual hosts); cloud backends
  (e.g. ConoHa, which is OpenStack-based) are future implementations of the same
  protocol.
- **L3 — manager node** (`oumigo.manager`): kept in two planes —
  - **control plane** (`manager.control`): provisions via L2, drives L1 lifecycle,
    tracks node state, feeds the dashboard.
  - **data plane** (`manager.router`): forwards client inference calls to healthy
    workers. On the hot path of every request.

Shared foundations: `oumigo.config` (typed specs + precedence resolution) and
`oumigo.protocol` (the wire contract both L1 and L3 import so it can't drift).

## Development

```bash
# from oumigo/
uv pip install -e ".[worker,dev]"     # on a GPU worker box
uv pip install -e ".[manager,dev]"    # on the manager box
oumigo version
```

To consume this in-development package from a sibling project, editable-install it
into that project's environment (e.g. `uv pip install -e ../oumigo`).
