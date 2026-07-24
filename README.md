# oumigo

A vertical-integration toolkit for running and managing **vLLM replica fleets** — a
manager/driver node coordinating many independent vLLM worker replicas behind a
health-aware router, with a pluggable cloud-provisioning layer.

> Status: early development. The CLI, manager console, config resolution, and the
> LAN provider exist; core worker↔manager registration, the router, and vLLM
> supervision are not built yet.

## Architecture

Two roles:

- **Worker** (`oumigo.service.worker`): a long-lived *coordinator* supervises a vLLM server
  as a child process, monitors health, executes start/stop/restart from the manager,
  and owns the node state machine + restart-with-give-up policy. Workers self-register
  with the manager and heartbeat.
- **Manager** (`oumigo.service.manager`): coordinates the fleet, split into sub-layers:
  - **control plane** (`manager.control`): tracks worker registrations and state,
    drives worker lifecycle, reconciles desired vs. actual. Low-frequency,
    correctness-critical.
  - **data plane / router** (`manager.router`): forwards client inference calls to
    healthy workers — on the hot path of every request.
  - **provisioning** (`oumigo.providers`): how workers come into existence — a
    minimal, lifecycle-shaped `Provider` protocol used by the control plane. Ships
    with `StaticProvider` (LAN: workers are hand-started and self-register, no
    provisioning); cloud backends (e.g. ConoHa, OpenStack-based) are future
    implementations of the same protocol.
  - **dashboard** (`manager.dashboard`): performance & diagnostics — later.

Shared foundations: `oumigo.config` (typed settings + precedence resolution) and
`oumigo.protocol` (the wire contract both roles import so it can't drift).

## Development

```bash
# from oumigo/
python -m venv .venv
source .venv/bin/activate
pip install -e ".[worker,dev]"     # on a GPU worker box
pip install -e ".[manager,dev]"    # on the manager box
oumigo version
```

To consume this in-development package from a sibling project, editable-install it
into that project's environment (e.g. `pip install -e ../oumigo`).
