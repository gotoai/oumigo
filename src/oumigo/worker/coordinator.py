"""Coordinator — the long-lived worker-node process.

For v1 (no vLLM yet) it: resolves the node identity, finds the manager (explicit
URL, else mDNS discovery), registers, and heartbeats until stopped. vLLM
supervision plugs in here later.
"""

from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path

import httpx

from oumigo import discovery
from oumigo.protocol.messages import HeartbeatRequest, NodeCapabilities, RegisterRequest
from oumigo.protocol.states import NodeState
from oumigo.worker import client
from oumigo.worker.identity import resolve_node_identity

log = logging.getLogger("oumigo.worker")


def run_worker(
    manager_url: str | None,
    token: str | None,
    state_dir: Path | None = None,
    discover_timeout: float = discovery.DEFAULT_DISCOVER_TIMEOUT,
) -> None:
    node_id, incarnation, path = resolve_node_identity(state_dir)
    log.info("node identity %s (incarnation %d) [%s]", node_id, incarnation, path)

    if not manager_url:
        log.info("no manager URL provided; discovering via mDNS (up to %.0fs) ...", discover_timeout)
        manager_url = discovery.discover_manager(discover_timeout)
        if not manager_url:
            raise SystemExit(
                f"could not discover a manager on the LAN within {discover_timeout:.0f}s; "
                "set --manager-url / $OUMIGO_MANAGER_URL"
            )
        log.info("discovered manager at %s", manager_url)

    address = discovery.get_lan_ip()
    register_req = RegisterRequest(
        node_id=node_id,
        address=address,
        incarnation=incarnation,
        state=NodeState.REGISTERING,
        capabilities=NodeCapabilities(),
    )
    try:
        resp = client.register(manager_url, register_req, token)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == 401:
            raise SystemExit("registration rejected (401): check the shared bearer token")
        raise SystemExit(f"registration failed: HTTP {code}")
    except httpx.RequestError as exc:
        raise SystemExit(f"could not reach manager at {manager_url}: {exc}")
    interval = resp.heartbeat_interval_s or 10
    log.info(
        "registered with manager %s (accepted=%s, heartbeat=%ds)",
        manager_url,
        resp.accepted,
        interval,
    )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    while not stop.wait(interval):
        try:
            hb = client.heartbeat(
                manager_url, HeartbeatRequest(node_id=node_id, state=NodeState.READY), token
            )
            if not hb.known:
                log.warning("manager does not know us; re-registering")
                client.register(manager_url, register_req, token)
            else:
                log.debug("heartbeat ok")
        except Exception as exc:  # noqa: BLE001 - keep the worker alive across blips
            log.warning("heartbeat failed: %s", exc)

    log.info("worker stopping")
