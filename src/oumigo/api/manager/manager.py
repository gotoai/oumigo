"""The manager *handle* — a client-side view of a manager, spawned or discovered.

This is the API-layer handle returned by
:func:`oumigo.api.api.oumigo_get_or_create_manager`, not the manager *service* (that lives
under ``oumigo.service.manager``). It answers health/worker/metrics queries over the
control plane and mints inference agents (:meth:`create_agent`) against the data plane.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from oumigo.common.proc import terminate


def _auth(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


@dataclass
class OumigoManager:
    """A running manager — either one this process spawned, or one found on the LAN.

    ``owned`` is True only when this process spawned the child; ``stop()`` is a no-op
    for a discovered (remote) manager, which this process does not own.
    """

    control_url: str
    data_url: str
    token: str | None = None
    provider: str = "LAN"
    owned: bool = False
    dashboard_url: str | None = None
    _child: subprocess.Popen | None = field(default=None, repr=False)
    _config_path: str | None = field(default=None, repr=False)

    def is_healthy(self, timeout_s: float = 2.0) -> bool:
        """True once the control plane answers 200 on ``/healthz``."""
        try:
            resp = httpx.get(f"{self.control_url}/healthz", timeout=timeout_s)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def workers(self, timeout_s: float = 5.0) -> list[dict]:
        """The fleet's current worker records (``GET /workers``)."""
        resp = httpx.get(
            f"{self.control_url}/workers", headers=_auth(self.token), timeout=timeout_s
        )
        resp.raise_for_status()
        return list(resp.json().get("workers", []))

    def metrics(
        self,
        *,
        since: str | None = None,
        prefixes: Iterable[str] | None = None,
        timeout_s: float = 5.0,
    ) -> list[dict]:
        """Worker metrics collected by the manager.

        Default (``since=None``): the **latest grid slot per node** — a list of
        ``{"node_id", "name", "timestamp", "metrics": {metric: value, ...}}`` (the
        ``name`` is the friendly ``Worker#N`` label, best-effort). Metric names look
        like ``worker:cpu_util_pct`` / ``gpu:*`` / ``vllm:*`` and values are floats;
        ``*_timestamp`` metrics are UTC epoch seconds.

        With ``since`` set to a ``"YYYY-MM-DD HH:MM:SS"`` UTC grid-slot string (or
        ``""`` for the whole retained window): **raw historical points** newer than
        that watermark — a list of ``{"node_id", "metric", "timestamp", "value"}``,
        ascending by timestamp so the caller can advance its own watermark.

        ``prefixes`` keeps only metrics whose name starts with one of them, e.g.
        ``("worker:", "gpu:")`` — applied server-side for ``since``, client-side for
        the latest snapshot.
        """
        prefix_tuple = tuple(prefixes) if prefixes else ()

        if since is not None:
            params: dict[str, str] = {"after": since}
            if prefix_tuple:
                params["prefix"] = ",".join(prefix_tuple)
            resp = httpx.get(
                f"{self.control_url}/metrics/since",
                params=params,
                headers=_auth(self.token),
                timeout=timeout_s,
            )
            resp.raise_for_status()
            return list(resp.json().get("points", []))

        resp = httpx.get(
            f"{self.control_url}/metrics/latest", headers=_auth(self.token), timeout=timeout_s
        )
        resp.raise_for_status()
        names = self._worker_names(timeout_s)

        out: list[dict] = []
        for record in resp.json().get("nodes", []):
            node_metrics = record.get("metrics", {})
            if prefix_tuple:
                node_metrics = {k: v for k, v in node_metrics.items() if k.startswith(prefix_tuple)}
            out.append(
                {
                    "node_id": record.get("node_id"),
                    "name": names.get(record.get("node_id")),
                    "timestamp": record.get("timestamp"),
                    "metrics": node_metrics,
                }
            )
        return out

    def _worker_names(self, timeout_s: float = 5.0) -> dict[str, str]:
        """Best-effort ``node_id -> Worker#N`` map for labeling metrics; ``{}`` on failure."""
        try:
            resp = httpx.get(
                f"{self.control_url}/workers", headers=_auth(self.token), timeout=timeout_s
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        return {
            w["node_id"]: w.get("name", "")
            for w in resp.json().get("workers", [])
            if "node_id" in w
        }

    def stop(self) -> None:
        """Stop the spawned control-plane child (and its dashboard). No-op if not owned."""
        if not self.owned:
            return
        terminate(self._child)
        self._child = None
        if self._config_path:
            Path(self._config_path).unlink(missing_ok=True)
            self._config_path = None

    def create_agent(
        self,
        tools: Any = None,
        *,
        max_iterations: int = 5,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> Any:
        """Create an agent that runs chats/tools against this manager's data plane.

        ``tools`` is a sequence of :class:`oumigo.api.agent.Tool` (or plain functions, which
        are wrapped via the strict ``@tool`` validator). Sampling defaults given here apply
        to every chat the agent spawns. ``max_iterations`` caps the model round-trips per
        request (the runaway tool-loop guard). Returns an ``OumigoAgent``; call
        ``.create_chat(...)`` on it to start a conversation.
        """
        from oumigo.api.agent.agent import OumigoAgent  # local import: optional inference layer

        sampling = {
            k: v
            for k, v in (
                ("temperature", temperature), ("max_tokens", max_tokens),
                ("top_p", top_p), ("stop", stop),
            )
            if v is not None
        }
        return OumigoAgent(
            data_url=self.data_url,
            token=self.token,
            tools=tools or [],
            max_iterations=max_iterations,
            sampling=sampling,
        )

    def __enter__(self) -> OumigoManager:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
