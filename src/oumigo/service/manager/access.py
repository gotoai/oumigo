"""Source-address checks on the worker-facing endpoints.

- `WorkerAllowlist`: the optional on/off list of worker addresses / CIDR ranges
  (`worker_allowlist` in manager.yaml). A layer on top of join tokens, never instead of
  them — rented pods (RunPod, Vast) connect out through datacenter NAT, so their
  address isn't known before launch; rejected addresses are logged so an operator
  can add a new pod's.
- `FailureLimiter`: throttles a source address after repeated auth failures.

The client address is uvicorn's: behind nginx it comes from X-Forwarded-For, believed
only from `control_plane.trusted_proxies`, so a worker can't spoof it.
"""

from __future__ import annotations

import ipaddress
import logging
import time
from collections import defaultdict, deque

log = logging.getLogger("oumigo.service.manager.access")


class WorkerAllowlist:
    """Allowed worker source networks; ``None`` entries mean the switch is off."""

    def __init__(self, entries: list[str] | None) -> None:
        self.enabled = entries is not None
        self._networks = [ipaddress.ip_network(e.strip(), strict=False) for e in entries or []]

    def allows(self, host: str | None) -> bool:
        if not self.enabled:
            return True
        try:
            address = ipaddress.ip_address(host or "")
        except ValueError:
            return False
        if address.version == 6 and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        return any(address in network for network in self._networks)


class FailureLimiter:
    """At most `max_failures` failed auth attempts per address per `window_s`; beyond
    that the address is answered 429 until old failures age out."""

    def __init__(self, max_failures: int = 30, window_s: float = 60.0) -> None:
        self.max_failures = max_failures
        self.window_s = window_s
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def _recent(self, host: str) -> deque[float]:
        times = self._failures[host]
        cutoff = time.monotonic() - self.window_s
        while times and times[0] < cutoff:
            times.popleft()
        return times

    def blocked(self, host: str | None) -> bool:
        return host is not None and len(self._recent(host)) >= self.max_failures

    def record(self, host: str | None) -> None:
        if host is None:
            return
        times = self._recent(host)
        times.append(time.monotonic())
        if len(times) == self.max_failures:
            log.warning("throttling %s after %d failed auth attempts in %.0fs",
                        host, self.max_failures, self.window_s)
