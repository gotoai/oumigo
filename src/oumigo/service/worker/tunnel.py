"""Worker side of the tunnel transport: keep N outbound WebSockets to the manager.

A `tunnel` worker needs no inbound port. It dials the manager's `/tunnel` WebSocket
(`wss://` under CUSTOM, through nginx) and keeps ``pool_size`` connections open — the
manager hands that number out, capacity + spare. Each connection is a *slot*:

1. connect and wait, parked, until the manager sends the first bytes of a request;
2. open a TCP connection to the local backend (127.0.0.1:<port>) and copy bytes both
   ways — one WebSocket carries exactly one HTTP/1.1 connection, unmodified;
3. when either side closes, close the other, then go back to 1 with a fresh connection.

So the worker never needs to know whether a connection is idle or in use, and closing
is how cancellation propagates: the manager closes the WebSocket when the client goes
away, the slot closes its backend connection, and vLLM aborts the request.

The coordinator is synchronous (threads + blocking httpx), so the client runs its own
asyncio loop in a daemon thread. Reconnects back off exponentially with jitter.
"""

from __future__ import annotations

import asyncio
import logging
import random
import ssl
import threading
from collections.abc import Callable

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

log = logging.getLogger("oumigo.service.worker.tunnel")

PING_INTERVAL_S = 20.0  # keeps NAT mappings alive; also samples RTT
BACKOFF_START_S = 0.5
BACKOFF_MAX_S = 30.0
_COPY_CHUNK = 64 * 1024


def tunnel_url(manager_url: str) -> str:
    """The `/tunnel` WebSocket URL for a manager control-plane URL."""
    base = manager_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):] + "/tunnel"
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):] + "/tunnel"
    raise ValueError(f"manager URL must be http:// or https://, got {manager_url!r}")


class TunnelClient:
    """Maintains this worker's pool of tunnel connections in a background thread."""

    def __init__(
        self,
        manager_url: str,
        token: Callable[[], str | None],
        backend_port: int,
        pool_size: int,
        ssl_context: ssl.SSLContext | None = None,
        backend_host: str = "127.0.0.1",
    ) -> None:
        self.url = tunnel_url(manager_url)
        self._token = token  # read per connect: the coordinator may swap credentials
        self.backend_host = backend_host
        self.backend_port = backend_port
        self._pool_size = max(1, pool_size)
        self._ssl = ssl_context if self.url.startswith("wss://") else None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._main: asyncio.Task | None = None
        self._resized: asyncio.Event | None = None
        self._latency: dict[int, float] = {}  # slot -> last ping RTT (s)
        self.connected = 0  # open WebSockets (parked or carrying a request)

    # --- control (called from the coordinator's thread) ---------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="oumigo-tunnel", daemon=True)
        self._thread.start()
        log.info("tunnel: keeping %d connection(s) to %s", self._pool_size, self.url)

    def stop(self, timeout: float = 5.0) -> None:
        loop, main = self._loop, self._main
        if loop is not None and main is not None:
            loop.call_soon_threadsafe(main.cancel)
        if self._thread is not None:
            self._thread.join(timeout)

    def set_pool_size(self, size: int | None) -> None:
        if size is None or max(1, size) == self._pool_size:
            return
        log.info("tunnel: pool size %d -> %d", self._pool_size, size)
        self._pool_size = max(1, size)
        if self._loop is not None and self._resized is not None:
            self._loop.call_soon_threadsafe(self._resized.set)

    @property
    def rtt_ms(self) -> float | None:
        """Smallest recent WebSocket ping RTT across the slots, in ms (None before any)."""
        samples = [v for v in self._latency.values() if v > 0]
        return round(min(samples) * 1000, 1) if samples else None

    # --- the loop -----------------------------------------------------------------

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self._supervise())
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.close()

    async def _supervise(self) -> None:
        """Keep exactly `pool_size` slot tasks running; resize when the manager asks."""
        self._main = asyncio.current_task()
        self._resized = asyncio.Event()
        slots: dict[int, asyncio.Task] = {}
        try:
            while True:
                for i in range(self._pool_size):
                    if i not in slots or slots[i].done():
                        slots[i] = asyncio.create_task(self._slot(i))
                for i in [i for i in slots if i >= self._pool_size]:
                    slots.pop(i).cancel()  # shrink: drop the highest slots
                    self._latency.pop(i, None)
                self._resized.clear()
                await self._resized.wait()
        finally:
            for task in slots.values():
                task.cancel()
            await asyncio.gather(*slots.values(), return_exceptions=True)

    async def _slot(self, index: int) -> None:
        backoff = BACKOFF_START_S
        while True:
            try:
                token = self._token()
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                async with connect(
                    self.url,
                    additional_headers=headers,
                    ssl=self._ssl,
                    ping_interval=PING_INTERVAL_S,
                    ping_timeout=PING_INTERVAL_S,
                    max_size=None,  # request bodies (images) can be large
                    open_timeout=10,
                    close_timeout=5,
                ) as ws:
                    backoff = BACKOFF_START_S
                    self.connected += 1
                    try:
                        await self._carry(index, ws)
                    finally:
                        self.connected -= 1
                continue  # served (or closed) — dial a fresh connection right away
            except asyncio.CancelledError:
                raise
            except InvalidStatus as exc:
                status = exc.response.status_code
                log.warning("tunnel slot %d refused by manager (HTTP %d); retrying", index, status)
            except (OSError, InvalidHandshake, ConnectionClosed, asyncio.TimeoutError) as exc:
                log.debug("tunnel slot %d: %s", index, exc)
            delay = backoff * random.uniform(0.5, 1.5)
            backoff = min(backoff * 2, BACKOFF_MAX_S)
            await asyncio.sleep(delay)

    async def _carry(self, index: int, ws: ClientConnection) -> None:
        """Park until a request arrives, then bridge it to the backend until either side
        closes."""
        first = await self._recv_parked(index, ws)
        if first is None:
            return
        try:
            reader, writer = await asyncio.open_connection(self.backend_host, self.backend_port)
        except OSError as exc:
            log.warning("tunnel: backend %s:%d unreachable: %s", self.backend_host,
                        self.backend_port, exc)
            await ws.close(code=1011, reason="backend unreachable")
            return
        writer.write(first)

        async def ws_to_backend() -> None:
            async for message in ws:
                writer.write(message if isinstance(message, bytes) else message.encode())
                await writer.drain()

        async def backend_to_ws() -> None:
            while data := await reader.read(_COPY_CHUNK):
                await ws.send(data)

        tasks = [asyncio.create_task(ws_to_backend()), asyncio.create_task(backend_to_ws())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            await ws.close()

    async def _recv_parked(self, index: int, ws: ClientConnection) -> bytes | None:
        """Wait for the first request bytes, sampling ping RTT meanwhile. None = closed."""
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), PING_INTERVAL_S)
            except asyncio.TimeoutError:
                self._latency[index] = ws.latency
                continue
            except ConnectionClosed:
                return None
            self._latency[index] = ws.latency
            return message if isinstance(message, bytes) else message.encode()
