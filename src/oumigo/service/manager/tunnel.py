"""Manager side of the tunnel transport: a reverse connection pool per worker.

A `tunnel` worker dials *out* to the control plane's `/tunnel` WebSocket and keeps
``capacity + pool_spare`` connections open. Each connection carries exactly one plain
HTTP/1.1 connection to the worker's backend: binary WebSocket messages are the raw
bytes, with no envelope. The router borrows a parked connection per upstream
connection, so no multiplexing, framing protocol or flow control is needed, and a lost
packet stalls only its own stream (see Memo/design_oumigo_tunnel_transport.md).

How the router uses it: the httpx client for tunnel workers runs on
`TunnelNetworkBackend`, whose `connect_tcp(host="<node_id>.tunnel")` hands out a parked
connection instead of opening a socket. Everything above that — HTTP/1.1, keep-alive,
SSE streaming, closing on client disconnect — is ordinary httpx/httpcore. Closing the
stream closes the WebSocket; the worker then closes its backend connection, which aborts
the request in vLLM and frees the GPU.

All state lives on the manager's single event loop, so nothing here needs locks.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque

import httpcore
import httpx
from starlette.websockets import WebSocket, WebSocketDisconnect

log = logging.getLogger("oumigo.service.manager.tunnel")

TUNNEL_HOST_SUFFIX = ".tunnel"  # router URL host: "<node_id>.tunnel"
_EOF = b""


def tunnel_host(node_id: str) -> str:
    return node_id + TUNNEL_HOST_SUFFIX


class TunnelConn:
    """One accepted `/tunnel` WebSocket = one HTTP/1.1 connection to a worker backend.

    `pump` (run by the endpoint) is the only reader of the socket: it queues incoming
    bytes, so a parked connection still notices the worker closing it.
    """

    def __init__(self, ws: WebSocket, node_id: str) -> None:
        self.ws = ws
        self.node_id = node_id
        self._incoming: asyncio.Queue[bytes] = asyncio.Queue()
        self._pending = b""  # bytes from a message larger than the last read asked for
        self.closed = asyncio.Event()
        self.borrowed = False  # handed to the router (never parked again)

    async def pump(self) -> None:
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is None and message.get("text") is not None:
                    data = message["text"].encode()
                if data:
                    self._incoming.put_nowait(data)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self.closed.set()
            self._incoming.put_nowait(_EOF)

    @property
    def readable(self) -> bool:
        """Data or EOF is waiting. For an idle pooled connection this means the worker
        side closed it — httpcore then discards it instead of reusing it."""
        return bool(self._pending) or not self._incoming.empty() or self.closed.is_set()

    async def read(self, max_bytes: int) -> bytes:
        if not self._pending:
            self._pending = await self._incoming.get()
            if self._pending == _EOF:
                self._incoming.put_nowait(_EOF)  # keep reporting EOF on later reads
                return _EOF
        data, self._pending = self._pending[:max_bytes], self._pending[max_bytes:]
        return data

    async def write(self, data: bytes) -> None:
        if self.closed.is_set():
            raise ConnectionResetError("tunnel connection closed")
        await self.ws.send_bytes(data)

    async def close(self) -> None:
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            await self.ws.close()
        except RuntimeError:
            pass  # already closing/closed from the other side


class TunnelRegistry:
    """Parked tunnel connections per node, and the router's waiters for them."""

    def __init__(self) -> None:
        self._idle: dict[str, deque[TunnelConn]] = defaultdict(deque)
        self._all: dict[str, set[TunnelConn]] = defaultdict(set)
        self._waiters: dict[str, deque[asyncio.Future[TunnelConn]]] = defaultdict(deque)

    def add(self, conn: TunnelConn) -> None:
        self._all[conn.node_id].add(conn)
        waiters = self._waiters[conn.node_id]
        while waiters:
            fut = waiters.popleft()
            if not fut.done():
                conn.borrowed = True
                fut.set_result(conn)
                return
        self._idle[conn.node_id].append(conn)

    def remove(self, conn: TunnelConn) -> None:
        self._all[conn.node_id].discard(conn)
        try:
            self._idle[conn.node_id].remove(conn)
        except ValueError:
            pass
        if not self._all[conn.node_id]:
            self._all.pop(conn.node_id, None)
            self._idle.pop(conn.node_id, None)

    async def acquire(self, node_id: str, timeout: float | None) -> TunnelConn:
        """Borrow a parked connection to `node_id`, waiting up to `timeout` for the worker
        to dial one. Raises httpcore.ConnectTimeout if none arrives."""
        idle = self._idle[node_id]
        while idle:
            conn = idle.popleft()
            if not conn.closed.is_set():
                conn.borrowed = True
                return conn
        fut: asyncio.Future[TunnelConn] = asyncio.get_running_loop().create_future()
        self._waiters[node_id].append(fut)
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise httpcore.ConnectTimeout(f"no tunnel connection from node {node_id}") from None
        finally:
            if fut in self._waiters[node_id]:
                self._waiters[node_id].remove(fut)

    def connected(self, node_id: str) -> int:
        return sum(1 for c in self._all.get(node_id, ()) if not c.closed.is_set())

    def stats(self, node_id: str) -> dict:
        conns = [c for c in self._all.get(node_id, ()) if not c.closed.is_set()]
        return {
            "tunnel_conns": len(conns),
            "tunnel_idle": sum(1 for c in conns if not c.borrowed),
        }

    async def close_node(self, node_id: str) -> int:
        """Close every tunnel connection of a node (e.g. revoked). Returns how many."""
        conns = list(self._all.get(node_id, ()))
        for conn in conns:
            await conn.close()
        return len(conns)


class TunnelStream(httpcore.AsyncNetworkStream):
    """httpcore's view of a borrowed tunnel connection."""

    def __init__(self, conn: TunnelConn) -> None:
        self.conn = conn

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        try:
            return await asyncio.wait_for(self.conn.read(max_bytes), timeout)
        except asyncio.TimeoutError:
            raise httpcore.ReadTimeout("tunnel read timed out") from None

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if not buffer:
            return
        try:
            await asyncio.wait_for(self.conn.write(buffer), timeout)
        except asyncio.TimeoutError:
            raise httpcore.WriteTimeout("tunnel write timed out") from None
        except (ConnectionError, RuntimeError, WebSocketDisconnect) as exc:
            raise httpcore.WriteError(f"tunnel connection lost: {exc}") from exc

    async def aclose(self) -> None:
        await self.conn.close()

    async def start_tls(self, *args, **kwargs) -> httpcore.AsyncNetworkStream:
        raise NotImplementedError("tunnel connections carry plain HTTP; TLS ends at the manager")

    def get_extra_info(self, info: str):
        if info == "is_readable":
            return self.conn.readable
        return None


class TunnelNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolves ``<node_id>.tunnel`` to a borrowed tunnel connection."""

    def __init__(self, tunnels: TunnelRegistry) -> None:
        self.tunnels = tunnels

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        if not host.endswith(TUNNEL_HOST_SUFFIX):
            raise httpcore.ConnectError(f"not a tunnel host: {host}")
        conn = await self.tunnels.acquire(host[: -len(TUNNEL_HOST_SUFFIX)], timeout)
        return TunnelStream(conn)

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise NotImplementedError

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class TunnelTransport(httpx.AsyncHTTPTransport):
    """An httpx transport whose connections are tunnel connections.

    httpx doesn't expose httpcore's `network_backend`, so the pool AsyncHTTPTransport
    builds is swapped for one on `TunnelNetworkBackend`; request/response mapping and
    error translation stay httpx's own. HTTP/1.1 only — one request per connection at a
    time is exactly the one-request-per-tunnel-connection model.
    """

    def __init__(self, tunnels: TunnelRegistry, limits: httpx.Limits) -> None:
        super().__init__(limits=limits)
        self._pool = httpcore.AsyncConnectionPool(
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=True,
            http2=False,
            network_backend=TunnelNetworkBackend(tunnels),
        )
