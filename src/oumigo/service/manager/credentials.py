"""Worker credentials: join tokens, per-node tokens, and backend API keys.

Three credentials, each with one job:

- **Join token** — what an operator gives a new worker (`OUMIGO_JOIN_TOKEN`). Either the
  *reusable* fleet token (`OUMIGO_MANAGER_TOKEN` / `auth.token_file`, the v1 shared
  token) or a *minted* one from `oumigo manager token create`, which can expire and/or
  be limited to N registrations. Minted tokens are stored only as SHA-256 hashes.
- **Node token** — issued at registration, bound to one `node_id`:
  ``oumigo-node.<node_id>.<HMAC(secret, node_id)>``. The worker presents it on every
  later call (heartbeat, metrics, re-register, `/tunnel`). It is derived, not stored,
  so it survives a manager restart (the signing secret is persisted); it dies when the
  node is revoked (`oumigo manager node revoke`).
- **Backend API key** — what a tunnel worker's vLLM requires (`VLLM_API_KEY`); also
  derived from the secret, so the router can recompute it per node.

The state (signing secret, minted tokens, revoked nodes) lives in a small SQLite file in
the manager's state directory. The CLI writes to the same file on the manager host —
filesystem permissions are the admin auth — and the manager reads it per request, so a
token minted or a node revoked while the manager runs takes effect immediately.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

AUTH_DB_FILE = "manager_auth.db"
JOIN_PREFIX = "oumigo-join-"
NODE_PREFIX = "oumigo-node."

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS join_tokens (
    id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    label TEXT,
    created_at REAL NOT NULL,
    expires_at REAL,          -- NULL = never
    uses_left INTEGER,        -- NULL = unlimited
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS revoked_nodes (node_id TEXT PRIMARY KEY, revoked_at REAL NOT NULL);
"""


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def bearer(authorization: str | None) -> str | None:
    """The credential from an ``Authorization: Bearer <x>`` header, or None."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


@dataclass
class JoinToken:
    id: str
    label: str | None
    created_at: float
    expires_at: float | None
    uses_left: int | None
    revoked: bool

    @property
    def usable(self) -> bool:
        if self.revoked:
            return False
        if self.expires_at is not None and self.expires_at <= time.time():
            return False
        return self.uses_left is None or self.uses_left > 0


class AuthStore:
    """The manager's auth state in SQLite. ``path=None`` keeps it in memory (tests, and a
    manager without a writable state dir — node tokens then die with the process)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        # One connection guarded by a lock: calls are tiny and the manager's handlers run
        # on one event loop. The CLI opens its own connection; SQLite locks across processes.
        self._conn = sqlite3.connect(
            str(path) if path is not None else ":memory:", timeout=5.0, check_same_thread=False
        )
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        if path is not None:
            path.chmod(0o600)  # holds the signing secret
        self.secret = self._signing_secret()

    @classmethod
    def at(cls, state_dir: Path) -> AuthStore:
        return cls(Path(state_dir) / AUTH_DB_FILE)

    def _signing_secret(self) -> bytes:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key='signing_secret'").fetchone()
            if row is None:
                # INSERT OR IGNORE + re-read: a CLI and a manager racing on a fresh file
                # both end up with the one secret that won.
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('signing_secret', ?)",
                    (secrets.token_hex(32),),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT value FROM meta WHERE key='signing_secret'"
                ).fetchone()
        return bytes.fromhex(row[0])

    # --- join tokens -----------------------------------------------------------

    def create_join_token(
        self, ttl_s: float | None = None, uses: int | None = None, label: str | None = None
    ) -> tuple[str, str]:
        """Mint a join token. Returns (id, token); the token itself is never stored."""
        token = JOIN_PREFIX + secrets.token_urlsafe(24)
        token_id = secrets.token_hex(4)
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO join_tokens (id, token_hash, label, created_at, expires_at, uses_left)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (token_id, _hash(token), label, now, now + ttl_s if ttl_s else None,
                 uses if uses else None),
            )
            self._conn.commit()
        return token_id, token

    def list_join_tokens(self) -> list[JoinToken]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, label, created_at, expires_at, uses_left, revoked FROM join_tokens"
                " ORDER BY created_at"
            ).fetchall()
        return [JoinToken(r[0], r[1], r[2], r[3], r[4], bool(r[5])) for r in rows]

    def revoke_join_token(self, token_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE join_tokens SET revoked=1 WHERE id=?", (token_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def check_join_token(self, token: str, consume: bool) -> bool:
        """True if `token` is a usable minted join token. ``consume`` spends one use —
        atomically, so two workers can't both register on a single-use token."""
        if not token.startswith(JOIN_PREFIX):
            return False
        now = time.time()
        usable = (
            "token_hash=? AND revoked=0 AND (expires_at IS NULL OR expires_at > ?)"
            " AND (uses_left IS NULL OR uses_left > 0)"
        )
        with self._lock:
            if consume:
                cur = self._conn.execute(
                    "UPDATE join_tokens SET uses_left = uses_left - 1"
                    f" WHERE {usable} AND uses_left IS NOT NULL",
                    (_hash(token), now),
                )
                if cur.rowcount == 0:  # unlimited tokens aren't decremented; just check them
                    cur = self._conn.execute(
                        f"SELECT 1 FROM join_tokens WHERE {usable} AND uses_left IS NULL",
                        (_hash(token), now),
                    )
                    self._conn.commit()
                    return cur.fetchone() is not None
                self._conn.commit()
                return True
            row = self._conn.execute(
                f"SELECT 1 FROM join_tokens WHERE {usable}", (_hash(token), now)
            ).fetchone()
        return row is not None

    # --- nodes -----------------------------------------------------------------

    def revoke_node(self, node_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO revoked_nodes (node_id, revoked_at) VALUES (?, ?)",
                (node_id, time.time()),
            )
            self._conn.commit()

    def unrevoke_node(self, node_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM revoked_nodes WHERE node_id=?", (node_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def is_node_revoked(self, node_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM revoked_nodes WHERE node_id=?", (node_id,)
            ).fetchone()
        return row is not None

    def revoked_nodes(self) -> set[str]:
        with self._lock:
            return {r[0] for r in self._conn.execute("SELECT node_id FROM revoked_nodes")}

    # --- derived credentials ---------------------------------------------------

    def _mac(self, purpose: str, node_id: str) -> str:
        return hmac.new(self.secret, f"{purpose}:{node_id}".encode(), hashlib.sha256).hexdigest()

    def node_token(self, node_id: str) -> str:
        return f"{NODE_PREFIX}{node_id}.{self._mac('node', node_id)}"

    def node_from_token(self, token: str) -> str | None:
        """The node_id a node token was issued to, if its signature is valid (else None).
        Does not check revocation."""
        if not token.startswith(NODE_PREFIX):
            return None
        node_id, _, mac = token[len(NODE_PREFIX):].partition(".")
        if not node_id or not hmac.compare_digest(mac, self._mac("node", node_id)):
            return None
        return node_id

    def vllm_api_key(self, node_id: str) -> str:
        return "oumigo-vllm-" + self._mac("vllm", node_id)[:48]


class Authenticator:
    """Decides who may call the worker-facing endpoints.

    Auth is **enforced** under provider CUSTOM, or whenever a reusable token is set. Only
    a LAN manager with no reusable token runs fail-open (the v1 dev behavior) — and even
    then `/tunnel` needs a node token, because that is how the connection is bound to its
    node.
    """

    def __init__(self, store: AuthStore, reusable_token: str | None, wan: bool) -> None:
        self.store = store
        self.reusable_token = reusable_token
        self.enforced = wan or reusable_token is not None

    def _is_reusable(self, token: str | None) -> bool:
        return (
            token is not None
            and self.reusable_token is not None
            and hmac.compare_digest(token, self.reusable_token)
        )

    def _valid_node(self, token: str | None) -> str | None:
        """node_id for a valid, unrevoked node token."""
        if token is None:
            return None
        node_id = self.store.node_from_token(token)
        if node_id is None or self.store.is_node_revoked(node_id):
            return None
        return node_id

    def may_read_spec(self, authorization: str | None) -> bool:
        """`/spec`: any join token (not consumed) or any valid node token."""
        if not self.enforced:
            return True
        token = bearer(authorization)
        return (
            self._is_reusable(token)
            or (token is not None and self.store.check_join_token(token, consume=False))
            or self._valid_node(token) is not None
        )

    def may_register(self, authorization: str | None, node_id: str) -> bool:
        """`/register`: the node's own token (re-register), or a join token (one use spent)."""
        if self.store.is_node_revoked(node_id):
            return False
        if not self.enforced:
            return True
        token = bearer(authorization)
        if self._is_reusable(token) or self._valid_node(token) == node_id:
            return True
        return token is not None and self.store.check_join_token(token, consume=True)

    def may_act_as(self, authorization: str | None, node_id: str) -> bool:
        """`/heartbeat`, `/metrics`: the node's own token (or the reusable fleet token)."""
        if self.store.is_node_revoked(node_id):
            return False
        if not self.enforced:
            return True
        token = bearer(authorization)
        return self._is_reusable(token) or self._valid_node(token) == node_id

    def tunnel_node(self, authorization: str | None) -> str | None:
        """`/tunnel`: the node a connection belongs to — always from a node token."""
        return self._valid_node(bearer(authorization))
