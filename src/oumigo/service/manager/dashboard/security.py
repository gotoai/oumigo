"""The console's sessions and browser protections.

- **Session cookie** (`oumigo_console_session`): a signed, not encrypted, JSON payload
  with the login (empty while sign-in is off), a random nonce, and the times it was
  issued and last used. A signed-in session also carries a fingerprint of the password
  hash, so changing the hash ends it. The key is derived from the manager's auth-store
  signing secret, so sessions survive a console restart without a secret of their own.
- **CSRF token**: an HMAC of the session nonce, sent with every state-changing request
  (a form field or the `X-CSRF-Token` header) and checked with the `Origin` header.
- **Host check**: only requests addressed to localhost are answered,
  which stops DNS rebinding while an SSH tunnel is open.
- **Sign-in lockout**: 5 wrong passwords in a row lock sign-in for 15 minutes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from oumigo.service.manager.posture import is_loopback  # noqa: F401 - re-exported

COOKIE_NAME = "oumigo_console_session"
CSRF_HEADER = "x-csrf-token"
CSRF_FIELD = "csrf_token"
IDLE_LIMIT_S = 2 * 3600
MAX_FAILURES = 5
LOCK_S = 15 * 60
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def host_of(header: str | None) -> str:
    """The host name in a `Host` header, without the port."""
    if not header:
        return ""
    return urlparse(f"//{header}").hostname or ""


@dataclass
class Session:
    login: str
    nonce: str
    issued: float
    last: float
    fingerprint: str = ""


class Sessions:
    """Signs and checks session cookies and CSRF tokens with one key."""

    def __init__(self, key: bytes) -> None:
        self._key = key

    @classmethod
    def from_secret(cls, secret: bytes) -> Sessions:
        return cls(hmac.new(secret, b"console-session", hashlib.sha256).digest())

    def _mac(self, purpose: str, data: bytes) -> str:
        return hmac.new(self._key, purpose.encode() + b":" + data, hashlib.sha256).hexdigest()

    def fingerprint(self, password_hash: str) -> str:
        return self._mac("password", password_hash.encode())[:16]

    def new(self, login: str = "", fingerprint: str = "") -> Session:
        now = time.time()
        return Session(login, secrets.token_hex(16), now, now, fingerprint)

    def encode(self, session: Session) -> str:
        payload = base64.urlsafe_b64encode(json.dumps(
            {"u": session.login, "n": session.nonce, "i": session.issued,
             "l": session.last, "f": session.fingerprint},
            separators=(",", ":")).encode()).decode().rstrip("=")
        return f"{payload}.{self._mac('cookie', payload.encode())}"

    def decode(self, cookie: str | None) -> Session | None:
        """The session in a cookie, or None if it is missing, forged or idle too long."""
        if not cookie or "." not in cookie:
            return None
        payload, mac = cookie.rsplit(".", 1)
        if not hmac.compare_digest(mac, self._mac("cookie", payload.encode())):
            return None
        try:
            data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            session = Session(str(data["u"]), str(data["n"]), float(data["i"]),
                              float(data["l"]), str(data.get("f", "")))
        except (ValueError, KeyError, TypeError):
            return None
        if time.time() - session.last > IDLE_LIMIT_S:
            return None
        return session

    def csrf_token(self, session: Session) -> str:
        return self._mac("csrf", session.nonce.encode())

    def check_csrf(self, session: Session | None, token: str | None) -> bool:
        return (session is not None and token is not None
                and hmac.compare_digest(token, self.csrf_token(session)))


def same_origin(origin: str | None, host_header: str | None) -> bool:
    """True when the request's `Origin` (if the browser sent one) is this server."""
    if origin is None:
        return True  # not a browser cross-site request: the CSRF token still applies
    parsed = urlparse(origin)
    return parsed.netloc != "" and parsed.netloc == (host_header or "")


class SignInLimiter:
    """Locks sign-in after too many wrong passwords in a row (one admin account, so the
    count is global)."""

    def __init__(self, max_failures: int = MAX_FAILURES, lock_s: float = LOCK_S) -> None:
        self.max_failures = max_failures
        self.lock_s = lock_s
        self.failures = 0
        self.locked_until = 0.0

    def locked(self) -> bool:
        return time.time() < self.locked_until

    def failed(self) -> None:
        self.failures += 1
        if self.failures >= self.max_failures:
            self.locked_until = time.time() + self.lock_s
            self.failures = 0

    def succeeded(self) -> None:
        self.failures = 0
