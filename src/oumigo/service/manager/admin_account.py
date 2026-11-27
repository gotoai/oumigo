"""The console's admin account: one login and a password hash, from the environment.

`MG_ADMIN_USER` and `MG_ADMIN_PASSWORD_HASH` (in the manager's `.env` or its
environment) turn the console's sign-in on. If either is missing or empty, sign-in is
off. `oumigo manager admin-password` makes the hash; the console's Create Admin Account
page writes both into `.env`.

The hash is scrypt from the standard library, written as
``scrypt:<n>:<r>:<p>:<salt hex>:<key hex>``: colons and hex only, so it needs no quoting
in a `.env` file, a shell or a systemd `EnvironmentFile`.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import string
from dataclasses import dataclass

from oumigo.common.envnames import MG_ADMIN_PASSWORD_HASH, MG_ADMIN_USER

ENV_ADMIN_USER = MG_ADMIN_USER
ENV_ADMIN_PASSWORD_HASH = MG_ADMIN_PASSWORD_HASH

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1  # about 16 MiB per check
SALT_BYTES, KEY_BYTES = 16, 32
_SCRYPT_MAXMEM = 64 * 1024 * 1024

LOGIN_RE = re.compile(r"^[a-z0-9._-]{3,40}$")
PASSWORD_MIN, PASSWORD_MAX = 10, 128
SYMBOLS = set(string.punctuation)  # the printable ASCII symbols


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    key = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return f"scrypt:{SCRYPT_N}:{SCRYPT_R}:{SCRYPT_P}:{salt.hex()}:{key.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """True if `password` matches `encoded`; False for a wrong password or a hash this
    code can't read."""
    try:
        scheme, n, r, p, salt, key = encoded.split(":")
        if scheme != "scrypt":
            return False
        expected = bytes.fromhex(key)
        actual = _scrypt(password, bytes.fromhex(salt), int(n), int(r), int(p),
                         len(expected))
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)


def is_password_hash(value: str) -> bool:
    parts = value.split(":")
    return len(parts) == 6 and parts[0] == "scrypt" and all(parts[1:])


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int,
            length: int = KEY_BYTES) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p,
                          maxmem=_SCRYPT_MAXMEM, dklen=length)


def login_problems(login: str) -> list[str]:
    if LOGIN_RE.match(login):
        return []
    rule = ("The login name must be 3 to 40 characters: lowercase letters, digits, "
            "'.', '_' or '-'.")
    return [rule]


def password_problems(password: str) -> list[str]:
    """What the password lacks, as sentences; empty when it meets the rule: 10 to 128
    printable ASCII characters (no spaces) with a lowercase letter, an uppercase letter, a
    digit and a symbol."""
    problems: list[str] = []
    bad = sorted({c for c in password if not ("!" <= c <= "~")})
    if bad:
        shown = ", ".join("space" if c == " " else repr(c) for c in bad)
        problems.append(f"It can't contain {shown}: use printable ASCII characters only, "
                        "without spaces.")
    if len(password) < PASSWORD_MIN:
        problems.append(f"It needs at least {PASSWORD_MIN} characters.")
    if len(password) > PASSWORD_MAX:
        problems.append(f"It can have at most {PASSWORD_MAX} characters.")
    for label, present in (
        ("a lowercase letter", any(c.islower() and c.isascii() for c in password)),
        ("an uppercase letter", any(c.isupper() and c.isascii() for c in password)),
        ("a digit", any(c.isdigit() and c.isascii() for c in password)),
        ("a symbol (one of " + "".join(sorted(SYMBOLS)) + ")",
         any(c in SYMBOLS for c in password)),
    ):
        if not present:
            problems.append(f"It needs {label}.")
    return problems


@dataclass
class AdminAccount:
    """The configured admin, or none (sign-in off)."""

    login: str = ""
    password_hash: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.login and self.password_hash)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> AdminAccount:
        environ = os.environ if environ is None else environ
        return cls(environ.get(ENV_ADMIN_USER, "").strip(),
                   environ.get(ENV_ADMIN_PASSWORD_HASH, "").strip())
