"""Passwords and tokens.

  * Passwords: salted scrypt (memory-hard, in the standard library: no extra dependency).
  * Tokens: HS256 JWT via PyJWT. The token carries ONLY who the user is (`sub`); their role is
    looked up on every request, so changing a role or disabling an account takes effect at once
    instead of whenever the old token expires.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

import jwt

ISSUER = "ekrp"
ALGORITHM = "HS256"
DEFAULT_SECRETS = {"", "change-me-in-real-use"}

_N, _R, _P, _DKLEN = 2 ** 14, 8, 1, 32


class AuthError(Exception):
    """Token missing, malformed, expired or forged."""


class ConfigError(RuntimeError):
    """Unsafe configuration (e.g. the default JWT secret in production)."""


# ----------------------------------------------------------------- passwords
def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    """False (never an exception) for a wrong password OR a malformed stored hash."""
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
            dklen=len(base64.b64decode(expected)), maxmem=128 * 1024 * 1024,
        )
        return hmac.compare_digest(digest, base64.b64decode(expected))
    except (ValueError, TypeError):
        return False


# Verified against when the username doesn't exist, so "no such user" costs the same time as
# "wrong password" and the login endpoint can't be used to discover valid usernames.
DUMMY_HASH = hash_password("not-a-real-password")


# -------------------------------------------------------------------- tokens
def issue_token(username: str, secret: str, minutes: int = 60, now: float | None = None
                ) -> tuple[str, int]:
    issued = int(now if now is not None else time.time())
    expires = issued + minutes * 60
    token = jwt.encode(
        {"sub": username, "iss": ISSUER, "iat": issued, "exp": expires}, secret, algorithm=ALGORITHM
    )
    return token, minutes * 60


def decode_token(token: str, secret: str) -> str:
    """Return the username, or raise AuthError. The algorithm is pinned, so a forged
    `alg: none` token (or one signed with a different algorithm) is rejected."""
    try:
        claims = jwt.decode(
            token, secret, algorithms=[ALGORITHM], issuer=ISSUER, leeway=5,
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError(type(exc).__name__) from exc
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise AuthError("bad subject")
    return sub


def check_secret(secret: str, app_env: str) -> list[str]:
    """Warnings for a weak JWT secret; raises ConfigError instead when app_env is 'production'."""
    problems = []
    if secret in DEFAULT_SECRETS:
        problems.append("JWT_SECRET is the built-in default: anyone can forge tokens")
    elif len(secret) < 32:
        problems.append("JWT_SECRET is shorter than 32 characters")
    if problems and app_env.lower() == "production":
        raise ConfigError("; ".join(problems) + ". Set a strong JWT_SECRET "
                          "(python -c \"import secrets; print(secrets.token_urlsafe(48))\").")
    return problems
