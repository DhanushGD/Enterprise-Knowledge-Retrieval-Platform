"""Step 11 core: everything the HTTP layer does, as plain Python so it is fully unit-tested.

    login      rate-limit (per IP and per username) -> verify password -> JWT
    authenticate  JWT -> user row -> role (looked up every time) -> Principal
    ask        validate -> rate-limit -> cache -> [queue for a generation slot] -> answer

serving/app.py only maps URLs onto these methods. Errors that should become HTTP statuses are
`ApiError`s; once a stream has started, problems are sent as an `error` event instead.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterator

from serving.cache import CACHEABLE_STATUSES, AnswerCache, cache_key, permission_fingerprint
from serving.concurrency import Busy, GenerationGate
from serving.generation import Answer, Source
from serving.metrics import NullMetrics
from serving.ratelimit import Limit, MemoryRateLimiter
from serving.security import (DUMMY_HASH, AuthError, decode_token, issue_token, verify_password)
from serving.users import normalize_username

audit = logging.getLogger("rag.audit")
log = logging.getLogger("rag.api")

PUBLIC_ABSTAIN_REASONS = {"no_results", "low_relevance", "model_declined", "empty_model_output"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after

    @property
    def headers(self) -> dict:
        return {"Retry-After": str(self.retry_after)} if self.retry_after else {}


@dataclass(frozen=True)
class Principal:
    username: str
    role: str


# --------------------------------------------------------------- public payloads
def source_dict(s: Source) -> dict:
    return {"n": s.n, "title": s.doc_title, "section": s.section, "path": s.source_path,
            "snippet": s.snippet}


def public_answer(a: Answer, cached: bool = False) -> dict:
    """What a client may see. Deliberately omits `raw_text` (unvalidated model output, may be
    invented), internal error text (hostnames, stack hints) and similarity scores."""
    return {
        "status": a.status,
        "text": a.text,
        "citations": [source_dict(s) for s in a.citations],
        "related": [source_dict(s) for s in a.sources] if a.status in ("ungrounded", "llm_unavailable") else [],
        "reason": a.reason if a.status == "abstained" and a.reason in PUBLIC_ABSTAIN_REASONS else None,
        "cached": cached,
        "timings_ms": dict(a.timings),
    }


def grafana_links(url: str) -> dict | None:
    """Validate the configured Grafana address; return what the admin page needs, or None.
    Only an http(s) origin is accepted, because the origin is placed into the CSP frame-src."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.username or parts.password:
        return None
    origin = f"{parts.scheme}://{parts.netloc}"
    return {"origin": origin,
            "dashboard_url": f"{origin}/d/ekrp-overview/ekrp-overview?orgId=1&kiosk&refresh=15s&from=now-6h&to=now"}


def sse(event: dict) -> str:
    """One Server-Sent Event frame. JSON never contains a raw newline, so one data line is safe."""
    import json

    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


class ApiService:
    def __init__(
        self,
        answers,
        users,
        roles: dict,
        *,
        secret: str,
        token_minutes: int = 60,
        cache: AnswerCache | None = None,
        limiter=None,
        gate: GenerationGate | None = None,
        corpus_version: Callable[[], str] = lambda: "0",
        model_id: str = "",
        ask_limit: Limit = Limit(12, 60),
        login_limit: Limit = Limit(5, 60),
        ip_login_limit: Limit = Limit(20, 60),
        max_question_chars: int = 500,
        readiness_checks: dict[str, Callable[[], None]] | None = None,
        warm_up: Callable[[], None] | None = None,
        metrics=None,
        admin_roles: tuple[str, ...] = ("admin",),
        grafana_url: str = "",
    ):
        self.metrics = metrics or NullMetrics()
        self.admin_roles = tuple(admin_roles)
        self.grafana = grafana_links(grafana_url)
        self.answers, self.users, self.roles = answers, users, roles
        self.secret, self.token_minutes = secret, token_minutes
        self.cache, self.gate = cache, gate
        self.limiter = limiter or MemoryRateLimiter()
        self.corpus_version, self.model_id = corpus_version, model_id
        self.ask_limit, self.login_limit, self.ip_login_limit = ask_limit, login_limit, ip_login_limit
        self.max_question_chars = max_question_chars
        self.readiness_checks = readiness_checks or {}
        self.warm_up = warm_up
        self._fingerprints = {name: permission_fingerprint(cfg) for name, cfg in roles.items()}

    # ------------------------------------------------------------------- auth
    def login(self, username: str, password: str, client_ip: str = "unknown") -> dict:
        try:
            result = self._login(username, password, client_ip)
        except ApiError as exc:
            self.metrics.login("throttled" if exc.code == "rate_limited" else "failed")
            raise
        self.metrics.login("ok")
        return result

    def _login(self, username: str, password: str, client_ip: str) -> dict:
        name = normalize_username(username)[:64]
        for key, lim in ((f"login:ip:{client_ip}", self.ip_login_limit),
                         (f"login:user:{name}", self.login_limit)):
            d = self.limiter.hit(key, lim.limit, lim.window)
            if not d.allowed:
                audit.warning("login_throttled user=%s ip=%s", name, client_ip)
                raise ApiError(429, "rate_limited", "Too many login attempts. Try again later.",
                               d.retry_after)

        user = self.users.get(name) if name else None
        # always run one scrypt verification, so timing doesn't reveal whether the user exists
        password_ok = verify_password(password, user.password_hash if user else DUMMY_HASH)
        if not (user and password_ok and not user.disabled):
            audit.warning("login_failed user=%s ip=%s", name, client_ip)
            raise ApiError(401, "invalid_credentials", "Invalid username or password.")

        token, expires_in = issue_token(user.username, self.secret, self.token_minutes)
        audit.info("login_ok user=%s role=%s", user.username, user.role)
        return {"access_token": token, "token_type": "bearer", "expires_in": expires_in,
                "username": user.username, "role": user.role}

    def authenticate(self, token: str | None) -> Principal:
        try:
            return self._authenticate(token)
        except ApiError as exc:
            self.metrics.rejected("auth", exc.code)
            raise

    def _authenticate(self, token: str | None) -> Principal:
        if not token:
            raise ApiError(401, "not_authenticated", "Log in to continue.")
        try:
            username = decode_token(token, self.secret)
        except AuthError:
            raise ApiError(401, "invalid_token", "Your session is invalid or has expired.") from None
        user = self.users.get(username)           # role is read fresh: revocation is immediate
        if not user or user.disabled:
            raise ApiError(401, "invalid_token", "Your session is invalid or has expired.")
        if user.role not in self.roles:
            raise ApiError(403, "role_not_allowed", "Your account has no access configured.")
        return Principal(user.username, user.role)

    # -------------------------------------------------------------------- ask
    def ask(self, principal: Principal, question: str) -> dict:
        """Non-streaming: returns the final answer payload."""
        final = None
        for event in self.ask_stream(principal, question):
            if event["type"] == "error":
                status = 503 if event["code"] == "busy" else 500
                raise ApiError(status, event["code"], event["message"], event.get("retry_after"))
            if event["type"] == "done":
                final = event["answer"]
        return final

    def ask_stream(self, principal: Principal, question: str) -> Iterator[dict]:
        """Checks that can fail with an HTTP status run NOW (before streaming begins); the
        returned generator does the slow work. Events: sources, token, done, or error."""
        try:
            return self._ask_stream(principal, question)
        except ApiError as exc:
            self.metrics.rejected("ask", exc.code)
            raise

    def _ask_stream(self, principal: Principal, question: str) -> Iterator[dict]:
        question = (question or "").strip()
        if not question:
            raise ApiError(422, "question_required", "Please enter a question.")
        if len(question) > self.max_question_chars:
            raise ApiError(422, "question_too_long",
                           f"Questions are limited to {self.max_question_chars} characters.")

        d = self.limiter.hit(f"ask:{principal.username}", self.ask_limit.limit, self.ask_limit.window)
        if not d.allowed:
            raise ApiError(429, "rate_limited", "You're asking too quickly. Please wait a moment.",
                           d.retry_after)

        began = time.perf_counter()
        key = None
        if self.cache:
            key = cache_key(question, self._fingerprints[principal.role], self.corpus_version(),
                            self.model_id)
            hit = self.cache.get(key)
            self.metrics.cache(bool(hit))
            if hit:
                return self._replay(principal, question, hit, began)

        if self.gate and self.gate.is_saturated():
            raise ApiError(503, "busy", "The assistant is busy right now. Please try again shortly.", 10)
        return self._live(principal, question, key, began)

    # -------------------------------------------------------------- internals
    def _replay(self, principal, question, hit: dict, began: float) -> Iterator[dict]:
        payload = dict(hit["answer"], cached=True)
        payload["timings_ms"] = {"total_ms": round((time.perf_counter() - began) * 1000, 1)}
        yield {"type": "sources", "sources": hit["sources"]}
        yield {"type": "token", "text": payload["text"]}
        self._audit(principal, question, payload["status"], True, payload["timings_ms"]["total_ms"])
        self.metrics.answered(principal.role, payload["status"], payload.get("reason"), True,
                              payload["timings_ms"])
        yield {"type": "done", "answer": payload}

    def _live(self, principal, question, key, began) -> Iterator[dict]:
        slot = self.gate.acquire() if self.gate else contextlib.nullcontext()
        try:
            with slot:       # taken lazily: a stream that is never read can't leak a slot
                for event in self.answers.stream(question, principal.role):
                    if event["type"] == "sources":
                        yield {"type": "sources", "sources": [source_dict(s) for s in event["sources"]]}
                    elif event["type"] == "token":
                        yield event
                    elif event["type"] == "done":
                        answer = event["answer"]
                        payload = public_answer(answer)
                        if key and self.cache and answer.status in CACHEABLE_STATUSES:
                            self.cache.set(key, {"answer": {**payload, "timings_ms": {}},
                                                 "sources": [source_dict(s) for s in answer.sources]})
                        self._audit(principal, question, answer.status, False,
                                    answer.timings.get("total_ms"))
                        self.metrics.answered(principal.role, answer.status, answer.reason, False,
                                              answer.timings, answer.prompt_tokens,
                                              answer.completion_tokens)
                        yield {"type": "done", "answer": payload}
        except Busy:
            self.metrics.stream_error("busy")
            yield {"type": "error", "code": "busy", "retry_after": 10,
                   "message": "The assistant is busy right now. Please try again shortly."}
        except Exception:
            log.exception("ask failed user=%s", principal.username)
            self.metrics.stream_error("internal")
            yield {"type": "error", "code": "internal", "message": "Something went wrong. Please try again."}

    @staticmethod
    def _audit(principal, question, status, cached, total_ms) -> None:
        # The question text itself is NOT logged: it can contain customer data.
        audit.info("ask user=%s role=%s status=%s cached=%s total_ms=%s qhash=%s qlen=%d",
                   principal.username, principal.role, status, cached, total_ms,
                   hashlib.sha256(question.encode()).hexdigest()[:12], len(question))

    # ------------------------------------------------------------------ admin
    def admin_summary(self, principal: Principal) -> dict:
        """Aggregate operating numbers for the admin page. Admin roles only; the permission is
        checked here, not in the route, so it cannot be forgotten when a route is added."""
        if principal.role not in self.admin_roles:
            self.metrics.rejected("auth", "forbidden")
            raise ApiError(403, "forbidden", "This page is for administrators.")
        return {
            "metrics": self.metrics.snapshot(),
            "controls": {
                "ask_limit": f"{self.ask_limit.limit} per {self.ask_limit.window}s per user",
                "login_limit": f"{self.login_limit.limit} per {self.login_limit.window}s per username",
                "ip_login_limit": f"{self.ip_login_limit.limit} per {self.ip_login_limit.window}s per IP",
                "limiter_backend": type(getattr(self.limiter, "inner", self.limiter)).__name__,
                "cache_backend": type(self.cache.backend).__name__ if self.cache else "off",
                "max_concurrent_generations": getattr(self.gate, "max_concurrent", None),
            },
            "readiness": self.readiness(),
            "grafana": self.grafana,
        }

    # -------------------------------------------------------------- readiness
    def readiness(self) -> dict[str, str]:
        out = {}
        for name, check in self.readiness_checks.items():
            try:
                check()
                out[name] = "ok"
            except Exception as exc:
                out[name] = f"{type(exc).__name__}"
        return out
