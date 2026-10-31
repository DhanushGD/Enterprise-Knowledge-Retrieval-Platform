"""Answer cache. An answer costs 15-30 s of CPU, so repeated questions should be free.

The dangerous part of caching RAG answers is *who may see what*. The key therefore contains:
  * the caller's PERMISSION SET (not their name): two roles that see exactly the same documents
    share entries; a role with wider access never reuses another role's narrower answer or
    vice versa, so an answer built from restricted pages can't be replayed to someone who may
    not see them;
  * the CORPUS VERSION: a sync (edit, delete, add) changes it, so stale or deleted content
    stops being served immediately;
  * the model + prompt + settings fingerprint, so changing any of them starts fresh.
Only good outcomes are cached (answered / abstained) - never outages or ungrounded answers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata

log = logging.getLogger("rag.cache")

CACHEABLE_STATUSES = {"answered", "abstained"}


def normalize_question(question: str) -> str:
    q = unicodedata.normalize("NFKC", question).lower()
    q = re.sub(r"\s+", " ", q).strip()
    return q.rstrip("?!. ").strip()


def permission_fingerprint(role_cfg: dict) -> str:
    perms = {
        "levels": sorted(role_cfg.get("allowed_access_levels") or []),
        "departments": sorted(role_cfg.get("allowed_departments") or []),
    }
    return hashlib.sha256(json.dumps(perms, sort_keys=True).encode()).hexdigest()[:16]


def cache_key(question: str, fingerprint: str, corpus_version: str, model_id: str) -> str:
    raw = json.dumps([normalize_question(question), fingerprint, corpus_version, model_id])
    return "ans:" + hashlib.sha256(raw.encode()).hexdigest()


class MemoryTTLCache:
    def __init__(self, clock=time.monotonic, max_items: int = 1000):
        self._clock, self._max = clock, max_items
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            if item[0] <= self._clock():
                del self._data[key]
                return None
            return item[1]

    def set(self, key: str, value: str, ttl: int) -> None:
        with self._lock:
            if len(self._data) >= self._max:
                now = self._clock()
                for k in [k for k, (exp, _) in self._data.items() if exp <= now]:
                    del self._data[k]
                while len(self._data) >= self._max:          # still full: drop the oldest
                    del self._data[next(iter(self._data))]
            self._data[key] = (self._clock() + ttl, value)


class RedisTTLCache:
    def __init__(self, client):
        self.client = client

    def get(self, key: str) -> str | None:
        value = self.client.get(key)
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def set(self, key: str, value: str, ttl: int) -> None:
        self.client.set(key, value, ex=ttl)


class AnswerCache:
    """JSON in, JSON out, and never an exception: a cache problem is just a cache miss."""

    def __init__(self, backend, ttl: int = 3600):
        self.backend, self.ttl = backend, ttl

    def get(self, key: str) -> dict | None:
        try:
            raw = self.backend.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            log.warning("cache read failed (%s: %s); treating as a miss", type(exc).__name__, exc)
            return None

    def set(self, key: str, value: dict) -> None:
        try:
            self.backend.set(key, json.dumps(value), self.ttl)
        except Exception as exc:
            log.warning("cache write failed (%s: %s)", type(exc).__name__, exc)


class TTLValue:
    """Call an (expensive) function at most once per `ttl` seconds - for the corpus version,
    which would otherwise be a database query on every request."""

    def __init__(self, fn, ttl: float = 30.0, clock=time.monotonic):
        self.fn, self.ttl, self._clock = fn, ttl, clock
        self._value, self._at = None, None
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            now = self._clock()
            if self._at is None or now - self._at >= self.ttl:
                self._value, self._at = self.fn(), now
            return self._value
