"""Rate limiting: how many requests a key (user, IP) may make per time window.

  MemoryRateLimiter  sliding window, one process. Fine for a single uvicorn worker.
  RedisRateLimiter   fixed window shared by every worker/instance (atomic Lua script).
  SafeLimiter        wraps either: if the backend errors, REQUESTS ARE ALLOWED and the error is
                     logged - a Redis blip must not lock every user out of the assistant.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

log = logging.getLogger("rag.ratelimit")


@dataclass(frozen=True)
class Limit:
    limit: int
    window: int          # seconds


@dataclass(frozen=True)
class Decision:
    allowed: bool
    remaining: int
    retry_after: int = 0     # seconds until a request would be allowed (0 when allowed)


class MemoryRateLimiter:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def hit(self, key: str, limit: int, window: int) -> Decision:
        now = self._clock()
        with self._lock:
            self._calls += 1
            if self._calls % 500 == 0:
                self._sweep(now, window)
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= now - window:
                q.popleft()
            if len(q) >= limit:
                return Decision(False, 0, max(1, math.ceil(q[0] + window - now)))
            q.append(now)
            return Decision(True, limit - len(q))

    def _sweep(self, now: float, window: int) -> None:
        for key in [k for k, q in self._hits.items() if not q or q[-1] <= now - window]:
            del self._hits[key]


_LUA = """
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
local ttl = redis.call('TTL', KEYS[1])
if ttl < 0 then redis.call('EXPIRE', KEYS[1], ARGV[1]); ttl = tonumber(ARGV[1]) end
return {n, ttl}
"""


class RedisRateLimiter:
    """Fixed window: simple and atomic, at the cost of allowing up to 2x the limit across a
    window boundary. The script also repairs a counter that somehow lost its expiry."""

    def __init__(self, client, prefix: str = "rl:"):
        self.client, self.prefix = client, prefix

    def hit(self, key: str, limit: int, window: int) -> Decision:
        n, ttl = self.client.eval(_LUA, 1, f"{self.prefix}{key}", window)
        n, ttl = int(n), int(ttl)
        if n > limit:
            return Decision(False, 0, max(1, ttl))
        return Decision(True, limit - n)


class SafeLimiter:
    def __init__(self, inner):
        self.inner = inner
        self.failures = 0

    def hit(self, key: str, limit: int, window: int) -> Decision:
        try:
            return self.inner.hit(key, limit, window)
        except Exception as exc:
            self.failures += 1
            log.warning("rate limiter backend failed (%s: %s); allowing request",
                        type(exc).__name__, exc)
            return Decision(True, limit)
