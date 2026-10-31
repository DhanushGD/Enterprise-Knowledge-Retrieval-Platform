import json
import threading
import time

import pytest

from serving.cache import (AnswerCache, MemoryTTLCache, RedisTTLCache, TTLValue, cache_key,
                           normalize_question, permission_fingerprint)
from serving.concurrency import Busy, GenerationGate
from serving.ratelimit import MemoryRateLimiter, RedisRateLimiter, SafeLimiter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- rate limiting
def test_limiter_allows_up_to_the_limit_then_says_when_to_retry():
    clock = Clock()
    rl = MemoryRateLimiter(clock)
    assert [rl.hit("u", 3, 60).allowed for _ in range(3)] == [True, True, True]
    blocked = rl.hit("u", 3, 60)
    assert not blocked.allowed and blocked.remaining == 0 and blocked.retry_after == 60
    clock.t += 45
    assert rl.hit("u", 3, 60).retry_after == 15


def test_window_slides_and_keys_are_independent():
    clock = Clock()
    rl = MemoryRateLimiter(clock)
    for _ in range(3):
        rl.hit("a", 3, 60)
    assert not rl.hit("a", 3, 60).allowed and rl.hit("b", 3, 60).allowed
    clock.t += 61
    assert rl.hit("a", 3, 60).allowed


def test_remaining_counts_down():
    rl = MemoryRateLimiter(Clock())
    assert [rl.hit("u", 3, 60).remaining for _ in range(3)] == [2, 1, 0]


def test_idle_keys_are_swept_so_memory_does_not_grow_forever():
    clock = Clock()
    rl = MemoryRateLimiter(clock)
    for i in range(300):
        rl.hit(f"ip{i}", 5, 60)
    clock.t += 120
    for _ in range(300):
        rl.hit("busy", 5, 60)
    assert len(rl._hits) < 50


class FakeRedis:
    """Emulates what the Lua script does, plus GET/SET for the cache."""

    def __init__(self):
        self.counts, self.ttls, self.kv = {}, {}, {}

    def eval(self, script, nkeys, key, window):
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] == 1 or self.ttls.get(key, -1) < 0:
            self.ttls[key] = int(window)
        return [self.counts[key], self.ttls[key]]

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value, ex=None):
        self.kv[key] = value.encode() if isinstance(value, str) else value


def test_redis_limiter_counts_and_reports_the_remaining_ttl():
    rl = RedisRateLimiter(FakeRedis())
    assert [rl.hit("k", 2, 30).allowed for _ in range(2)] == [True, True]
    d = rl.hit("k", 2, 30)
    assert not d.allowed and d.retry_after == 30


def test_a_broken_limiter_backend_lets_requests_through():
    class Down:
        def hit(self, *a):
            raise ConnectionError("redis down")

    safe = SafeLimiter(Down())
    assert safe.hit("k", 1, 60).allowed and safe.failures == 1


# -------------------------------------------------------------------- cache keys
def test_question_normalisation_ignores_case_spacing_and_trailing_punctuation():
    variants = ["How do I reset a password?", "  how do I   reset a password ", "HOW DO I RESET A PASSWORD?!"]
    assert len({normalize_question(v) for v in variants}) == 1
    assert normalize_question("a b") != normalize_question("a c")


ADMIN = {"allowed_access_levels": ["public", "internal", "restricted"], "allowed_departments": ["support", "hr"]}
AGENT = {"allowed_access_levels": ["public"], "allowed_departments": ["support"]}
AGENT_TWIN = {"allowed_departments": ["support"], "allowed_access_levels": ["public"]}   # same, reordered


def test_key_depends_on_permissions_not_on_role_name_or_list_order():
    fa, ft, fadm = map(permission_fingerprint, (AGENT, AGENT_TWIN, ADMIN))
    assert fa == ft and fa != fadm
    k = lambda fp, ver="1", model="m": cache_key("q?", fp, ver, model)      # noqa: E731
    assert k(fa) == k(ft) and k(fa) != k(fadm)
    assert k(fa) != k(fa, ver="2") and k(fa) != k(fa, model="m2")


# ----------------------------------------------------------------- cache backends
def test_memory_cache_expires_entries_and_evicts_when_full():
    clock = Clock()
    c = MemoryTTLCache(clock, max_items=3)
    c.set("a", "1", 10)
    assert c.get("a") == "1"
    clock.t += 11
    assert c.get("a") is None
    for i in range(5):
        c.set(f"k{i}", "v", 100)
    assert len(c._data) <= 3 and c.get("k4") == "v" and c.get("k0") is None


def test_answer_cache_roundtrips_json_through_redis_bytes():
    cache = AnswerCache(RedisTTLCache(FakeRedis()), ttl=60)
    cache.set("k", {"answer": {"text": "héllo"}, "sources": [1, 2]})
    assert cache.get("k") == {"answer": {"text": "héllo"}, "sources": [1, 2]}
    assert cache.get("missing") is None


def test_cache_backend_failures_are_misses_not_errors():
    class Down:
        def get(self, k):
            raise ConnectionError()

        def set(self, k, v, ttl):
            raise ConnectionError()

    cache = AnswerCache(Down())
    cache.set("k", {"a": 1})
    assert cache.get("k") is None


def test_corrupt_cache_entry_is_a_miss():
    backend = MemoryTTLCache()
    backend.set("k", "{not json", 60)
    assert AnswerCache(backend).get("k") is None


def test_ttl_value_refreshes_only_after_the_ttl():
    clock, calls = Clock(), []

    def fetch():
        calls.append(1)
        return len(calls)

    v = TTLValue(fetch, ttl=30, clock=clock)
    assert (v(), v()) == (1, 1) and len(calls) == 1
    clock.t += 31
    assert v() == 2


# ------------------------------------------------------------------- generation gate
def test_gate_counts_slots_and_releases_even_when_the_work_fails():
    gate = GenerationGate(max_concurrent=1, max_waiting=1, queue_timeout=1)
    with pytest.raises(RuntimeError):
        with gate.acquire():
            assert gate.in_flight == 1
            raise RuntimeError("boom")
    assert gate.in_flight == 0 and gate.waiting == 0


def test_waiter_gets_the_slot_when_it_frees_up():
    gate = GenerationGate(1, 2, queue_timeout=5)
    order, release = [], threading.Event()

    def first():
        with gate.acquire():
            order.append("first-in")
            release.wait(2)
        order.append("first-out")

    t = threading.Thread(target=first)
    t.start()
    while "first-in" not in order:
        time.sleep(0.005)
    threading.Timer(0.05, release.set).start()
    with gate.acquire():
        order.append("second-in")
    t.join()
    assert order == ["first-in", "first-out", "second-in"]


def test_waiting_too_long_raises_busy_and_does_not_leak_the_waiting_count():
    gate = GenerationGate(1, 2, queue_timeout=0.05)
    with gate.acquire():
        with pytest.raises(Busy):
            with gate.acquire():
                pass
        assert gate.waiting == 0
    assert gate.in_flight == 0


def test_saturated_only_when_all_slots_are_busy_and_the_waiting_room_is_full():
    gate = GenerationGate(1, 1, queue_timeout=5)
    assert not gate.is_saturated()
    done = threading.Event()
    held = threading.Event()

    def hold():
        with gate.acquire():
            held.set()
            done.wait(3)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(1)
    assert not gate.is_saturated()                      # busy, but nobody waiting yet
    waiter = threading.Thread(target=lambda: gate.acquire().__enter__() and None)
    waiter.daemon = True
    waiter.start()
    deadline = time.time() + 2
    while gate.waiting < 1 and time.time() < deadline:
        time.sleep(0.005)
    assert gate.is_saturated()
    done.set()
    holder.join()
