"""Step 13: Prometheus metrics for the API.

What is measured, and why (every label has a small, fixed set of values, so the number of time
series stays bounded - never put usernames, questions or paths from the user into a label):

  ekrp_asks_total{role,status,cached,reason}     what happened to each question
        status: answered | abstained | ungrounded | llm_unavailable.  The ratio of abstained and
        ungrounded to answered is the closest thing to a live quality signal.
  ekrp_ask_seconds{status,cached}                end-to-end latency (histogram)
  ekrp_retrieval_seconds / ekrp_llm_seconds / ekrp_first_token_seconds    where the time goes
  ekrp_tokens_total{kind}                        prompt / completion tokens (CPU cost)
  ekrp_cache_total{result}                       answer-cache hit / miss
  ekrp_rejections_total{scope,code}              requests turned away: rate_limited, busy, ...
  ekrp_stream_errors_total{code}                 failures after streaming started
  ekrp_logins_total{outcome}                     ok / failed / throttled
  ekrp_http_requests_total / ekrp_http_request_seconds    per route template, not per URL

  Read at scrape time (always current, never stale between requests):
  ekrp_generation_in_flight / _waiting / _capacity   the CPU gate
  ekrp_documents{status}                         corpus by pipeline status (indexed, *_failed, ...)
  ekrp_last_sync_timestamp_seconds               when the last ingestion run finished
  ekrp_dependency_up{name}                       postgres / qdrant / ollama / redis

A dedicated registry keeps tests isolated and avoids duplicate-registration errors on reload.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

log = logging.getLogger("rag.metrics")

ASK_BUCKETS = (0.05, 0.25, 1, 2.5, 5, 10, 15, 20, 30, 45, 60, 120)
FAST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)

KNOWN_REASONS = {"no_results", "low_relevance", "model_declined", "empty_model_output",
                 "model_cited_nonexistent_sources", "no_citations", "stripped_invalid_citations"}
KNOWN_CODES = {"rate_limited", "busy", "question_required", "question_too_long", "not_authenticated",
               "invalid_token", "role_not_allowed", "invalid_credentials", "internal", "forbidden"}
KNOWN_STATUSES = {"answered", "abstained", "ungrounded", "llm_unavailable"}


def _label(value, allowed: set[str], default: str = "other") -> str:
    return value if value in allowed else default


class NullMetrics:
    """Used when metrics are off, and by tests that don't care. Every method does nothing."""

    def login(self, outcome): pass
    def rejected(self, scope, code): pass
    def cache(self, hit): pass
    def answered(self, role, status, reason, cached, timings, prompt_tokens=0, completion_tokens=0): pass
    def stream_error(self, code): pass
    def http(self, method, path, status, seconds): pass

    def snapshot(self, now=None) -> dict:
        return {}

    def render(self) -> tuple[bytes, str]:
        return b"", "text/plain; charset=utf-8"


class Metrics:
    def __init__(
        self,
        *,
        registry=None,
        gate=None,
        documents: Callable[[], dict[str, int]] | None = None,
        last_sync: Callable[[], float | None] | None = None,
        dependencies: Callable[[], dict[str, bool]] | None = None,
        info: dict[str, str] | None = None,
    ):
        from prometheus_client import CollectorRegistry, Counter, Histogram

        self.registry = registry or CollectorRegistry()
        r = self.registry
        self._asks = Counter("ekrp_asks", "Questions answered, by outcome",
                             ["role", "status", "cached", "reason"], registry=r)
        self._ask_seconds = Histogram("ekrp_ask_seconds", "End-to-end time to answer a question",
                                      ["status", "cached"], buckets=ASK_BUCKETS, registry=r)
        self._retrieval = Histogram("ekrp_retrieval_seconds", "Time spent retrieving sources",
                                    buckets=FAST_BUCKETS, registry=r)
        self._llm = Histogram("ekrp_llm_seconds", "Time spent in the language model",
                              buckets=ASK_BUCKETS, registry=r)
        self._first_token = Histogram("ekrp_first_token_seconds", "Time until the first answer token",
                                      buckets=ASK_BUCKETS, registry=r)
        self._tokens = Counter("ekrp_tokens", "Model tokens processed", ["kind"], registry=r)
        self._cache = Counter("ekrp_cache", "Answer cache lookups", ["result"], registry=r)
        self._rejected = Counter("ekrp_rejections", "Requests turned away", ["scope", "code"], registry=r)
        self._stream_errors = Counter("ekrp_stream_errors", "Failures after streaming began",
                                      ["code"], registry=r)
        self._logins = Counter("ekrp_logins", "Login attempts", ["outcome"], registry=r)
        self._http = Counter("ekrp_http_requests", "HTTP requests", ["method", "path", "status"],
                             registry=r)
        self._http_seconds = Histogram("ekrp_http_request_seconds", "HTTP request duration",
                                       ["path"], buckets=ASK_BUCKETS, registry=r)
        r.register(_LiveCollector(gate, documents, last_sync, dependencies, info or {}))

    # ---------------------------------------------------------------- recording
    def login(self, outcome: str) -> None:
        self._logins.labels(_label(outcome, {"ok", "failed", "throttled"})).inc()

    def rejected(self, scope: str, code: str) -> None:
        self._rejected.labels(_label(scope, {"ask", "auth", "login"}), _label(code, KNOWN_CODES)).inc()

    def cache(self, hit: bool) -> None:
        self._cache.labels("hit" if hit else "miss").inc()

    def answered(self, role, status, reason, cached, timings, prompt_tokens=0, completion_tokens=0):
        status = _label(status, KNOWN_STATUSES)
        cached_label = "true" if cached else "false"
        total_ms = (timings or {}).get("total_ms")
        reason = "none" if reason is None else _label(reason, KNOWN_REASONS)
        self._asks.labels(str(role), status, cached_label, reason).inc()
        if total_ms is not None:
            self._ask_seconds.labels(status, cached_label).observe(total_ms / 1000)
        if not cached:                                   # a cache replay did none of this work
            t = timings or {}
            if "retrieve_ms" in t:
                self._retrieval.observe(t["retrieve_ms"] / 1000)
            if "llm_ms" in t:
                self._llm.observe(t["llm_ms"] / 1000)
            if "first_token_ms" in t:
                self._first_token.observe(t["first_token_ms"] / 1000)
            if prompt_tokens:
                self._tokens.labels("prompt").inc(prompt_tokens)
            if completion_tokens:
                self._tokens.labels("completion").inc(completion_tokens)

    def stream_error(self, code: str) -> None:
        self._stream_errors.labels(_label(code, KNOWN_CODES)).inc()

    def http(self, method: str, path: str, status: int, seconds: float) -> None:
        method = method if method in {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"} else "OTHER"
        self._http.labels(method, path, str(status)).inc()
        self._http_seconds.labels(path).observe(seconds)

    def snapshot(self, now: float | None = None) -> dict:
        """The headline numbers for the admin page, read from this process's own registry."""
        return summarize(self.registry.collect(), now)

    # ---------------------------------------------------------------- exposition
    def render(self) -> tuple[bytes, str]:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST


class _LiveCollector:
    """Gauges that are computed when Prometheus scrapes, so they are never stale. Each source is
    optional and isolated: if the database is down, the scrape still succeeds and the missing
    gauges (plus ekrp_dependency_up=0) say so."""

    def __init__(self, gate, documents, last_sync, dependencies, info):
        self.gate, self.documents, self.last_sync = gate, documents, last_sync
        self.dependencies, self.info = dependencies, info

    def describe(self):
        return []                       # dynamic: the metric families depend on live data

    def collect(self):
        from prometheus_client.core import GaugeMetricFamily

        if self.info:
            fam = GaugeMetricFamily("ekrp_info", "Build and model information", labels=list(self.info))
            fam.add_metric([str(v) for v in self.info.values()], 1)
            yield fam

        if self.gate is not None:
            for name, doc, value in (
                ("ekrp_generation_in_flight", "Answers being generated now", self.gate.in_flight),
                ("ekrp_generation_waiting", "Requests waiting for a generation slot", self.gate.waiting),
                ("ekrp_generation_capacity", "Concurrent generations allowed", self.gate.max_concurrent),
            ):
                fam = GaugeMetricFamily(name, doc)
                fam.add_metric([], value)
                yield fam

        counts = self._safely("documents", self.documents)
        if counts is not None:
            fam = GaugeMetricFamily("ekrp_documents", "Documents by pipeline status", labels=["status"])
            for status, n in sorted(counts.items()):
                fam.add_metric([status], n)
            yield fam

        synced = self._safely("last_sync", self.last_sync)
        if synced is not None:
            fam = GaugeMetricFamily("ekrp_last_sync_timestamp_seconds",
                                    "Unix time the last ingestion run finished")
            fam.add_metric([], synced)
            yield fam

        up = self._safely("dependencies", self.dependencies)
        if up is not None:
            fam = GaugeMetricFamily("ekrp_dependency_up", "1 if the dependency answered its check",
                                    labels=["name"])
            for name, ok in sorted(up.items()):
                fam.add_metric([name], 1 if ok else 0)
            yield fam

    @staticmethod
    def _safely(what: str, fn):
        if fn is None:
            return None
        try:
            return fn()
        except Exception as exc:
            log.warning("metrics: could not read %s: %s: %s", what, type(exc).__name__, exc)
            return None


def dependency_probe(checks: dict[str, Callable[[], None]]) -> Callable[[], dict[str, bool]]:
    """Turn {name: check()} (raises when down) into one call returning {name: up?}."""

    def probe() -> dict[str, bool]:
        out = {}
        for name, check in checks.items():
            try:
                check()
                out[name] = True
            except Exception:
                out[name] = False
        return out

    return probe


def to_timestamp(value) -> float | None:
    """datetime (or None) -> unix seconds, for ekrp_last_sync_timestamp_seconds."""
    return value.timestamp() if value is not None else None


# ------------------------------------------------------------------ admin summary

def _samples(families, suffix: str):
    """(labels, value) for every sample whose full name is `suffix`."""
    for fam in families:
        for s in fam.samples:
            if s.name == suffix:
                yield s.labels, s.value


def _by(families, sample_name: str, label: str, where: dict | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for labels, value in _samples(families, sample_name):
        if where and any(labels.get(k) != v for k, v in where.items()):
            continue
        key = labels.get(label, "")
        out[key] = out.get(key, 0) + value
    return out


def quantile_from_buckets(buckets: list[tuple[float, float]], q: float) -> float | None:
    """Prometheus-style estimate: linear interpolation inside the bucket that holds the q-th
    observation. `buckets` are cumulative (upper_bound, count) pairs, +Inf last."""
    buckets = sorted(buckets)
    if not buckets or buckets[-1][1] <= 0:
        return None
    target = q * buckets[-1][1]
    previous_bound, previous_count = 0.0, 0.0
    for bound, count in buckets:
        if count >= target:
            if bound == float("inf"):
                return previous_bound                 # beyond the last finite bucket: report its edge
            span = count - previous_count
            fraction = (target - previous_count) / span if span else 0.0
            return previous_bound + (bound - previous_bound) * fraction
        previous_bound, previous_count = bound, count
    return previous_bound


def summarize(families, now: float | None = None) -> dict:
    """Pure function over prometheus_client metric families (anything with `.samples`), so it is
    testable without the library. Aggregates only; never per-user data."""
    import time

    families = list(families)
    now = time.time() if now is None else now

    asks_by_status = _by(families, "ekrp_asks_total", "status")
    cached_asks = _by(families, "ekrp_asks_total", "cached").get("true", 0)
    total_asks = sum(asks_by_status.values())

    live = {}
    for labels, value in _samples(families, "ekrp_ask_seconds_bucket"):
        if labels.get("cached") == "false":
            bound = float("inf") if labels["le"] == "+Inf" else float(labels["le"])
            live[bound] = live.get(bound, 0) + value
    live_count = live.get(float("inf"), 0)
    sums = _by(families, "ekrp_ask_seconds_sum", "cached")

    cache = _by(families, "ekrp_cache_total", "result")
    lookups = cache.get("hit", 0) + cache.get("miss", 0)

    http = _by(families, "ekrp_http_requests_total", "status")
    http_total = sum(http.values())
    http_5xx = sum(v for k, v in http.items() if k.startswith("5"))

    gauge = lambda name: next((v for _, v in _samples(families, name)), None)   # noqa: E731
    last_sync = gauge("ekrp_last_sync_timestamp_seconds")

    return {
        "asks": {"total": int(total_asks), "cached": int(cached_asks),
                 "by_status": {k: int(v) for k, v in sorted(asks_by_status.items())}},
        "latency_seconds": {
            "count": int(live_count),
            "mean": (sums.get("false", 0) / live_count) if live_count else None,
            "p50": quantile_from_buckets(list(live.items()), 0.50),
            "p95": quantile_from_buckets(list(live.items()), 0.95),
        },
        "cache": {"hit": int(cache.get("hit", 0)), "miss": int(cache.get("miss", 0)),
                  "hit_ratio": (cache.get("hit", 0) / lookups) if lookups else None},
        "rejections": {k: int(v) for k, v in sorted(_by(families, "ekrp_rejections_total", "code").items())},
        "logins": {k: int(v) for k, v in sorted(_by(families, "ekrp_logins_total", "outcome").items())},
        "tokens": {k: int(v) for k, v in sorted(_by(families, "ekrp_tokens_total", "kind").items())},
        "http": {"requests": int(http_total), "server_errors": int(http_5xx)},
        "generation": {"in_flight": gauge("ekrp_generation_in_flight"),
                       "waiting": gauge("ekrp_generation_waiting"),
                       "capacity": gauge("ekrp_generation_capacity")},
        "dependencies": {k: bool(v) for k, v in sorted(_by(families, "ekrp_dependency_up", "name").items())},
        "documents": {k: int(v) for k, v in sorted(_by(families, "ekrp_documents", "status").items())},
        "last_sync_age_seconds": (now - last_sync) if last_sync else None,
    }
