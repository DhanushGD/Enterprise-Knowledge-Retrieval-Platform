"""Builds the real ApiService from config, .env and the running infrastructure."""
from __future__ import annotations

import hashlib
import logging

from serving.api_core import ApiService
from serving.cache import AnswerCache, MemoryTTLCache, RedisTTLCache, TTLValue
from serving.concurrency import GenerationGate
from serving.metrics import Metrics, dependency_probe, to_timestamp
from serving.ratelimit import Limit, MemoryRateLimiter, RedisRateLimiter, SafeLimiter
from serving.security import check_secret

log = logging.getLogger("rag.api")


def connect_redis(url: str):
    """A working Redis client, or None (the API then falls back to in-process limiter/cache)."""
    try:
        import redis

        client = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        return client
    except Exception as exc:
        log.warning("Redis unavailable (%s: %s): using in-process cache and rate limiting. "
                    "Fine for one process; not shared across instances.", type(exc).__name__, exc)
        return None


def build_api_service() -> ApiService:
    import requests

    from core.db import SqlRegistry, init_db
    from core.settings import company, settings
    from serving.generation import SYSTEM_PROMPT, build_answer_service
    from serving.users import SqlUserStore

    for warning in check_secret(settings.jwt_secret, settings.app_env):    # raises in production
        log.warning("SECURITY: %s", warning)

    init_db()                                          # creates the users table on first run
    cfg = company.get("serving", {}) or {}
    ask_cfg, login_cfg = cfg.get("ask_rate_limit", {}), cfg.get("login_rate_limit", {})

    redis_client = connect_redis(settings.redis_url)
    if redis_client:
        limiter, backend = SafeLimiter(RedisRateLimiter(redis_client)), RedisTTLCache(redis_client)
    else:
        limiter, backend = MemoryRateLimiter(), MemoryTTLCache()
    cache = AnswerCache(backend, ttl=cfg.get("cache_ttl_seconds", 3600)) if cfg.get("cache_enabled", True) else None

    answers = build_answer_service()
    registry = SqlRegistry()
    fingerprint = hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:8]
    model_id = "|".join(map(str, (
        settings.llm_model, fingerprint, answers.context_chunks, answers.max_chunk_chars,
        answers.max_answer_tokens, answers.min_dense_score)))

    def check_ollama():
        requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2).raise_for_status()

    checks = {
        "postgres": registry.status_counts,
        "qdrant": lambda: answers.retriever.index.client.get_collection(answers.retriever.index.collection),
        "ollama": check_ollama,
    }
    if redis_client:
        checks["redis"] = redis_client.ping

    gate = GenerationGate(cfg.get("max_concurrent_generations", 1), cfg.get("max_waiting", 2),
                          cfg.get("queue_timeout_seconds", 30))
    # Scrape-time gauges hit Postgres / Ollama, so cache them for 30 s: Prometheus asks every 15 s.
    metrics = Metrics(
        gate=gate,
        documents=TTLValue(registry.status_counts, ttl=30),
        last_sync=TTLValue(lambda: to_timestamp(registry.last_sync_at()), ttl=30),
        dependencies=TTLValue(dependency_probe(checks), ttl=30),
        info={"llm_model": settings.llm_model, "embedding_model": settings.embedding_model},
    )

    return ApiService(
        answers, SqlUserStore(), company["roles"],
        secret=settings.jwt_secret, token_minutes=settings.jwt_expire_minutes,
        cache=cache, limiter=limiter,
        gate=gate, metrics=metrics,
        corpus_version=TTLValue(registry.corpus_version, ttl=30), model_id=model_id,
        ask_limit=Limit(ask_cfg.get("limit", 12), ask_cfg.get("window_seconds", 60)),
        login_limit=Limit(login_cfg.get("limit", 5), login_cfg.get("window_seconds", 60)),
        ip_login_limit=Limit(cfg.get("ip_login_limit", 20), 60),
        max_question_chars=cfg.get("max_question_chars", 500),
        readiness_checks=checks,
        warm_up=getattr(answers.llm, "warm_up", None),
        admin_roles=tuple(cfg.get("admin_roles", ["admin"])),
        grafana_url=settings.grafana_url,
    )
