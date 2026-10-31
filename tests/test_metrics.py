"""Step 13: what the API records, and that the dashboards/alerts only use metrics that exist.

Most tests use a recording fake, so they run anywhere. The ones marked below need
prometheus_client (and FastAPI for the HTTP ones) and are skipped when those are missing.
"""
import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

from serving.api_core import ApiError
from serving.concurrency import GenerationGate
from serving.metrics import NullMetrics, dependency_probe, to_timestamp
from tests.test_api_core import FakeAnswers, make, who

ROOT = Path(__file__).resolve().parent.parent


class Recorder(NullMetrics):
    def __init__(self):
        self.calls = []

    def _rec(self, name, *args):
        self.calls.append((name, *args))

    def login(self, outcome): self._rec("login", outcome)
    def rejected(self, scope, code): self._rec("rejected", scope, code)
    def cache(self, hit): self._rec("cache", hit)
    def stream_error(self, code): self._rec("stream_error", code)

    def answered(self, role, status, reason, cached, timings, prompt_tokens=0, completion_tokens=0):
        self._rec("answered", role, status, reason, cached, dict(timings), prompt_tokens, completion_tokens)

    def named(self, name):
        return [c[1:] for c in self.calls if c[0] == name]


def svc_with(**kw):
    rec = Recorder()
    return make(metrics=rec, **kw), rec


def drain(stream):
    return list(stream)


# ------------------------------------------------------------------ login and auth

def test_login_outcomes_are_counted():
    svc, rec = svc_with()
    svc.login("alice", "alice-pw", "ip")
    with pytest.raises(ApiError):
        svc.login("alice", "wrong", "ip")
    assert rec.named("login") == [("ok",), ("failed",)]


def test_throttled_logins_are_told_apart_from_wrong_passwords():
    from serving.ratelimit import Limit

    svc, rec = svc_with(login_limit=Limit(1, 60))
    with pytest.raises(ApiError):
        svc.login("alice", "wrong", "ip")
    with pytest.raises(ApiError):
        svc.login("alice", "wrong", "ip")
    assert rec.named("login") == [("failed",), ("throttled",)]


def test_rejected_tokens_are_counted_by_code():
    svc, rec = svc_with()
    for token in (None, "garbage"):
        with pytest.raises(ApiError):
            svc.authenticate(token)
    assert rec.named("rejected") == [("auth", "not_authenticated"), ("auth", "invalid_token")]


# ------------------------------------------------------------------ asking

def test_a_live_answer_is_recorded_with_its_timings_and_outcome():
    svc, rec = svc_with()
    drain(svc.ask_stream(who(), "How do I reset a password?"))
    assert rec.named("cache") == [(False,)]
    [(role, status, reason, cached, timings, _p, _c)] = rec.named("answered")
    assert (role, status, reason, cached) == ("agent", "answered", None, False)
    assert timings == {"total_ms": 12.0}


def test_a_cache_hit_is_recorded_as_cached_and_does_not_count_model_work():
    svc, rec = svc_with()
    drain(svc.ask_stream(who(), "same question"))
    drain(svc.ask_stream(who(), "same question"))
    assert rec.named("cache") == [(False,), (True,)]
    answered = rec.named("answered")
    assert [a[3] for a in answered] == [False, True]          # second one: cached=True
    assert answered[1][5] == 0 and answered[1][6] == 0        # no model tokens on a replay


def test_refusals_carry_their_reason():
    svc, rec = svc_with(answers=FakeAnswers(status="abstained", reason="low_relevance", text="No."))
    drain(svc.ask_stream(who(), "weather?"))
    assert rec.named("answered")[0][1:3] == ("abstained", "low_relevance")


def test_rejections_before_streaming_are_counted_by_code():
    from serving.ratelimit import Limit

    svc, rec = svc_with(ask_limit=Limit(1, 60), max_question_chars=10)
    with pytest.raises(ApiError):
        svc.ask_stream(who(), "")
    with pytest.raises(ApiError):
        svc.ask_stream(who(), "x" * 11)
    drain(svc.ask_stream(who(), "fine"))
    with pytest.raises(ApiError):
        svc.ask_stream(who(), "again")
    assert rec.named("rejected") == [("ask", "question_required"), ("ask", "question_too_long"),
                                     ("ask", "rate_limited")]


def test_load_shedding_is_counted():
    gate = GenerationGate(1, 0, 0.01)
    svc, rec = svc_with(gate=gate)
    with gate.acquire():                                       # the only slot is taken
        with pytest.raises(ApiError) as e:
            svc.ask_stream(who(), "q?")
    assert e.value.code == "busy" and rec.named("rejected") == [("ask", "busy")]


def test_a_busy_timeout_after_streaming_started_is_a_stream_error():
    gate = GenerationGate(1, 5, 0.05)
    svc, rec = svc_with(gate=gate)
    with gate.acquire():
        events = drain(svc.ask_stream(who(), "q?"))
    assert events[-1]["type"] == "error" and rec.named("stream_error") == [("busy",)]


def test_unexpected_failures_are_counted_as_internal():
    class Broken:
        def stream(self, question, role):
            raise RuntimeError("boom")
            yield

    svc, rec = svc_with(answers=Broken())
    assert drain(svc.ask_stream(who(), "q?"))[-1]["code"] == "internal"
    assert rec.named("stream_error") == [("internal",)]


def test_no_metric_call_carries_a_username_or_question_text():
    svc, rec = svc_with()
    drain(svc.ask_stream(who("alice"), "my secret customer question"))
    with pytest.raises(ApiError):
        svc.login("alice", "wrong", "10.0.0.9")
    flat = json.dumps(rec.calls, default=str)
    assert "alice" not in flat and "secret customer" not in flat and "10.0.0.9" not in flat


def test_metrics_are_optional():
    svc = make()                                               # no metrics argument
    drain(svc.ask_stream(who(), "q?"))
    assert svc.metrics.render() == (b"", "text/plain; charset=utf-8")


# ------------------------------------------------------------------ helpers

def test_dependency_probe_reports_each_check():
    def down():
        raise ConnectionError("x")

    assert dependency_probe({"postgres": lambda: None, "ollama": down})() == {"postgres": True, "ollama": False}


def test_to_timestamp():
    from datetime import datetime, timezone

    assert to_timestamp(None) is None
    assert to_timestamp(datetime(2026, 1, 1, tzinfo=timezone.utc)) == 1767225600.0


# ------------------------------------------------------------------ dashboards and alerts

def metric_names_defined():
    literals = set(re.findall(r'"(ekrp_[a-z_]+)"', (ROOT / "serving" / "metrics.py").read_text()))
    names = set()
    for base in literals:
        names |= {base, f"{base}_total", f"{base}_bucket", f"{base}_sum", f"{base}_count"}
    return names


def metrics_in(expr: str) -> set[str]:
    return set(re.findall(r"\bekrp_[a-z_]+", expr))


def test_alert_rules_are_well_formed_and_use_real_metrics():
    rules = yaml.safe_load((ROOT / "infra" / "prometheus" / "alerts.yml").read_text())["groups"]
    defined, seen = metric_names_defined(), set()
    for group in rules:
        for rule in group["rules"]:
            assert rule["alert"] not in seen
            seen.add(rule["alert"])
            assert rule["labels"]["severity"] in ("warning", "critical")
            assert rule["annotations"]["summary"] and rule["annotations"]["description"]
            unknown = metrics_in(rule["expr"]) - defined
            assert not unknown, f"{rule['alert']} uses undefined metrics {unknown}"
    assert len(seen) >= 10


def test_prometheus_loads_the_alert_file_that_exists():
    cfg = yaml.safe_load((ROOT / "infra" / "prometheus" / "prometheus.yml").read_text())
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    mounts = " ".join(compose["services"]["prometheus"]["volumes"])
    for rule_file in cfg["rule_files"]:
        assert Path(rule_file).name in mounts and (ROOT / "infra" / "prometheus" / Path(rule_file).name).exists()
    assert any(j["job_name"] == "rag-api" and j["metrics_path"] == "/metrics" for j in cfg["scrape_configs"])


def load_dashboard_builder():
    spec = importlib.util.spec_from_file_location("build_dashboard", ROOT / "infra" / "grafana" / "build_dashboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboard_file_matches_its_generator_and_uses_real_metrics():
    builder = load_dashboard_builder()
    on_disk = json.loads((ROOT / "infra" / "grafana" / "dashboards" / "ekrp-overview.json").read_text())
    assert on_disk == builder.build(), "run: python infra/grafana/build_dashboard.py"
    defined, ids = metric_names_defined(), set()
    for panel in on_disk["panels"]:
        assert panel["id"] not in ids
        ids.add(panel["id"])
        g = panel["gridPos"]
        assert 0 <= g["x"] and g["x"] + g["w"] <= 24 and g["w"] > 0 and g["h"] > 0
        assert panel["datasource"]["uid"] == "prometheus"
        for target in panel["targets"]:
            unknown = metrics_in(target["expr"]) - defined
            assert not unknown, f"{panel['title']}: undefined metrics {unknown}"


def test_grafana_provisioning_points_at_the_right_places():
    ds = yaml.safe_load((ROOT / "infra/grafana/provisioning/datasources/prometheus.yml").read_text())
    assert ds["datasources"][0]["uid"] == "prometheus" and ds["datasources"][0]["url"] == "http://prometheus:9090"
    provider = yaml.safe_load((ROOT / "infra/grafana/provisioning/dashboards/dashboards.yml").read_text())
    path = provider["providers"][0]["options"]["path"]
    mounts = " ".join(yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["grafana"]["volumes"])
    assert f"{path}:ro" in mounts and "/etc/grafana/provisioning" in mounts


# ------------------------------------------------------------------ real Prometheus exposition

prom = pytest.mark.skipif(importlib.util.find_spec("prometheus_client") is None,
                          reason="prometheus_client not installed")


def sample(text: str, series: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(series + " "):
            return float(line.rsplit(" ", 1)[1])
    return None


@prom
def test_exposition_contains_what_was_recorded():
    from serving.metrics import Metrics

    m = Metrics()
    m.login("ok")
    m.cache(False)
    m.rejected("ask", "rate_limited")
    m.stream_error("internal")
    m.answered("agent", "answered", None, False,
               {"total_ms": 12000, "retrieve_ms": 40, "llm_ms": 11000, "first_token_ms": 900}, 300, 80)
    m.http("POST", "/ask", 200, 1.5)
    text = m.render()[0].decode()

    assert sample(text, 'ekrp_asks_total{cached="false",reason="none",role="agent",status="answered"}') == 1.0
    assert sample(text, 'ekrp_ask_seconds_count{cached="false",status="answered"}') == 1.0
    assert sample(text, 'ekrp_ask_seconds_sum{cached="false",status="answered"}') == pytest.approx(12.0)
    assert sample(text, 'ekrp_tokens_total{kind="prompt"}') == 300.0
    assert sample(text, 'ekrp_tokens_total{kind="completion"}') == 80.0
    assert sample(text, 'ekrp_logins_total{outcome="ok"}') == 1.0
    assert sample(text, 'ekrp_cache_total{result="miss"}') == 1.0
    assert sample(text, 'ekrp_rejections_total{code="rate_limited",scope="ask"}') == 1.0
    assert sample(text, 'ekrp_stream_errors_total{code="internal"}') == 1.0
    assert sample(text, 'ekrp_http_requests_total{method="POST",path="/ask",status="200"}') == 1.0
    assert sample(text, "ekrp_retrieval_seconds_count") == 1.0


@prom
def test_a_cache_replay_adds_no_model_timings():
    from serving.metrics import Metrics

    m = Metrics()
    m.answered("agent", "answered", None, True, {"total_ms": 5})
    text = m.render()[0].decode()
    assert sample(text, 'ekrp_asks_total{cached="true",reason="none",role="agent",status="answered"}') == 1.0
    assert sample(text, "ekrp_llm_seconds_count") == 0.0 and sample(text, "ekrp_retrieval_seconds_count") == 0.0


@prom
def test_unexpected_label_values_collapse_so_series_cannot_explode():
    from serving.metrics import Metrics

    m = Metrics()
    for i in range(50):
        m.rejected("ask", f"made-up-{i}")
        m.answered("agent", f"weird-{i}", f"reason-{i}", False, {})
    text = m.render()[0].decode()
    assert sample(text, 'ekrp_rejections_total{code="other",scope="ask"}') == 50.0
    assert text.count("ekrp_asks_total{") == 1
    assert sample(text, 'ekrp_asks_total{cached="false",reason="other",role="agent",status="other"}') == 50.0


@prom
def test_scrape_time_gauges_and_failure_isolation():
    from serving.metrics import Metrics

    gate = GenerationGate(2, 3, 1)

    def broken():
        raise ConnectionError("db down")

    m = Metrics(gate=gate, documents=lambda: {"indexed": 393, "skipped": 7, "index_failed": 1},
                last_sync=lambda: 1_700_000_000.0, dependencies=lambda: {"postgres": True, "ollama": False},
                info={"llm_model": "llama3.2:3b"})
    with gate.acquire():
        text = m.render()[0].decode()
    assert sample(text, "ekrp_generation_in_flight") == 1.0 and sample(text, "ekrp_generation_capacity") == 2.0
    assert sample(text, 'ekrp_documents{status="indexed"}') == 393.0
    assert sample(text, "ekrp_last_sync_timestamp_seconds") == 1_700_000_000.0
    assert sample(text, 'ekrp_dependency_up{name="ollama"}') == 0.0
    assert sample(text, 'ekrp_info{llm_model="llama3.2:3b"}') == 1.0

    degraded = Metrics(documents=broken, last_sync=broken, dependencies=broken).render()[0].decode()
    assert "ekrp_documents" not in degraded and "ekrp_last_sync" not in degraded     # omitted, no crash


@prom
def test_each_instance_has_its_own_registry():
    from serving.metrics import Metrics

    a, b = Metrics(), Metrics()
    a.login("ok")
    assert sample(a.render()[0].decode(), 'ekrp_logins_total{outcome="ok"}') == 1.0
    assert sample(b.render()[0].decode(), 'ekrp_logins_total{outcome="ok"}') in (None, 0.0)


# ------------------------------------------------------------------ over HTTP

@prom
def test_metrics_endpoint_and_route_template_labels():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from serving.app import create_app
    from serving.metrics import Metrics

    svc = make(FakeAnswers(), metrics=Metrics())
    client = TestClient(create_app(svc))
    token = client.post("/auth/login", json={"username": "alice", "password": "alice-pw"}).json()["access_token"]
    client.post("/ask", headers={"Authorization": f"Bearer {token}"}, json={"question": "How do I reset it?"})
    client.get("/definitely/not/a/route/12345")
    client.get("/health")

    r = client.get("/metrics")
    text = r.text
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert sample(text, 'ekrp_http_requests_total{method="POST",path="/ask",status="200"}') == 1.0
    assert sample(text, 'ekrp_http_requests_total{method="POST",path="/auth/login",status="200"}') == 1.0
    assert sample(text, 'ekrp_http_requests_total{method="GET",path="unmatched",status="404"}') == 1.0
    assert "/definitely" not in text and "12345" not in text          # raw URLs never become labels
    assert 'path="/metrics"' not in text                              # scraping isn't counted
    assert sample(text, 'ekrp_logins_total{outcome="ok"}') == 1.0
    assert "alice" not in text


@prom
def test_metrics_token_is_enforced_when_configured():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from serving.app import create_app
    from serving.metrics import Metrics

    client = TestClient(create_app(make(FakeAnswers(), metrics=Metrics()), metrics_token="s3cret"))
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200
