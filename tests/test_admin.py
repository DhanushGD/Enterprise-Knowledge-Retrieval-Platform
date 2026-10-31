"""Step 15: the admin page - metrics summary, role check, Grafana embedding, page hygiene."""
import re
from collections import namedtuple
from pathlib import Path

import pytest

from serving.api_core import ApiError, grafana_links
from serving.metrics import quantile_from_buckets, summarize
from tests.test_api_core import make, who
from tests.test_http_and_ui import call

Sample = namedtuple("Sample", "name labels value")
Family = namedtuple("Family", "name samples")
STATIC = Path(__file__).resolve().parent.parent / "serving" / "static"
INF = "+Inf"


def families():
    """What prometheus_client would hand over after: 10 asks (3 cached), 2 refusals, 5 cache lookups..."""
    def bucket(cached, counts):          # counts: cumulative per le
        return [Sample("ekrp_ask_seconds_bucket", {"status": "answered", "cached": cached, "le": le}, n)
                for le, n in counts]
    return [
        Family("ekrp_asks", [
            Sample("ekrp_asks_total", {"role": "agent", "status": "answered", "cached": "false", "reason": "none"}, 5),
            Sample("ekrp_asks_total", {"role": "agent", "status": "answered", "cached": "true", "reason": "none"}, 3),
            Sample("ekrp_asks_total", {"role": "agent", "status": "abstained", "cached": "false", "reason": "low_relevance"}, 2)]),
        Family("ekrp_ask_seconds", bucket("false", [("1.0", 2), ("10.0", 6), ("30.0", 7), (INF, 7)])
                + [Sample("ekrp_ask_seconds_sum", {"status": "answered", "cached": "false"}, 70.0)]),
        Family("ekrp_cache", [Sample("ekrp_cache_total", {"result": "hit"}, 3), Sample("ekrp_cache_total", {"result": "miss"}, 1)]),
        Family("ekrp_rejections", [Sample("ekrp_rejections_total", {"scope": "ask", "code": "rate_limited"}, 4)]),
        Family("ekrp_logins", [Sample("ekrp_logins_total", {"outcome": "ok"}, 9), Sample("ekrp_logins_total", {"outcome": "failed"}, 2)]),
        Family("ekrp_tokens", [Sample("ekrp_tokens_total", {"kind": "prompt"}, 1000)]),
        Family("ekrp_http_requests", [Sample("ekrp_http_requests_total", {"method": "GET", "path": "/me", "status": "200"}, 50),
                                      Sample("ekrp_http_requests_total", {"method": "POST", "path": "/ask", "status": "500"}, 1)]),
        Family("ekrp_generation_in_flight", [Sample("ekrp_generation_in_flight", {}, 1)]),
        Family("ekrp_generation_capacity", [Sample("ekrp_generation_capacity", {}, 1)]),
        Family("ekrp_dependency_up", [Sample("ekrp_dependency_up", {"name": "redis"}, 1), Sample("ekrp_dependency_up", {"name": "qdrant"}, 0)]),
        Family("ekrp_documents", [Sample("ekrp_documents", {"status": "indexed"}, 393)]),
        Family("ekrp_last_sync_timestamp_seconds", [Sample("ekrp_last_sync_timestamp_seconds", {}, 900.0)]),
    ]


# ------------------------------------------------------------------ the numbers
def test_summary_aggregates_counters_and_gauges():
    s = summarize(families(), now=1000.0)
    assert s["asks"] == {"total": 10, "cached": 3, "by_status": {"abstained": 2, "answered": 8}}
    assert s["cache"] == {"hit": 3, "miss": 1, "hit_ratio": 0.75}
    assert s["rejections"] == {"rate_limited": 4} and s["logins"] == {"failed": 2, "ok": 9}
    assert s["http"] == {"requests": 51, "server_errors": 1}
    assert s["dependencies"] == {"qdrant": False, "redis": True}
    assert s["generation"]["in_flight"] == 1 and s["generation"]["waiting"] is None
    assert s["documents"] == {"indexed": 393} and s["last_sync_age_seconds"] == 100.0
    assert s["latency_seconds"]["count"] == 7 and s["latency_seconds"]["mean"] == 10.0


def test_summary_of_an_empty_registry_has_no_division_by_zero():
    s = summarize([], now=1.0)
    assert s["asks"]["total"] == 0 and s["cache"]["hit_ratio"] is None
    assert s["latency_seconds"]["p50"] is None and s["last_sync_age_seconds"] is None


def test_latency_percentiles_use_only_model_work_not_cache_replays():
    s = summarize(families(), now=0)
    # 7 uncached answers: 2 under 1 s, 6 under 10 s, 7 under 30 s -> p50 falls in (1, 10]
    assert 1.0 < s["latency_seconds"]["p50"] <= 10.0
    assert 10.0 < s["latency_seconds"]["p95"] <= 30.0


def test_quantile_interpolation_and_edges():
    buckets = [(1.0, 0), (2.0, 10), (float("inf"), 10)]
    assert quantile_from_buckets(buckets, 0.5) == pytest.approx(1.5)
    assert quantile_from_buckets([(1.0, 0), (float("inf"), 4)], 0.9) == 1.0     # beyond last bucket: its edge
    assert quantile_from_buckets([], 0.5) is None


# ------------------------------------------------------------------ who may see it
def test_only_admin_roles_get_the_summary_and_others_are_counted():
    calls = []

    class Spy:
        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name, a)) or {}

    svc = make(metrics=Spy())
    out = svc.admin_summary(who("carol", "admin"))
    assert set(out) == {"metrics", "controls", "readiness", "grafana"}
    with pytest.raises(ApiError) as e:
        svc.admin_summary(who("alice", "agent"))
    assert e.value.status == 403 and e.value.code == "forbidden"
    assert ("rejected", ("auth", "forbidden")) in calls


def test_admin_roles_are_configurable():
    svc = make(admin_roles=("agent",))
    assert svc.admin_summary(who("alice", "agent"))["controls"]
    with pytest.raises(ApiError):
        svc.admin_summary(who("carol", "admin"))


def test_summary_reports_protections_without_secrets():
    out = make().admin_summary(who("carol", "admin"))
    text = str(out).lower()
    assert "secret" not in text and "password" not in text
    assert out["controls"]["ask_limit"].startswith("12 per 60s")
    assert out["controls"]["cache_backend"] == "MemoryTTLCache"


# ------------------------------------------------------------------ grafana address
@pytest.mark.parametrize("url", ["", "   ", "javascript:alert(1)", "ftp://x", "http://", "not a url",
                                 "http://user:pw@host:3000", "//host/path"])
def test_unsafe_or_empty_grafana_addresses_are_ignored(url):
    assert grafana_links(url) is None


def test_grafana_origin_is_extracted_and_the_dashboard_link_is_built():
    g = grafana_links("https://grafana.example.com:3000/some/path?x=1")
    assert g["origin"] == "https://grafana.example.com:3000"
    assert g["dashboard_url"].startswith("https://grafana.example.com:3000/d/ekrp-overview/")
    assert "kiosk" in g["dashboard_url"]


def test_the_dashboard_uid_matches_the_generated_dashboard():
    import json
    dash = json.loads((Path(__file__).resolve().parent.parent / "infra/grafana/dashboards/ekrp-overview.json").read_text())
    assert grafana_links("http://g:3000")["dashboard_url"].split("/d/")[1].split("/")[0] == dash["uid"]


# ------------------------------------------------------------------ CSP: only /admin may frame Grafana
def test_only_the_admin_page_may_frame_the_configured_origin():
    pytest.importorskip("starlette")
    from serving.http_security import SecurityHeaders

    async def ok(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"x"})

    app = SecurityHeaders(ok, frame_origin="http://localhost:3000")
    assert "frame-src http://localhost:3000" in call(app, "/admin")[1]["content-security-policy"]
    assert "frame-src" not in call(app, "/")[1]["content-security-policy"]
    assert "frame-src" not in call(SecurityHeaders(ok), "/admin")[1]["content-security-policy"]
    assert call(app, "/admin")[1]["x-frame-options"] == "DENY"            # our own pages still cannot be framed


# ------------------------------------------------------------------ page hygiene (strict CSP)
def test_admin_page_has_no_inline_scripts_or_html_injection():
    html = (STATIC / "admin.html").read_text(encoding="utf-8")
    assert not re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", html) and not re.findall(r"\son\w+\s*=", html)
    js = (STATIC / "admin.js").read_text(encoding="utf-8")
    for banned in ("innerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert banned not in js
    assert "/admin/summary" in js


def test_chat_page_links_to_admin_only_hidden_by_default():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<a id="admin-link" href="/admin" hidden>', html)


# ------------------------------------------------------------------ over HTTP (needs FastAPI)
def test_admin_routes_over_http():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from serving.app import create_app

    svc = make(grafana_url="http://localhost:3000")
    client = TestClient(create_app(svc, docs=False))
    page = client.get("/admin")
    assert page.status_code == 200 and "frame-src http://localhost:3000" in page.headers["content-security-policy"]
    assert client.get("/admin/summary").status_code == 401
    token = lambda n: svc.login(n, f"{n}-pw", "1.1.1.1")["access_token"]   # noqa: E731
    assert client.get("/admin/summary", headers={"Authorization": f"Bearer {token('alice')}"}).status_code == 403
    ok = client.get("/admin/summary", headers={"Authorization": f"Bearer {token('carol')}"})
    assert ok.status_code == 200 and ok.json()["grafana"]["origin"] == "http://localhost:3000"
