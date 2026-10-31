"""HTTP-level tests. They need FastAPI + httpx (pip install -r requirements.txt) and are skipped
otherwise; the rules themselves are covered without a web framework in test_api_core.py."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from serving.app import create_app  # noqa: E402
from serving.ratelimit import Limit  # noqa: E402
from tests.test_api_core import FakeAnswers, make  # noqa: E402


def client_for(**kw):
    svc = make(FakeAnswers(), **kw)
    return TestClient(create_app(svc)), svc


def login(client, user="alice", pw="alice-pw"):
    r = client.post("/auth/login", json={"username": user, "password": pw})
    return r, {"Authorization": f"Bearer {r.json().get('access_token', '')}"}


def parse_sse(text):
    events = []
    for frame in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.split("\n") if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health_and_readiness():
    client, svc = client_for()
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200
    svc.readiness_checks = {"ollama": lambda: (_ for _ in ()).throw(ConnectionError())}
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["checks"] == {"ollama": "ConnectionError"}


def test_login_then_me_and_ask():
    client, _ = client_for()
    r, auth = login(client)
    assert r.status_code == 200 and r.json()["role"] == "agent"
    assert client.get("/me", headers=auth).json() == {"username": "alice", "role": "agent"}
    out = client.post("/ask", headers=auth, json={"question": "How do I reset a password?"})
    assert out.status_code == 200 and out.json()["status"] == "answered" and out.json()["citations"]


def test_errors_share_one_json_shape_and_login_does_not_reveal_which_part_was_wrong():
    client, _ = client_for()
    a = client.post("/auth/login", json={"username": "alice", "password": "bad"})
    b = client.post("/auth/login", json={"username": "ghost", "password": "bad"})
    assert a.status_code == b.status_code == 401 and a.json() == b.json()
    assert set(a.json()["error"]) == {"code", "message"}


def test_protected_routes_reject_missing_and_bad_tokens():
    client, _ = client_for()
    for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic abc"}):
        for path, method in (("/me", "get"), ("/ask", "post"), ("/ask/stream", "post")):
            r = getattr(client, method)(path, headers=headers, **({"json": {"question": "q"}} if method == "post" else {}))
            assert r.status_code == 401, (path, headers)


def test_empty_question_is_422_and_oversize_body_is_rejected():
    client, _ = client_for()
    _, auth = login(client)
    assert client.post("/ask", headers=auth, json={"question": "  "}).status_code == 422
    assert client.post("/ask", headers=auth, json={"question": "x" * 5000}).status_code == 422


def test_rate_limit_returns_429_with_retry_after_header():
    client, _ = client_for(ask_limit=Limit(1, 60), cache=False)
    _, auth = login(client)
    assert client.post("/ask", headers=auth, json={"question": "one"}).status_code == 200
    r = client.post("/ask", headers=auth, json={"question": "two"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_streaming_endpoint_sends_sources_tokens_then_done():
    client, _ = client_for(cache=False)
    _, auth = login(client)
    with client.stream("POST", "/ask/stream", headers=auth, json={"question": "q"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-store"
        events = parse_sse("".join(r.iter_text()))
    assert [e for e, _ in events] == ["sources", "token", "token", "done"]
    assert events[-1][1]["answer"]["status"] == "answered"


def test_stream_failures_before_the_stream_starts_are_real_http_errors_not_event_streams():
    client, _ = client_for(ask_limit=Limit(1, 60), cache=False)
    _, auth = login(client)
    client.post("/ask", headers=auth, json={"question": "one"})
    r = client.post("/ask/stream", headers=auth, json={"question": "two"})
    assert r.status_code == 429 and r.headers["content-type"].startswith("application/json")


def test_page_and_static_files_are_served_with_security_headers():
    client, _ = client_for()
    page = client.get("/")
    assert page.status_code == 200 and "Support Knowledge Assistant" in page.text
    assert page.headers["x-frame-options"] == "DENY" and "default-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/lib.js").status_code == 200
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_a_cached_answer_for_one_role_is_not_visible_to_another_over_http():
    client, _ = client_for()
    _, admin = login(client, "carol", "carol-pw")
    _, agent = login(client, "alice", "alice-pw")
    q = {"question": "what is the leave policy"}
    assert "RESTRICTED" in client.post("/ask", headers=admin, json=q).json()["text"]
    assert "RESTRICTED" not in client.post("/ask", headers=agent, json=q).text


# ------------------------------------------------------------------ Step 14 hardening

def test_api_docs_are_on_in_development_and_off_in_production():
    dev = TestClient(create_app(make(FakeAnswers()), docs=True))
    prod = TestClient(create_app(make(FakeAnswers()), docs=False))
    assert dev.get("/openapi.json").status_code == 200 and dev.get("/docs").status_code == 200
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert prod.get(path).status_code == 404


def test_oversized_request_bodies_are_refused_with_413():
    client, _ = client_for()
    r = client.post("/auth/login", content=b"x" * 40_000, headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
    assert r.headers["x-content-type-options"] == "nosniff"          # still wrapped by the security headers
