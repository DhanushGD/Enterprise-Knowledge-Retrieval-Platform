import json
import logging
import time

import pytest

import serving.api_core as api_core
from serving.api_core import ApiError, ApiService, Principal, public_answer, sse
from serving.cache import AnswerCache, MemoryTTLCache
from serving.concurrency import GenerationGate
from serving.generation import Answer, Source
from serving.ratelimit import Limit, MemoryRateLimiter
from serving.security import decode_token, hash_password, issue_token
from serving.users import MemoryUserStore

SECRET = "s" * 48
ROLES = {
    "agent": {"allowed_access_levels": ["public"], "allowed_departments": ["support"]},
    "agent2": {"allowed_access_levels": ["public"], "allowed_departments": ["support"]},
    "admin": {"allowed_access_levels": ["public", "restricted"], "allowed_departments": ["support", "hr"]},
}


def src(n=1, path="docs/a.md"):
    return Source(n, f"Doc {n}", "Sec", path, f"c{n}", "snippet")


class FakeAnswers:
    """Stands in for AnswerService. Admin's answers cite a restricted page."""

    def __init__(self, status="answered", reason=None, text=None, hold=None):
        self.calls, self.status, self.reason, self.text, self.hold = [], status, reason, text, hold

    def stream(self, question, role):
        self.calls.append((question, role))
        secret = role == "admin"
        text = self.text or ("RESTRICTED-HR-POLICY [1]" if secret else "public how-to [1]")
        sources = [src(1, "hr/restricted/leave.md" if secret else "docs/a.md")]
        yield {"type": "sources", "sources": sources}
        yield {"type": "token", "text": text[:5]}
        yield {"type": "token", "text": text[5:]}
        if self.hold:
            self.hold.wait(3)
        yield {"type": "done", "answer": Answer(
            question=question, role=role, status=self.status, text=text,
            citations=sources if self.status == "answered" else [], sources=sources,
            reason=self.reason, raw_text="RAW-UNVALIDATED", timings={"total_ms": 12.0},
            dense_top=0.9)}


def make(answers=None, cache=True, gate=None, **kw):
    users = MemoryUserStore()
    for name, role in (("alice", "agent"), ("bob", "agent2"), ("carol", "admin")):
        users.upsert(name, hash_password(f"{name}-pw"), role)
    version = {"v": "1"}
    svc = ApiService(
        answers or FakeAnswers(), users, ROLES, secret=SECRET,
        cache=AnswerCache(MemoryTTLCache(), ttl=60) if cache else None,
        limiter=kw.pop("limiter", MemoryRateLimiter()), gate=gate,
        corpus_version=lambda: version["v"], model_id="m1", **kw)
    svc.users_, svc.version_ = users, version
    return svc


def who(name="alice", role="agent"):
    return Principal(name, role)


# ------------------------------------------------------------------------ login
def test_login_returns_a_token_that_identifies_the_user():
    out = make().login("Alice ", "alice-pw", "1.2.3.4")
    assert decode_token(out["access_token"], SECRET) == "alice"
    assert (out["role"], out["token_type"], out["expires_in"]) == ("agent", "bearer", 3600)


def test_wrong_password_and_unknown_user_look_identical():
    svc = make()
    errors = []
    for user, pw in (("alice", "nope"), ("nobody", "alice-pw"), ("", "x")):
        with pytest.raises(ApiError) as e:
            svc.login(user, pw, "ip")
        errors.append((e.value.status, e.value.code, e.value.message))
    assert len(set(errors)) == 1 and errors[0][:2] == (401, "invalid_credentials")


def test_unknown_users_still_cost_one_password_verification(monkeypatch):
    calls = []
    real = api_core.verify_password
    monkeypatch.setattr(api_core, "verify_password", lambda p, h: calls.append(h) or real(p, h))
    svc = make()
    for user in ("alice", "ghost"):
        with pytest.raises(ApiError):
            svc.login(user, "wrong", "ip")
    assert len(calls) == 2                              # no early exit for the unknown user


def test_disabled_account_cannot_log_in():
    svc = make()
    svc.users_.set_disabled("alice", True)
    with pytest.raises(ApiError) as e:
        svc.login("alice", "alice-pw", "ip")
    assert e.value.status == 401


def test_repeated_failures_for_one_username_are_throttled_even_with_the_right_password():
    svc = make(login_limit=Limit(3, 60))
    for _ in range(3):
        with pytest.raises(ApiError):
            svc.login("alice", "bad", "ip")
    with pytest.raises(ApiError) as e:
        svc.login("alice", "alice-pw", "ip")
    assert e.value.status == 429 and e.value.retry_after and e.value.headers["Retry-After"]


def test_one_ip_guessing_many_usernames_is_throttled():
    svc = make(ip_login_limit=Limit(4, 60))
    for i in range(4):
        with pytest.raises(ApiError):
            svc.login(f"user{i}", "bad", "6.6.6.6")
    with pytest.raises(ApiError) as e:
        svc.login("alice", "alice-pw", "6.6.6.6")
    assert e.value.status == 429
    assert make().login("alice", "alice-pw", "7.7.7.7")                   # other IPs unaffected


# ------------------------------------------------------------------ authenticate
def test_authenticate_accepts_a_valid_token_and_reads_the_role_fresh():
    svc = make()
    token = svc.login("alice", "alice-pw")["access_token"]
    assert svc.authenticate(token) == Principal("alice", "agent")
    svc.users_.upsert("alice", hash_password("alice-pw"), "admin")        # promoted after login
    assert svc.authenticate(token).role == "admin"


@pytest.mark.parametrize("token", [None, "", "garbage", "a.b.c"])
def test_missing_or_garbage_tokens_are_401(token):
    with pytest.raises(ApiError) as e:
        make().authenticate(token)
    assert e.value.status == 401


def test_expired_token_deleted_user_and_disabled_user_are_401():
    svc = make()
    old, _ = issue_token("alice", SECRET, 1, now=time.time() - 7200)
    ghost, _ = issue_token("ghost", SECRET)
    for token in (old, ghost):
        with pytest.raises(ApiError) as e:
            svc.authenticate(token)
        assert e.value.status == 401
    live = svc.login("alice", "alice-pw")["access_token"]
    svc.users_.set_disabled("alice", True)
    with pytest.raises(ApiError) as e:
        svc.authenticate(live)
    assert e.value.status == 401


def test_a_user_whose_role_is_not_configured_gets_403():
    svc = make()
    svc.users_.upsert("dave", hash_password("pw"), "wizard")
    token, _ = issue_token("dave", SECRET)
    with pytest.raises(ApiError) as e:
        svc.authenticate(token)
    assert e.value.status == 403


# -------------------------------------------------------------------- validation
@pytest.mark.parametrize("q,code", [("", "question_required"), ("   \n", "question_required"),
                                    (None, "question_required"), ("x" * 501, "question_too_long")])
def test_bad_questions_are_422(q, code):
    with pytest.raises(ApiError) as e:
        make().ask(who(), q)
    assert (e.value.status, e.value.code) == (422, code)


def test_ask_is_rate_limited_per_user_with_a_retry_hint():
    svc = make(ask_limit=Limit(2, 60), cache=False)
    svc.ask(who(), "q1")
    svc.ask(who(), "q2")
    with pytest.raises(ApiError) as e:
        svc.ask(who(), "q3")
    assert e.value.status == 429 and e.value.retry_after
    assert svc.ask(who("bob", "agent2"), "q1")["status"] == "answered"    # other users unaffected


def test_rate_limit_applies_to_cached_answers_too():
    svc = make(ask_limit=Limit(2, 60))
    svc.ask(who(), "same")
    svc.ask(who(), "same")
    with pytest.raises(ApiError):
        svc.ask(who(), "same")


# ----------------------------------------------------------------------- caching
def test_second_identical_question_is_served_from_cache_without_the_model():
    fake = FakeAnswers()
    svc = make(fake)
    first = svc.ask(who(), "How do I reset a password?")
    again = svc.ask(who(), "  how do i reset a PASSWORD ")
    assert len(fake.calls) == 1 and first["cached"] is False and again["cached"] is True
    assert again["text"] == first["text"] and again["citations"] == first["citations"]


def test_cached_stream_replays_sources_text_and_done():
    svc = make()
    list(svc.ask_stream(who(), "q"))
    events = list(svc.ask_stream(who(), "q"))
    assert [e["type"] for e in events] == ["sources", "token", "done"]
    assert events[-1]["answer"]["cached"] is True and events[0]["sources"][0]["path"] == "docs/a.md"


def test_an_admins_cached_answer_is_never_served_to_an_agent():
    fake = FakeAnswers()
    svc = make(fake)
    admin_answer = svc.ask(who("carol", "admin"), "what is the leave policy")
    assert "RESTRICTED" in admin_answer["text"]
    agent_answer = svc.ask(who("alice", "agent"), "what is the leave policy")
    assert "RESTRICTED" not in json.dumps(agent_answer) and agent_answer["cached"] is False
    assert [c[1] for c in fake.calls] == ["admin", "agent"]
    # and the reverse direction: the agent's narrower answer is not reused for the admin
    assert svc.ask(who("carol", "admin"), "what is the leave policy")["text"].startswith("RESTRICTED")


def test_roles_with_identical_permissions_share_cache_entries():
    fake = FakeAnswers()
    svc = make(fake)
    svc.ask(who("alice", "agent"), "shared question")
    assert svc.ask(who("bob", "agent2"), "shared question")["cached"] is True and len(fake.calls) == 1


def test_a_sync_that_changes_the_corpus_invalidates_cached_answers():
    fake = FakeAnswers()
    svc = make(fake)
    svc.ask(who(), "q")
    svc.version_["v"] = "2"                                                # documents changed
    assert svc.ask(who(), "q")["cached"] is False and len(fake.calls) == 2


@pytest.mark.parametrize("status", ["ungrounded", "llm_unavailable"])
def test_failures_are_never_cached(status):
    fake = FakeAnswers(status=status)
    svc = make(fake)
    svc.ask(who(), "q")
    svc.ask(who(), "q")
    assert len(fake.calls) == 2


def test_abstentions_are_cached():
    fake = FakeAnswers(status="abstained", reason="low_relevance", text="I couldn't find that.")
    svc = make(fake)
    svc.ask(who(), "q")
    assert svc.ask(who(), "q")["cached"] is True and len(fake.calls) == 1


# ------------------------------------------------------------------ public payload
def test_clients_never_see_raw_model_output_or_internal_error_details():
    fake = FakeAnswers(status="llm_unavailable", reason="LLMUnavailable: Cannot reach Ollama at http://localhost:11434")
    out = make(fake).ask(who(), "q")
    blob = json.dumps(out)
    assert "RAW-UNVALIDATED" not in blob and "localhost" not in blob and "dense_top" not in blob
    assert out["reason"] is None and out["related"]                      # related pages still offered


def test_abstain_reasons_are_exposed_only_when_they_are_safe_codes():
    safe = Answer("q", "agent", "abstained", "no", reason="low_relevance")
    odd = Answer("q", "agent", "abstained", "no", reason="Traceback: secret path")
    assert public_answer(safe)["reason"] == "low_relevance" and public_answer(odd)["reason"] is None


# ---------------------------------------------------------------------- streaming
def test_stream_event_order_and_shapes():
    events = list(make(cache=False).ask_stream(who(), "q"))
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    assert events[0]["sources"][0] == {"n": 1, "title": "Doc 1", "section": "Sec",
                                       "path": "docs/a.md", "snippet": "snippet"}
    assert events[-1]["answer"]["status"] == "answered"


def test_sse_frames_are_single_data_lines_even_for_multiline_text():
    frame = sse({"type": "token", "text": "line one\nline two é"})
    assert frame.startswith("event: token\ndata: ") and frame.endswith("\n\n")
    assert frame.count("\n") == 3                                         # event, data, blank
    assert json.loads(frame.split("data: ", 1)[1])["text"] == "line one\nline two é"


# ---------------------------------------------------------- concurrency / busy / leaks
def test_closing_a_stream_midway_frees_the_generation_slot():
    gate = GenerationGate(1, 2, 1)
    gen = make(cache=False, gate=gate).ask_stream(who(), "q")
    assert next(gen)["type"] == "sources" and gate.in_flight == 1
    gen.close()                                                           # client disconnected
    assert gate.in_flight == 0


def test_a_stream_that_is_never_read_cannot_leak_a_slot():
    gate = GenerationGate(1, 2, 1)
    gen = make(cache=False, gate=gate).ask_stream(who(), "q")
    del gen
    assert gate.in_flight == 0


def test_saturated_server_rejects_before_streaming_with_503():
    gate = GenerationGate(1, 0, 0.01)                                     # no waiting room
    svc = make(cache=False, gate=gate)
    with gate.acquire():
        with pytest.raises(ApiError) as e:
            svc.ask_stream(who(), "q")
    assert (e.value.status, e.value.code) == (503, "busy") and e.value.retry_after


def test_timing_out_in_the_queue_becomes_a_busy_event_and_a_503_for_non_streaming():
    gate = GenerationGate(1, 2, 0.02)
    svc = make(cache=False, gate=gate)
    with gate.acquire():
        events = list(svc.ask_stream(who(), "q"))
        assert events[-1]["type"] == "error" and events[-1]["code"] == "busy"
        with pytest.raises(ApiError) as e:
            svc.ask(who(), "q")
    assert e.value.status == 503 and gate.in_flight == 0


def test_cached_answers_are_served_even_when_the_model_is_saturated():
    gate = GenerationGate(1, 0, 0.01)
    svc = make(gate=gate)
    svc.ask(who(), "popular question")
    with gate.acquire():
        assert svc.ask(who(), "popular question")["cached"] is True


def test_unexpected_failures_become_a_generic_error_event():
    class Boom:
        def stream(self, q, role):
            raise RuntimeError("secret internal detail")
            yield

    events = list(make(Boom(), cache=False).ask_stream(who(), "q"))
    assert events[-1]["code"] == "internal" and "secret" not in json.dumps(events)


# ----------------------------------------------------------------------- auditing
def test_audit_log_records_who_and_what_happened_but_not_the_question(caplog):
    caplog.set_level(logging.INFO, logger="rag.audit")
    make().ask(who(), "customer jane.doe@example.com lost her API token")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "user=alice" in text and "status=answered" in text and "qhash=" in text
    assert "jane.doe" not in text and "API token" not in text


def test_failed_logins_are_audited_without_the_password(caplog):
    caplog.set_level(logging.INFO, logger="rag.audit")
    with pytest.raises(ApiError):
        make().login("alice", "hunter2-attempt", "9.9.9.9")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "login_failed" in text and "user=alice" in text and "hunter2" not in text


# ----------------------------------------------------------------------- readiness
def test_readiness_reports_each_dependency_by_name_without_error_details():
    def bad():
        raise ConnectionError("password=hunter2 host=10.0.0.5")

    svc = make(readiness_checks={"qdrant": lambda: None, "ollama": bad})
    assert svc.readiness() == {"qdrant": "ok", "ollama": "ConnectionError"}
