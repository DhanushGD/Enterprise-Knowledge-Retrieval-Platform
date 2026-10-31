import hashlib
from types import SimpleNamespace

import pytest

from eval.retrieval_eval import Case, evaluate, format_scores, rank_of
from ingestion.records import Embedding
from serving.retrieval import Retriever, UnknownRoleError
from tests.fakes import FakeEmbedder

ROLES = {
    "agent": {"allowed_access_levels": ["public"], "allowed_departments": ["support", "product"]},
    "admin": {"allowed_access_levels": ["public", "internal", "restricted"],
              "allowed_departments": ["support", "product", "hr"]},
    "locked": {"allowed_access_levels": [], "allowed_departments": ["support"]},
}


def words(text):
    return {int(hashlib.md5(w.encode()).hexdigest()[:6], 16) for w in text.lower().split()}


def point(i, doc, text, level="public", dept="support", section="Intro"):
    return SimpleNamespace(id=i, payload={
        "chunk_id": f"{doc}-{i}", "doc_id": doc, "doc_title": doc.title(), "section": section,
        "source_path": f"{doc}.md", "text": text, "access_level": level, "department": dept,
        "product": "core", "page": None,
    }, score=0.0)


class SpyIndex:
    """Scores by word overlap, honours permission filters the way Qdrant would, records calls."""

    def __init__(self, points):
        self.points = points
        self.calls = []

    def _run(self, kind, embedding, limit, access_levels, departments):
        self.calls.append((kind, access_levels, departments))
        q = set(embedding.sparse_indices)
        out = []
        for p in self.points:
            if access_levels is not None and p.payload["access_level"] not in access_levels:
                continue
            if departments is not None and p.payload["department"] not in departments:
                continue
            overlap = len(q & words(p.payload["text"]))
            out.append(SimpleNamespace(id=p.id, payload=p.payload, score=float(overlap)))
        out.sort(key=lambda p: -p.score)
        return out[:limit]

    def dense_search(self, e, limit=30, access_levels=None, departments=None):
        return self._run("dense", e, limit, access_levels, departments)

    def sparse_search(self, e, limit=30, access_levels=None, departments=None):
        return self._run("sparse", e, limit, access_levels, departments)

    def hybrid_search(self, e, limit=30, access_levels=None, departments=None):
        return self._run("hybrid", e, limit, access_levels, departments)


class KeywordReranker:
    """Prefers passages containing 'escalate'; records what it was asked to score."""

    def __init__(self):
        self.seen = []

    def rerank(self, query, passages):
        self.seen.append(passages)
        return [10.0 if "escalate" in p else 1.0 for p in passages]


class BrokenReranker:
    def rerank(self, query, passages):
        raise TimeoutError("model server down")


CORPUS = [
    point(1, "refunds", "refund a customer payment quickly"),
    point(2, "escalation", "escalate a ticket to tier two engineers"),
    point(3, "leave", "vacation leave policy for employees", level="restricted", dept="hr"),
    point(4, "sla", "ticket response times for internal staff only", level="internal"),
]


def make(points=CORPUS, reranker=None, **kw):
    index = SpyIndex(points)
    return Retriever(FakeEmbedder(), index, ROLES, reranker, **kw), index


# ----------------------------------------------------------------- permissions
def test_role_filters_are_passed_to_the_search_itself():
    r, index = make()
    r.retrieve("ticket", "agent")
    kind, levels, depts = index.calls[0]
    assert kind == "hybrid" and levels == ["public"] and depts == ["support", "product"]


def test_agent_never_sees_restricted_or_internal_text_admin_does():
    r, _ = make()
    agent_paths = {h.source_path for h in r.retrieve("vacation leave policy ticket staff", "agent").hits}
    admin_paths = {h.source_path for h in r.retrieve("vacation leave policy ticket staff", "admin").hits}
    assert not agent_paths & {"leave.md", "sla.md"}
    assert {"leave.md", "sla.md"} <= admin_paths


def test_unknown_role_is_an_error_and_searches_nothing():
    r, index = make()
    with pytest.raises(UnknownRoleError):
        r.retrieve("anything", "intern")
    assert index.calls == []


def test_role_with_empty_allow_list_fails_closed_without_searching():
    r, index = make()
    result = r.retrieve("ticket", "locked")
    assert result.hits == [] and index.calls == []


def test_blank_query_returns_nothing_and_does_no_work():
    r, index = make()
    assert r.retrieve("   ", "agent").hits == [] and index.calls == []


# ------------------------------------------------------------------- modes
def test_each_mode_calls_the_matching_search():
    r, index = make(reranker=KeywordReranker())
    for mode, kind in (("dense", "dense"), ("sparse", "sparse"), ("hybrid", "hybrid"),
                       ("hybrid_rerank", "hybrid")):
        index.calls.clear()
        r.retrieve("ticket", "agent", mode=mode)
        assert index.calls[0][0] == kind


def test_default_mode_depends_on_whether_a_reranker_is_configured():
    assert make()[0].retrieve("ticket", "agent").mode == "hybrid"
    assert make(reranker=KeywordReranker())[0].retrieve("ticket", "agent").mode == "hybrid_rerank"


def test_bad_mode_and_rerank_without_reranker_are_rejected():
    r, _ = make()
    with pytest.raises(ValueError):
        r.retrieve("x", "agent", mode="magic")
    with pytest.raises(ValueError):
        r.retrieve("x", "agent", mode="hybrid_rerank")


# ----------------------------------------------------------------- reranking
def test_reranker_reorders_the_candidates():
    # "refund a customer payment ticket" overlaps the refunds chunk most; the reranker still
    # promotes the escalation chunk.
    r, _ = make(reranker=KeywordReranker())
    query = "refund customer payment quickly ticket"
    assert r.retrieve(query, "agent", mode="hybrid").hits[0].source_path == "refunds.md"
    top = r.retrieve(query, "agent", mode="hybrid_rerank").hits[0]
    assert top.source_path == "escalation.md" and top.rerank_score == 10.0


def test_reranker_sees_title_and_section_framing():
    rr = KeywordReranker()
    make(reranker=rr)[0].retrieve("ticket escalate", "agent")
    assert any(p.startswith("Escalation > Intro\n\n") for p in rr.seen[0])


def test_reranker_failure_degrades_instead_of_failing():
    r, _ = make(reranker=BrokenReranker())
    result = r.retrieve("escalate a ticket", "agent")
    assert result.hits and "TimeoutError" in result.degraded
    assert all(h.rerank_score is None for h in result.hits)


def test_reranker_returning_wrong_count_is_treated_as_failure():
    class Short:
        def rerank(self, q, passages):
            return [1.0]

    result = make(reranker=Short())[0].retrieve("escalate a ticket", "agent")
    assert result.degraded and result.hits

    class Long:                    # extra scores would silently misalign with zip()
        def rerank(self, q, passages):
            return [5.0] * (len(passages) + 3)

    result = make(reranker=Long())[0].retrieve("escalate a ticket", "agent")
    assert result.degraded and all(h.rerank_score is None for h in result.hits)


# ------------------------------------------------------------------ diversity
def test_at_most_max_per_doc_chunks_from_one_document():
    pts = [point(i, "bigdoc", "reset password steps part", section=f"S{i}") for i in range(6)]
    pts.append(point(9, "other", "reset password steps part"))
    r, _ = make(pts, max_per_doc=2, top_k=5)
    hits = r.retrieve("reset password steps", "agent").hits
    assert sum(h.doc_id == "bigdoc" for h in hits) == 2 and any(h.doc_id == "other" for h in hits)


def test_top_k_limits_results_and_timings_are_reported():
    pts = [point(i, f"doc{i}", "common words everywhere") for i in range(10)]
    result = make(pts, top_k=3)[0].retrieve("common words", "agent")
    assert len(result.hits) == 3
    assert {"embed_ms", "search_ms", "total_ms"} <= set(result.timings)
    assert result.candidates == 10


# --------------------------------------------------------------- the eval itself
def test_rank_counts_distinct_documents_and_matches_substrings():
    case = Case("q", ["escalat"])
    assert rank_of(case, ["a.md", "a.md", "escalation.md"]) == 2     # repeat chunks count once
    assert rank_of(case, ["a.md", "b.md"]) is None
    assert rank_of(Case("q", ["X.MD"]), ["dir/x.md"]) == 1            # case-insensitive


def test_evaluate_scores_modes_and_lists_misses():
    r, _ = make(reranker=KeywordReranker())
    cases = [Case("escalate a ticket", ["escalation"]), Case("vacation leave", ["leave"])]
    scores = evaluate(r, cases, "agent", ["hybrid", "hybrid_rerank"])
    s = scores["hybrid_rerank"]
    assert s.n == 2 and s.hit1 == 1                                   # agent can't see "leave"
    assert [c.question for c, _ in s.misses] == ["vacation leave"]
    assert "hybrid_rerank" in format_scores(scores) and "50%" in format_scores(scores)


# ----------------------------------------------- real Qdrant (runs where installed)
def test_qdrant_hybrid_search_fuses_and_enforces_permissions():
    qdrant = pytest.importorskip("qdrant_client")
    from core.vectorstore import QdrantIndex
    from ingestion.records import IndexPoint
    from tests.fakes import FakeEmbedder
    import uuid

    emb = FakeEmbedder()
    index = QdrantIndex(client=qdrant.QdrantClient(":memory:"), collection="t")
    index.recreate(8)
    texts = {
        "refund a customer payment": ("public", "support"),
        "escalate a ticket to tier two": ("public", "support"),
        "vacation leave policy": ("restricted", "hr"),
    }
    index.upsert([
        IndexPoint(str(uuid.uuid5(uuid.NAMESPACE_URL, t)), emb.embed_query(t),
                   {"doc_id": t, "text": t, "access_level": lvl, "department": dep,
                    "doc_title": t, "section": "", "source_path": t, "chunk_id": t, "product": "core"})
        for t, (lvl, dep) in texts.items()
    ])
    q = emb.embed_query("vacation leave policy")
    agent = index.hybrid_search(q, limit=10, access_levels=["public"], departments=["support"])
    admin = index.hybrid_search(q, limit=10, access_levels=["public", "restricted"],
                                departments=["support", "hr"])
    assert "vacation leave policy" not in {p.payload["text"] for p in agent}
    assert admin[0].payload["text"] == "vacation leave policy"
    assert index.sparse_search(Embedding([0.0] * 8, [], []), access_levels=["public"]) == []


# ------------------------------------------- eval: visibility and permission-leak checks
def doc(path, level="public", dept="support", status="indexed"):
    return SimpleNamespace(source_path=path, access_level=level, department=dept, status=status)


def test_classify_separates_answerable_hidden_and_missing():
    from eval.retrieval_eval import ANSWERABLE, HIDDEN, NO_DOC, classify

    docs = [doc("a/escalation.md"), doc("b/leave.md", "restricted", "hr"), doc("c/old.md", status="deleted")]
    agent, admin = ROLES["agent"], ROLES["admin"]
    assert classify(Case("q", ["escalation"]), docs, agent) == ANSWERABLE
    assert classify(Case("q", ["leave"]), docs, agent) == HIDDEN
    assert classify(Case("q", ["leave"]), docs, admin) == ANSWERABLE
    assert classify(Case("q", ["nothing"]), docs, agent) == NO_DOC
    assert classify(Case("q", ["old.md"]), docs, agent) == NO_DOC          # deleted pages don't count
    # a wrong department hides a page even when the access level is fine
    assert classify(Case("q", ["x"]), [doc("x.md", "public", "engineering")], agent) == HIDDEN


def test_evaluate_excludes_hidden_cases_from_scores_and_counts_them():
    from eval.retrieval_eval import classify

    docs = [doc("escalation.md"), doc("leave.md", "restricted", "hr")]
    cases = [Case("escalate a ticket", ["escalation"]), Case("vacation leave", ["leave"]),
             Case("something else", ["ghost"])]
    r, _ = make()
    scores = evaluate(r, cases, "agent", ["hybrid"], classifier=lambda c: classify(c, docs, ROLES["agent"]))
    s = scores["hybrid"]
    assert (s.n, s.hit1, s.hidden, s.no_doc, s.leaks) == (1, 1, 1, 1, [])
    text = format_scores(scores)
    assert "PERMISSION LEAKS: 0" in text and "100%" in text


def test_a_hidden_page_that_is_retrieved_is_reported_as_a_leak():
    class LeakyIndex(SpyIndex):
        def hybrid_search(self, e, limit=30, access_levels=None, departments=None):
            return self._run("hybrid", e, limit, None, None)         # ignores the permission filter

    from eval.retrieval_eval import classify

    index = LeakyIndex(CORPUS)
    r = Retriever(FakeEmbedder(), index, ROLES)
    docs = [doc("leave.md", "restricted", "hr")]
    scores = evaluate(r, [Case("vacation leave policy", ["leave"])], "agent", ["hybrid"],
                      classifier=lambda c: classify(c, docs, ROLES["agent"]))
    assert len(scores["hybrid"].leaks) == 1
    assert "PERMISSION LEAKS: 1 !!" in format_scores(scores)


# --------------------------------------------- dense_top: the scale-stable relevance signal
def test_dense_top_comes_from_the_dense_ranking_in_every_mode_that_has_one():
    r, index = make(reranker=KeywordReranker())
    for mode in ("dense", "hybrid", "hybrid_rerank"):
        result = r.retrieve("escalate a ticket", "agent", mode=mode)
        assert result.dense_top == 3.0, mode               # word-overlap score of the best chunk
    assert r.retrieve("escalate a ticket", "agent", mode="sparse").dense_top is None


def test_dense_top_respects_the_role_filter():
    r, _ = make()
    # the only strong match is restricted: an agent must not get a high score because of it
    assert r.retrieve("vacation leave policy employees", "agent").dense_top < 2
    assert r.retrieve("vacation leave policy employees", "admin").dense_top == 4.0


def test_failing_dense_probe_does_not_break_retrieval():
    class Flaky(SpyIndex):
        def dense_search(self, *a, **k):
            raise TimeoutError("qdrant hiccup")

    r = Retriever(FakeEmbedder(), Flaky(CORPUS), ROLES)
    result = r.retrieve("escalate a ticket", "agent", mode="hybrid")
    assert result.hits and result.dense_top is None


def test_empty_results_have_no_dense_top():
    assert make([])[0].retrieve("anything", "agent").dense_top is None
