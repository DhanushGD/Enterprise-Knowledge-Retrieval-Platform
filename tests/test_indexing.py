import uuid
from pathlib import Path

from connectors.local_folder import LocalFolderConnector
from ingestion.chunking import build_embed_text
from ingestion.indexer import IndexingService, point_id
from ingestion.landing import LandingService
from ingestion.pipeline import ProcessingService
from tests.fakes import FakeEmbedder, FakeIndex, FakeRegistry, FakeStore

TAXONOMY = {
    "department": ["support", "product", "engineering", "hr"],
    "access_level": ["public", "internal", "restricted"],
    "product": ["core", "billing", "integrations", "security"],
}


def body(tag, n=15):
    return " ".join(f"{tag} sentence {i} explains how the process works in detail." for i in range(n))


def doc_md(title, sections):
    parts = [f"# {title}"]
    for name in sections:
        parts.append(f"## {name}\n\n{body(name)}")
    return "\n\n".join(parts).encode()


def build(tmp_path, files, embedder=None):
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    store, registry, index = FakeStore(), FakeRegistry(), FakeIndex()
    connector = LocalFolderConnector("docs", tmp_path, default_department="support")
    LandingService(connector, store, registry, TAXONOMY).run()
    processing = ProcessingService(store, registry)
    processing.parse_pending()
    processing.chunk_pending()
    embedder = embedder or FakeEmbedder()
    return store, registry, index, embedder, IndexingService(registry, embedder, index), connector, processing


def test_point_id_is_deterministic_valid_uuid():
    assert point_id("abc-0001") == point_id("abc-0001")
    assert point_id("abc-0001") != point_id("abc-0002")
    uuid.UUID(point_id("abc-0001"))


def test_build_embed_text_adds_context_header_without_duplication():
    assert build_embed_text("Guide", "Runner > Install", "Run it.") == "Guide > Runner > Install\n\nRun it."
    assert build_embed_text("Runner", "Runner > Install", "Run it.") == "Runner > Install\n\nRun it."
    assert build_embed_text("Guide", "", "Run it.") == "Guide\n\nRun it."


def test_indexes_every_chunk_with_payload_and_marks_indexed(tmp_path):
    _, registry, index, embedder, service, *_ = build(
        tmp_path, {"support/public/refunds.md": doc_md("Refunds", ["Policy", "Process", "Limits"])}
    )
    result = service.index_pending()
    assert (result.ok, result.failed) == (1, 0)
    assert index.dimension == 8
    assert {r.status for r in registry.rows.values()} == {"indexed"}
    assert index.count() == 3

    payloads = [p.payload for p in index.points.values()]
    first = next(p for p in payloads if p["section"] == "Refunds > Policy")
    assert first["doc_title"] == "Refunds"
    assert first["source_name"] == "docs" and first["source_path"] == "support/public/refunds.md"
    assert first["department"] == "support" and first["access_level"] == "public"
    assert first["doc_version"] == 1 and first["chunk_id"].endswith("-0000")
    assert "Policy sentence 0" in first["text"]


def test_embedding_input_carries_the_heading_context(tmp_path):
    _, _, _, embedder, service, *_ = build(
        tmp_path, {"a.md": doc_md("Refunds", ["Policy", "Process"])}
    )
    service.index_pending()
    assert any(t.startswith("Refunds > Policy\n\n") for t in embedder.seen)
    assert any(t.startswith("Refunds > Process\n\n") for t in embedder.seen)


def test_second_run_is_a_noop(tmp_path):
    _, _, index, embedder, service, *_ = build(tmp_path, {"a.md": doc_md("A", ["One", "Two"])})
    service.index_pending()
    seen_before = len(embedder.seen)
    assert service.index_pending().ok == 0
    assert len(embedder.seen) == seen_before and index.count() == 2


def test_failure_isolated_and_retried(tmp_path):
    _, registry, index, _, _, *_ = build(tmp_path, {
        "good.md": doc_md("Good", ["One"]),
        "bad.md": doc_md("Bad", ["Explodes"]),
    })
    service = IndexingService(registry, FakeEmbedder(fail_on="Explodes"), index)
    result = service.index_pending()
    assert (result.ok, result.failed) == (1, 1)
    assert "bad.md" in result.failures[0] and "model exploded" in result.failures[0]
    statuses = {r.source_path: r.status for r in registry.rows.values()}
    assert statuses == {"good.md": "indexed", "bad.md": "index_failed"}

    healed = IndexingService(registry, FakeEmbedder(), index).index_pending()   # model fixed
    assert (healed.ok, healed.failed) == (1, 0)
    assert {r.status for r in registry.rows.values()} == {"indexed"}


def test_changed_doc_replaces_points_and_removes_stale_ones(tmp_path):
    store, registry, index, embedder, service, connector, processing = build(
        tmp_path, {"a.md": doc_md("A", ["One", "Two", "Three"])}
    )
    service.index_pending()
    assert index.count() == 3
    old_ids = set(index.points)

    (tmp_path / "a.md").write_bytes(doc_md("A", ["One"]))        # document shrinks to 1 section
    LandingService(connector, store, registry, TAXONOMY).run()
    processing.parse_pending()
    processing.chunk_pending()
    assert service.index_pending().ok == 1

    assert index.count() == 1                                    # stale chunks gone
    only = next(iter(index.points.values()))
    assert only.payload["doc_version"] == 2
    assert only.point_id in old_ids                              # same chunk id -> overwritten, not duplicated


def test_limit(tmp_path):
    _, registry, _, _, service, *_ = build(tmp_path, {
        "a.md": doc_md("A", ["x"]), "b.md": doc_md("B", ["x"]), "c.md": doc_md("C", ["x"]),
    })
    assert service.index_pending(limit=2).ok == 2
    assert sorted(r.status for r in registry.rows.values()) == ["chunked", "indexed", "indexed"]


def test_bundled_sample_docs_index_cleanly():
    sample = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    store, registry, index = FakeStore(), FakeRegistry(), FakeIndex()
    connector = LocalFolderConnector("docs", sample, default_department="support")
    LandingService(connector, store, registry, TAXONOMY).run()
    processing = ProcessingService(store, registry)
    processing.parse_pending(); processing.chunk_pending()
    result = IndexingService(registry, FakeEmbedder(), index).index_pending()
    assert (result.ok, result.failed) == (5, 0)
    levels = {p.payload["source_path"]: p.payload["access_level"] for p in index.points.values()}
    assert levels["hr/restricted/leave-policy.md"] == "restricted"
    assert levels["support/public/password-reset.md"] == "public"
