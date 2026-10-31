from pathlib import Path

from connectors.local_folder import LocalFolderConnector
from ingestion.landing import LandingService
from ingestion.parsing import ParsedDocument
from ingestion.pipeline import ProcessingService, parsed_key
from tests.fakes import FakeRegistry, FakeStore

TAXONOMY = {
    "department": ["support", "product", "engineering", "hr"],
    "access_level": ["public", "internal", "restricted"],
    "product": ["core", "billing", "integrations", "security"],
}


def land(root: Path, store, registry):
    connector = LocalFolderConnector("docs", root, default_department="support")
    return LandingService(connector, store, registry, TAXONOMY).run()


def make_env(tmp_path, files: dict[str, bytes]):
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    store, registry = FakeStore(), FakeRegistry()
    land(tmp_path, store, registry)
    return store, registry, ProcessingService(store, registry)


def statuses(registry):
    return {r.source_path: r.status for r in registry.rows.values()}


def test_documents_move_new_to_parsed_to_chunked(tmp_path):
    store, registry, svc = make_env(tmp_path, {
        "support/refunds.md": b"# Refunds\n\n## Policy\n\nRefund within 14 days.",
        "notes.txt": b"plain notes",
    })
    assert set(statuses(registry).values()) == {"new"}

    parsed = svc.parse_pending()
    assert (parsed.ok, parsed.failed) == (2, 0)
    assert set(statuses(registry).values()) == {"parsed"}
    stored = ParsedDocument.from_json(store.objects[parsed_key("docs", "support/refunds.md")].decode())
    assert stored.title == "Refunds" and stored.sections[0].heading_path == ["Refunds", "Policy"]

    chunked = svc.chunk_pending()
    assert (chunked.ok, chunked.failed) == (2, 0)
    assert set(statuses(registry).values()) == {"chunked"}
    title, chunks = next(v for v in registry.chunks.values() if v[0] == "Refunds")
    assert chunks[0].section == "Refunds > Policy"


def test_second_run_is_a_noop(tmp_path):
    _, registry, svc = make_env(tmp_path, {"a.md": b"# A\n\ntext"})
    svc.parse_pending(); svc.chunk_pending()
    assert svc.parse_pending().ok == 0 and svc.chunk_pending().ok == 0


def test_bad_document_fails_alone_and_is_retried_later(tmp_path):
    store, registry, svc = make_env(tmp_path, {
        "good.md": b"# Good\n\nfine text",
        "bad.docx": b"not really a docx",
    })
    parsed = svc.parse_pending()
    assert (parsed.ok, parsed.failed) == (1, 1)
    assert "bad.docx" in parsed.failures[0]
    assert statuses(registry) == {"good.md": "parsed", "bad.docx": "parse_failed"}

    svc.chunk_pending()
    assert statuses(registry)["good.md"] == "chunked"
    assert svc.parse_pending().failed == 1               # failed docs are retried each run


def test_changed_document_is_reprocessed_and_chunks_replaced(tmp_path):
    store, registry, svc = make_env(tmp_path, {"a.md": b"# A\n\noriginal text"})
    svc.parse_pending(); svc.chunk_pending()
    doc_id = next(iter(registry.rows))
    assert "original text" in registry.chunks[doc_id][1][0].text

    (tmp_path / "a.md").write_bytes(b"# A\n\nbrand new text")
    land(tmp_path, store, registry)
    assert statuses(registry) == {"a.md": "changed"}
    svc.parse_pending(); svc.chunk_pending()
    texts = [c.text for c in registry.chunks[doc_id][1]]
    assert texts == ["brand new text"]


def test_limit(tmp_path):
    _, registry, svc = make_env(tmp_path, {"a.md": b"# A\n\nx", "b.md": b"# B\n\ny", "c.md": b"# C\n\nz"})
    assert svc.parse_pending(limit=2).ok == 2
    assert sorted(statuses(registry).values()) == ["new", "parsed", "parsed"]


def test_chunking_config_is_applied(tmp_path):
    long_text = " ".join(f"Sentence {i} is here." for i in range(300))
    store, registry, _ = make_env(tmp_path, {"long.md": f"# Long\n\n{long_text}".encode()})
    svc = ProcessingService(store, registry, {"max_tokens": 100, "overlap_tokens": 10})
    svc.parse_pending(); svc.chunk_pending()
    chunks = next(iter(registry.chunks.values()))[1]
    assert len(chunks) > 5 and max(c.token_count for c in chunks) <= 100


def test_bundled_sample_docs_process_cleanly():
    sample = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    store, registry = FakeStore(), FakeRegistry()
    land(sample, store, registry)
    svc = ProcessingService(store, registry)
    assert svc.parse_pending().failed == 0
    assert svc.chunk_pending().failed == 0
    assert set(statuses(registry).values()) == {"chunked"}
    titles = {title for title, _ in registry.chunks.values()}
    assert "Resetting a customer's password" in titles
