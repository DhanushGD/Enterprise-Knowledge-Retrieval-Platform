from dataclasses import replace

import pytest

from connectors.local_folder import LocalFolderConnector
from ingestion.landing import LandingService

TAXONOMY = {
    "department": ["support", "product", "engineering", "hr"],
    "access_level": ["public", "internal", "restricted"],
    "product": ["core", "billing", "integrations", "security"],
}


class FakeStore:
    def __init__(self):
        self.objects = {}
        self.ensured = False

    def ensure(self):
        self.ensured = True

    def put(self, key, data, content_type=None):
        self.objects[key] = data


class FakeRegistry:
    def __init__(self):
        self.rows = {}
        self.runs = []

    def get(self, doc_id):
        row = self.rows.get(doc_id)
        return replace(row) if row else None

    def upsert(self, rec):
        self.rows[rec.doc_id] = rec

    def log_run(self, source_name, result, started, finished):
        self.runs.append((source_name, result))


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "support").mkdir()
    (tmp_path / "support" / "billing-refunds.md").write_text("refund text")
    (tmp_path / "hr").mkdir()
    (tmp_path / "hr" / "leave.md").write_text("leave text")
    connector = LocalFolderConnector("docs", tmp_path, default_department="support")
    store, registry = FakeStore(), FakeRegistry()
    service = LandingService(connector, store, registry, TAXONOMY)
    return tmp_path, service, store, registry


def test_first_run_lands_everything(setup):
    _, service, store, registry = setup
    r = service.run()
    assert (r.new, r.changed, r.unchanged, r.failed) == (2, 0, 0, 0)
    assert store.ensured
    assert set(store.objects) == {"raw/docs/support/billing-refunds.md", "raw/docs/hr/leave.md"}
    assert store.objects["raw/docs/hr/leave.md"] == b"leave text"
    assert all(row.status == "new" and row.version == 1 for row in registry.rows.values())


def test_second_run_is_a_noop(setup):
    _, service, store, registry = setup
    service.run()
    before = dict(store.objects)
    r = service.run()
    assert (r.new, r.changed, r.unchanged) == (0, 0, 2)
    assert store.objects == before


def test_edit_is_detected_by_content_hash(setup):
    root, service, store, registry = setup
    service.run()
    (root / "support" / "billing-refunds.md").write_text("refund text, updated")
    r = service.run()
    assert (r.new, r.changed, r.unchanged) == (0, 1, 1)
    row = next(x for x in registry.rows.values() if x.source_path.endswith("billing-refunds.md"))
    assert row.version == 2 and row.status == "changed"
    assert store.objects["raw/docs/support/billing-refunds.md"] == b"refund text, updated"


def test_metadata_is_stable_across_edits(setup):
    root, service, store, registry = setup
    service.run()
    snapshot = {k: (v.department, v.access_level, v.product) for k, v in registry.rows.items()}
    (root / "hr" / "leave.md").write_text("completely different")
    service.run()
    after = {k: (v.department, v.access_level, v.product) for k, v in registry.rows.items()}
    assert snapshot == after


def test_hr_document_is_restricted(setup):
    _, service, _, registry = setup
    service.run()
    hr = next(x for x in registry.rows.values() if x.source_path == "hr/leave.md")
    assert hr.department == "hr" and hr.access_level == "restricted"


def test_one_failing_document_does_not_stop_the_run(setup, monkeypatch):
    _, service, _, registry = setup
    original = service.connector.fetch

    def flaky(path):
        if path.endswith("leave.md"):
            raise OSError("disk exploded")
        return original(path)

    monkeypatch.setattr(service.connector, "fetch", flaky)
    r = service.run()
    assert (r.new, r.failed) == (1, 1)
    assert "leave.md" in r.failures[0] and "disk exploded" in r.failures[0]


def test_limit(setup):
    _, service, store, _ = setup
    assert service.run(limit=1).new == 1
    assert len(store.objects) == 1


def test_run_is_logged(setup):
    _, service, _, registry = setup
    service.run()
    assert registry.runs[0][0] == "docs" and registry.runs[0][1].new == 2


def test_bundled_sample_docs_give_each_role_something(tmp_path):
    """Guards the demo: the front-line agent role must be able to see public how-tos."""
    from pathlib import Path

    sample = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    connector = LocalFolderConnector("opendesk-sample-docs", sample, default_department="support")
    registry = FakeRegistry()
    LandingService(connector, FakeStore(), registry, TAXONOMY).run()

    rows = {r.source_path: r for r in registry.rows.values()}
    assert rows["support/public/password-reset.md"].access_level == "public"
    assert rows["support/public/billing-refunds.md"].access_level == "public"
    assert rows["product/internal/sla-tiers.md"].access_level == "internal"
    assert rows["hr/restricted/leave-policy.md"].access_level == "restricted"
