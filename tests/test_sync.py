import pytest

from connectors.local_folder import LocalFolderConnector
from ingestion.indexer import IndexingService
from ingestion.pipeline import parsed_key
from ingestion.sync import SyncService, format_report, run_with_retries
from tests.fakes import FakeEmbedder, FakeIndex, FakeRegistry, FakeStore

TAXONOMY = {
    "department": ["support", "product", "engineering", "hr"],
    "access_level": ["public", "internal", "restricted"],
    "product": ["core", "billing", "integrations", "security"],
}


def md(title, extra=""):
    body = " ".join(f"{title} sentence {i} explains the process in some detail." for i in range(12))
    return f"# {title}\n\n{body} {extra}".encode()


class Env:
    def __init__(self, tmp_path, files, **kw):
        self.root = tmp_path
        self.store, self.registry, self.index = FakeStore(), FakeRegistry(), FakeIndex()
        for name, content in files.items():
            self.write(name, content)
        self.connector = LocalFolderConnector("docs", tmp_path, default_department="support")
        self.sync = SyncService(
            [self.connector], self.store, self.registry, self.index,
            IndexingService(self.registry, FakeEmbedder(), self.index), TAXONOMY, **kw,
        )

    def write(self, name, content):
        (self.root / name).write_bytes(content)

    def remove(self, name):
        (self.root / name).unlink()

    def statuses(self):
        return {r.source_path: r.status for r in self.registry.rows.values()}

    def points_for(self, name):
        doc_id = next(r.doc_id for r in self.registry.rows.values() if r.source_path == name)
        return [p for p in self.index.points.values() if p.payload["doc_id"] == doc_id]


def test_first_sync_lands_processes_and_indexes_everything(tmp_path):
    env = Env(tmp_path, {"a.md": md("Alpha"), "b.md": md("Beta")})
    report = env.sync.run()
    assert env.statuses() == {"a.md": "indexed", "b.md": "indexed"}
    assert (report.sources[0].landing.new, report.parse.ok, report.chunk.ok, report.index.ok) == (2, 2, 2, 2)
    assert not report.has_problems
    assert env.index.count() >= 2
    assert len(env.registry.runs) == 1 and env.registry.runs[0][1].new == 2


def test_second_sync_changes_nothing(tmp_path):
    env = Env(tmp_path, {"a.md": md("Alpha")})
    env.sync.run()
    before = dict(env.index.points)
    report = env.sync.run()
    assert report.sources[0].landing.unchanged == 1
    assert (report.parse.ok, report.chunk.ok, report.index.ok) == (0, 0, 0)
    assert env.index.points == before


def test_edit_is_reindexed_and_old_text_disappears(tmp_path):
    env = Env(tmp_path, {"a.md": md("Alpha")})
    env.sync.run()
    env.write("a.md", md("Alpha", extra="NEWFACT about refunds."))
    report = env.sync.run()
    assert report.sources[0].landing.changed == 1 and report.index.ok == 1
    assert any("NEWFACT" in p.payload["text"] for p in env.points_for("a.md"))
    assert next(iter(env.registry.rows.values())).version == 2


def test_deleted_file_is_removed_everywhere(tmp_path):
    env = Env(tmp_path, {"a.md": md("Alpha"), "b.md": md("Beta")})
    env.sync.run()
    b = next(r for r in env.registry.rows.values() if r.source_path == "b.md")

    env.remove("b.md")
    report = env.sync.run()

    assert report.sources[0].deleted == 1 and not report.has_problems
    assert env.statuses() == {"a.md": "indexed", "b.md": "deleted"}       # soft-deleted row kept
    assert not [p for p in env.index.points.values() if p.payload["doc_id"] == b.doc_id]
    assert b.doc_id not in env.registry.chunks
    assert b.s3_key not in env.store.objects
    assert parsed_key("docs", "b.md") not in env.store.objects
    assert env.registry.runs[-1][1].deleted == 1


def test_file_that_returns_is_reindexed_even_if_identical(tmp_path):
    env = Env(tmp_path, {"a.md": md("Alpha"), "b.md": md("Beta")})
    env.sync.run()
    env.remove("b.md"); env.sync.run()
    assert env.statuses()["b.md"] == "deleted"

    env.write("b.md", md("Beta"))                                          # byte-identical to before
    env.sync.run()
    assert env.statuses()["b.md"] == "indexed" and env.points_for("b.md")


def test_partial_run_with_limit_never_deletes(tmp_path):
    env = Env(tmp_path, {"a.md": md("Alpha"), "b.md": md("Beta"), "c.md": md("Gamma")})
    env.sync.run()
    env.remove("c.md")
    report = env.sync.run(limit=1)
    assert report.sources[0].deleted == 0 and report.sources[0].deletions_blocked is None
    assert env.statuses()["c.md"] == "indexed"


def test_mass_deletion_is_refused_unless_forced(tmp_path):
    env = Env(tmp_path, {f"d{i}.md": md(f"Doc{i}") for i in range(10)})
    env.sync.run()
    for i in range(8):
        env.remove(f"d{i}.md")

    report = env.sync.run()
    assert report.sources[0].deleted == 0
    assert "refused to delete 8 of 10" in report.sources[0].deletions_blocked
    assert report.has_problems
    assert sum(1 for s in env.statuses().values() if s == "indexed") == 10      # untouched

    forced = env.sync.run(allow_mass_delete=True)
    assert forced.sources[0].deleted == 8
    assert sum(1 for s in env.statuses().values() if s == "deleted") == 8


def test_empty_listing_is_refused_even_for_a_small_source(tmp_path):
    env = Env(tmp_path, {"a.md": md("Alpha"), "b.md": md("Beta")})
    env.sync.run()
    env.remove("a.md"); env.remove("b.md")              # e.g. a sync glitch returned nothing
    report = env.sync.run()
    assert report.sources[0].deleted == 0 and report.sources[0].deletions_blocked
    assert set(env.statuses().values()) == {"indexed"}


def test_small_deletions_under_the_floor_are_allowed(tmp_path):
    env = Env(tmp_path, {f"d{i}.md": md(f"Doc{i}") for i in range(6)})
    env.sync.run()
    env.remove("d0.md"); env.remove("d1.md")
    assert env.sync.run().sources[0].deleted == 2


def test_unlistable_source_is_isolated_and_deletes_nothing(tmp_path):
    healthy_dir = tmp_path / "healthy"
    healthy_dir.mkdir()
    env = Env(healthy_dir, {"a.md": md("Alpha")})
    broken = LocalFolderConnector("ghost", tmp_path / "does-not-exist")
    env.sync.connectors.append(broken)
    env.sync.run()                                       # first pass indexes the healthy source

    report = env.sync.run()
    ghost = next(s for s in report.sources if s.source == "ghost")
    healthy = next(s for s in report.sources if s.source == "docs")
    assert "FileNotFoundError" in ghost.error and ghost.landing is None
    assert healthy.landing.unchanged == 1 and report.has_problems
    assert env.statuses() == {"a.md": "indexed"}


def test_empty_pages_are_skipped_quietly_not_failed(tmp_path):
    env = Env(tmp_path, {"real.md": md("Real"), "stub.md": b"---\ntitle: Section\n---\n\n# Section\n"})
    report = env.sync.run()
    assert env.statuses() == {"real.md": "indexed", "stub.md": "skipped"}
    assert (report.parse.skipped, report.parse.failed) == (1, 0) and not report.has_problems
    assert env.sync.run().parse.skipped == 0             # not retried every run

    env.write("stub.md", md("Stub"))                      # gains content later -> picked up
    env.sync.run()
    assert env.statuses()["stub.md"] == "indexed"


def test_report_is_readable(tmp_path):
    env = Env(tmp_path, {"a.md": md("Alpha")})
    text = format_report(env.sync.run())
    assert "[docs] new=1" in text and "deleted=0" in text and "all clean" in text


# ------------------------------------------------------------------------ retries

def test_retries_back_off_then_succeed():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("db down")
        return "ok"

    assert run_with_retries(flaky, attempts=3, base_delay=30, sleep=waits.append) == "ok"
    assert waits == [30, 120]


def test_retries_give_up_after_the_last_attempt():
    waits = []
    with pytest.raises(ConnectionError):
        run_with_retries(lambda: (_ for _ in ()).throw(ConnectionError("down")),
                         attempts=3, base_delay=1, sleep=waits.append)
    assert len(waits) == 2


def test_no_retry_exceptions_propagate_immediately():
    class Busy(RuntimeError):
        pass

    waits = []
    with pytest.raises(Busy):
        run_with_retries(lambda: (_ for _ in ()).throw(Busy()), no_retry=(Busy,), sleep=waits.append)
    assert waits == []


def test_source_that_becomes_unreachable_keeps_all_its_documents(tmp_path):
    """The dangerous case: documents are indexed, then the source vanishes (clone fails,
    bucket unreachable, folder unmounted). Nothing may be deleted."""
    import shutil

    env = Env(tmp_path, {"a.md": md("Alpha"), "b.md": md("Beta")})
    env.sync.run()
    indexed_points = dict(env.index.points)

    shutil.rmtree(tmp_path)                                # source now raises on listing
    report = env.sync.run()

    assert "FileNotFoundError" in report.sources[0].error and report.has_problems
    assert report.sources[0].deleted == 0
    assert set(env.statuses().values()) == {"indexed"}
    assert env.index.points == indexed_points
    assert any(key.startswith("raw/docs/") for key in env.store.objects)
