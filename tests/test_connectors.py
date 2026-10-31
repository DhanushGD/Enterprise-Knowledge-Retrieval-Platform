from datetime import datetime, timedelta, timezone

import pytest

from connectors.base import SourceDocument
from connectors.local_folder import LocalFolderConnector


@pytest.fixture
def docs_dir(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.md").write_text("# One")
    (tmp_path / "a" / "image.png").write_bytes(b"\x89PNG")   # unsupported type
    (tmp_path / "two.txt").write_text("two")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.md").write_text("must be skipped")
    return tmp_path


def test_lists_only_supported_files_and_skips_git(docs_dir):
    c = LocalFolderConnector("src", docs_dir)
    paths = [d.path for d in c.list_documents()]
    assert sorted(paths) == ["a/one.md", "two.txt"]


def test_listing_is_deterministic(docs_dir):
    c = LocalFolderConnector("src", docs_dir)
    assert [d.path for d in c.list_documents()] == [d.path for d in c.list_documents()]


def test_max_documents(docs_dir):
    c = LocalFolderConnector("src", docs_dir, max_documents=1)
    assert len(list(c.list_documents())) == 1


def test_fetch_returns_bytes(docs_dir):
    c = LocalFolderConnector("src", docs_dir)
    assert c.fetch("a/one.md") == b"# One"


def test_fetch_blocks_path_traversal(docs_dir):
    c = LocalFolderConnector("src", docs_dir / "a")
    with pytest.raises(ValueError):
        c.fetch("../two.txt")


def test_missing_folder_raises(tmp_path):
    c = LocalFolderConnector("src", tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        list(c.list_documents())


def test_list_changes_filters_by_timestamp(docs_dir):
    c = LocalFolderConnector("src", docs_dir)
    future = datetime.now(timezone.utc) + timedelta(days=1)
    past = datetime.now(timezone.utc) - timedelta(days=1)
    assert list(c.list_changes(future)) == []
    assert len(list(c.list_changes(past))) == 2


def test_default_department_propagates(docs_dir):
    c = LocalFolderConnector("src", docs_dir, default_department="support")
    assert all(d.default_department == "support" for d in c.list_documents())


def test_doc_id_is_stable_and_source_scoped():
    now = datetime.now(timezone.utc)
    a = SourceDocument("s1", "x.md", 1, now)
    b = SourceDocument("s1", "x.md", 99, now)
    c = SourceDocument("s2", "x.md", 1, now)
    assert a.doc_id == b.doc_id        # size/time changes do not change identity
    assert a.doc_id != c.doc_id        # same path in another source is a different doc


def test_max_documents_samples_across_folders_not_just_the_first(tmp_path):
    for folder in "abcdefgh":
        (tmp_path / folder).mkdir()
        for i in range(10):
            (tmp_path / folder / f"doc{i}.md").write_text(f"# {folder}{i}")
    c = LocalFolderConnector("src", tmp_path, max_documents=24)
    picked = [d.path for d in c.list_documents()]
    folders = {p.split("/")[0] for p in picked}
    assert len(picked) == 24
    assert len(folders) >= 6                       # not just a/ and b/ (the alphabetical first 24)
    assert picked == sorted(picked)                # still returned in path order


def test_sampling_is_stable_when_a_file_is_added(tmp_path):
    for i in range(60):
        (tmp_path / f"doc{i:02d}.md").write_text("x")
    before = {d.path for d in LocalFolderConnector("s", tmp_path, max_documents=20).list_documents()}
    (tmp_path / "brand-new.md").write_text("x")
    after = {d.path for d in LocalFolderConnector("s", tmp_path, max_documents=20).list_documents()}
    assert len(before - after) <= 1                # at most one document displaced
