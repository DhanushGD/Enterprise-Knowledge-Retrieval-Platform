"""Exercises the real git code path (sparse + partial clone) against a local repo."""
import shutil
import subprocess

import pytest

from connectors.git_repo import GitRepoConnector

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo, check=True, capture_output=True,
    )


def write(repo, rel, text):
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode())      # bytes: Windows text mode would turn \n into \r\n


@pytest.fixture
def upstream(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "uploadpack.allowFilter", "true")          # allow partial clone
    git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")   # allow lazy blob fetch
    write(repo, "docs/guide/install.md", "# Install\n\nrun it")
    write(repo, "docs/guide/deep/faq.md", "# FAQ\n\nanswers")
    write(repo, "other/unrelated.md", "# Nope")
    write(repo, "README.md", "# root")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def make(upstream, tmp_path, subpath="docs/guide", **kw):
    return GitRepoConnector(
        name="up", url=upstream.as_uri(), clone_root=tmp_path / "clones", subpath=subpath, **kw
    )


def test_clones_only_the_requested_subpath(upstream, tmp_path):
    c = make(upstream, tmp_path)
    paths = sorted(d.path for d in c.list_documents())
    assert paths == ["deep/faq.md", "install.md"]
    assert c.fetch("install.md") == b"# Install\n\nrun it"
    assert not (c.clone_dir / "other" / "unrelated.md").exists()    # sparse checkout worked


def test_sync_picks_up_edits_additions_and_deletions(upstream, tmp_path):
    c = make(upstream, tmp_path)
    list(c.list_documents())

    write(upstream, "docs/guide/install.md", "# Install\n\nrun it, updated")
    write(upstream, "docs/guide/new.md", "# New\n\nfresh")
    git(upstream, "rm", "-q", "docs/guide/deep/faq.md")
    git(upstream, "add", "-A")
    git(upstream, "commit", "-q", "-m", "update")

    fresh = make(upstream, tmp_path)                 # same clone dir, new process
    paths = sorted(d.path for d in fresh.list_documents())
    assert paths == ["install.md", "new.md"]
    assert fresh.fetch("install.md").endswith(b"updated")


def test_missing_subpath_error_lists_what_exists(upstream, tmp_path):
    c = make(upstream, tmp_path, subpath="docs/guides")          # typo: guides vs guide
    with pytest.raises(RuntimeError) as err:
        list(c.list_documents())
    message = str(err.value)
    assert "does not exist" in message and "docs/guide" in message
    assert "company.yaml" in message


def test_wrong_url_fails_fast_instead_of_prompting(tmp_path):
    c = GitRepoConnector(
        name="gone", url=(tmp_path / "no-such-repo").as_uri(), clone_root=tmp_path / "clones",
        subpath="x",
    )
    with pytest.raises(RuntimeError, match="git clone"):
        list(c.list_documents())


@pytest.fixture
def windows_git(tmp_path, monkeypatch):
    """Make git behave like a default Windows install (core.autocrlf=true)."""
    cfg = tmp_path / "gitconfig-windows"
    cfg.write_text("[core]\n\tautocrlf = true\n")
    from connectors import git_repo
    monkeypatch.setattr(git_repo, "_GIT_ENV", {**git_repo._GIT_ENV, "GIT_CONFIG_GLOBAL": str(cfg)})
    return cfg


def test_files_are_byte_identical_even_when_git_wants_crlf(upstream, tmp_path, windows_git):
    c = make(upstream, tmp_path)
    list(c.list_documents())
    assert c.fetch("install.md") == b"# Install\n\nrun it"        # no CRLF injected


def test_old_crlf_clone_is_rebuilt_byte_exact(upstream, tmp_path, windows_git):
    from connectors import git_repo
    clone = tmp_path / "clones" / "up"
    clone.parent.mkdir(parents=True)
    # A clone made the way the previous version did it (no -c core.autocrlf=false):
    git_repo._git("clone", "--depth", "1", "--filter=blob:none", "--sparse",
                  upstream.as_uri(), str(clone))
    git_repo._git("sparse-checkout", "set", "docs/guide", cwd=clone)
    assert b"\r\n" in (clone / "docs/guide/install.md").read_bytes()    # the problem, reproduced

    c = make(upstream, tmp_path)
    list(c.list_documents())
    assert c.fetch("install.md") == b"# Install\n\nrun it"
