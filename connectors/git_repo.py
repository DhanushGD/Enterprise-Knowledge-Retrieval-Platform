"""Clones (shallow + sparse) a git repo and exposes one folder of it as a source.

Works for GitHub, GitLab or any git host. A sparse, blob-filtered clone downloads
only the folder you ask for, so even a huge repo stays small on disk.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath

from connectors.local_folder import LocalFolderConnector

# Never hang waiting for a username/password prompt (private or mistyped repo URLs
# make GitLab/GitHub ask for credentials; on Windows that can pop a GUI dialog).
_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False, env=_GIT_ENV
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def _rmtree(path: Path) -> None:
    """rmtree that also deletes git's read-only object files (needed on Windows)."""
    def _writable_then_retry(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)
    shutil.rmtree(path, onerror=_writable_then_retry)


class GitRepoConnector(LocalFolderConnector):
    def __init__(
        self,
        name: str,
        url: str,
        clone_root: str | Path,
        subpath: str = "",
        **kwargs,
    ):
        self.url = url
        self.repo_subpath = subpath.strip("/")
        self.clone_dir = (Path(clone_root) / name).resolve()
        self._synced = False
        super().__init__(name=name, root=self.clone_dir, subpath=self.repo_subpath, **kwargs)

    def sync(self) -> None:
        """Clone on first use, fast-forward to the remote's latest after that.

        Files deleted upstream disappear locally too (reset --hard), which is how the
        incremental sync in Step 8 notices deletions.
        """
        if (self.clone_dir / ".git").exists() and not self._byte_exact():
            # Cloned under the user's autocrlf=true (Windows default): files carry CRLF, so
            # their hashes differ from every other machine's. Rebuild once, byte-exact.
            _rmtree(self.clone_dir)

        if not (self.clone_dir / ".git").exists():
            self.clone_dir.parent.mkdir(parents=True, exist_ok=True)
            # autocrlf=false: check files out exactly as committed, so a document's SHA-256
            # is the same on Windows, Linux and a server. (-c on clone persists in the repo.)
            _git(
                "clone", "-c", "core.autocrlf=false", "--depth", "1", "--filter=blob:none",
                "--sparse", self.url, str(self.clone_dir),
            )
            if self.repo_subpath:
                _git("sparse-checkout", "set", self.repo_subpath, cwd=self.clone_dir)
        else:
            _git("fetch", "--depth", "1", "origin", "HEAD", cwd=self.clone_dir)
            _git("reset", "--hard", "FETCH_HEAD", cwd=self.clone_dir)
        self._synced = True

        if self.repo_subpath and not (self.clone_dir / self.repo_subpath).is_dir():
            raise RuntimeError(self._missing_subpath_message())

    def _byte_exact(self) -> bool:
        try:
            return _git("config", "--local", "--get", "core.autocrlf",
                        cwd=self.clone_dir).strip() == "false"
        except RuntimeError:                       # key not set in the repo
            return False

    def _prepare(self) -> None:
        if not self._synced:
            self.sync()

    def _missing_subpath_message(self) -> str:
        """Tell the user what *does* exist, so fixing config.yaml is one edit."""
        for ancestor in PurePosixPath(self.repo_subpath).parents:
            rel = "" if str(ancestor) == "." else ancestor.as_posix()
            args = ["ls-tree", "--name-only", "HEAD"] + ([f"{rel}/"] if rel else [])
            try:
                entries = _git(*args, cwd=self.clone_dir).splitlines()
            except RuntimeError:
                continue
            if entries:
                shown = "\n  ".join(entries[:30])
                where = f"'{rel}/'" if rel else "the repo root"
                return (
                    f"Folder '{self.repo_subpath}' does not exist in {self.url}.\n"
                    f"Closest existing location is {where}, which contains:\n  {shown}\n"
                    f"Fix `subpath` for source '{self.name}' in config/company.yaml."
                )
        return f"Folder '{self.repo_subpath}' does not exist in {self.url}."
