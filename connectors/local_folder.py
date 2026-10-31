"""Reads documents from a folder on disk. Also the base for the git connector."""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from connectors.base import SourceConnector, SourceDocument

SUPPORTED_EXTENSIONS = {".md", ".mdx", ".txt", ".html", ".htm", ".pdf", ".docx"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".github"}


class LocalFolderConnector(SourceConnector):
    def __init__(
        self,
        name: str,
        root: str | Path,
        subpath: str = "",
        default_department: str | None = None,
        max_documents: int | None = None,
        extensions: Iterable[str] | None = None,
    ):
        self.name = name
        self.default_department = default_department
        self.max_documents = max_documents
        self.extensions = {e.lower() for e in (extensions or SUPPORTED_EXTENSIONS)}
        self.base = (Path(root) / subpath).resolve()

    def _prepare(self) -> None:
        """Hook for subclasses that must fetch files before listing (git)."""

    def _walk(self) -> Iterator[SourceDocument]:
        for dirpath, dirnames, filenames in os.walk(self.base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for filename in sorted(filenames):
                full = Path(dirpath) / filename
                if full.suffix.lower() not in self.extensions:
                    continue
                stat = full.stat()
                yield SourceDocument(
                    source_name=self.name,
                    path=full.relative_to(self.base).as_posix(),
                    size=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    default_department=self.default_department,
                )

    def list_documents(self) -> Iterator[SourceDocument]:
        self._prepare()
        if not self.base.is_dir():
            raise FileNotFoundError(f"Source folder not found: {self.base}")
        docs = list(self._walk())
        if self.max_documents and len(docs) > self.max_documents:
            # Keep the N documents with the smallest path hash. Unlike "first N", this is
            # spread across every folder, and it is stable: adding or removing one file
            # barely changes which others are selected (matters for incremental sync).
            ranked = sorted(
                docs, key=lambda d: hashlib.sha1(f"{self.name}:{d.path}".encode()).hexdigest()
            )
            docs = sorted(ranked[: self.max_documents], key=lambda d: d.path)
        yield from docs

    def fetch(self, path: str) -> bytes:
        target = (self.base / path).resolve()
        if not target.is_relative_to(self.base):
            raise ValueError(f"Path escapes source root: {path}")
        return target.read_bytes()
