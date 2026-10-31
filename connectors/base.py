"""The contract every document source must satisfy.

Everything downstream (landing, parsing, chunking...) only knows about
`SourceDocument` and `SourceConnector`. To support SharePoint later, write a
`SharePointConnector` that implements these three methods and nothing else changes.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator


@dataclass(frozen=True)
class SourceDocument:
    source_name: str          # e.g. "gitlab-handbook-sample"
    path: str                 # posix path relative to the source root
    size: int
    last_modified: datetime   # timezone-aware (UTC)
    etag: str | None = None   # provider's version marker, if it has one
    default_department: str | None = None

    @property
    def doc_id(self) -> str:
        """Stable id: same source + path always gives the same id."""
        return hashlib.sha1(f"{self.source_name}:{self.path}".encode()).hexdigest()[:16]


class SourceConnector(ABC):
    name: str
    default_department: str | None = None

    @abstractmethod
    def list_documents(self) -> Iterator[SourceDocument]:
        """Yield every document currently in the source."""

    @abstractmethod
    def fetch(self, path: str) -> bytes:
        """Return the raw bytes of one document."""

    def list_changes(self, since: datetime) -> Iterator[SourceDocument]:
        """Documents modified after `since`.

        Default filters `list_documents()` by timestamp. Connectors with a native
        delta API (Microsoft Graph delta query, S3 events) should override this.
        Timestamps can lie (git checkouts reset them), so the pipeline always
        decides "changed or not" by content hash, never by this alone.
        """
        for doc in self.list_documents():
            if doc.last_modified > since:
                yield doc
