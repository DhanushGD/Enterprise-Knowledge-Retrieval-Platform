"""Step 4: copy documents from a source into the raw zone and record them.

Idempotent by design: running it twice in a row changes nothing the second time,
because a document is only re-landed when its SHA-256 content hash differs.
The store and registry are injected, so tests use in-memory fakes and production
uses MinIO/S3 + Postgres.
"""
from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import islice

from connectors.base import SourceConnector
from ingestion.metadata import assign_metadata
from ingestion.records import DocRecord


@dataclass
class LandingResult:
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    failed: int = 0
    deleted: int = 0                                   # filled in by SyncService
    failures: list[str] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)        # doc_ids the source listed this run
    complete: bool = False                             # True only if the whole listing was read


def raw_key(source_name: str, path: str) -> str:
    return f"raw/{source_name}/{path}"


class LandingService:
    def __init__(self, connector: SourceConnector, store, registry, taxonomy: dict):
        self.connector = connector
        self.store = store
        self.registry = registry
        self.taxonomy = taxonomy

    def run(self, limit: int | None = None, log: bool = True) -> LandingResult:
        started = datetime.now(timezone.utc)
        result = LandingResult()
        self.store.ensure()

        docs = self.connector.list_documents()
        if limit:
            docs = islice(docs, limit)

        for doc in docs:
            result.seen.add(doc.doc_id)
            try:
                self._land_one(doc, result)
            except Exception as exc:  # one bad file must not stop the whole run
                result.failed += 1
                result.failures.append(f"{doc.path}: {type(exc).__name__}: {exc}")

        result.complete = limit is None
        if log:
            self.registry.log_run(
                self.connector.name, result, started, datetime.now(timezone.utc)
            )
        return result

    def _land_one(self, doc, result: LandingResult) -> None:
        data = self.connector.fetch(doc.path)
        content_hash = hashlib.sha256(data).hexdigest()
        existing = self.registry.get(doc.doc_id)

        # A document that was deleted and has come back must be re-landed even if its
        # content is byte-identical to the version we removed.
        if existing and existing.content_hash == content_hash and existing.status != "deleted":
            result.unchanged += 1
            return

        key = raw_key(doc.source_name, doc.path)
        self.store.put(key, data, mimetypes.guess_type(doc.path)[0])

        if existing is None:
            meta = assign_metadata(
                doc.source_name, doc.path, doc.default_department, self.taxonomy
            )
            version, status = 1, "new"
            result.new += 1
        else:
            # Metadata stays stable across edits; only content-derived fields move.
            meta = {
                "department": existing.department,
                "access_level": existing.access_level,
                "product": existing.product,
            }
            version, status = existing.version + 1, "changed"
            result.changed += 1

        self.registry.upsert(DocRecord(
            doc_id=doc.doc_id,
            source_name=doc.source_name,
            source_path=doc.path,
            s3_key=key,
            content_hash=content_hash,
            size_bytes=len(data),
            etag=doc.etag,
            last_modified=doc.last_modified,
            status=status,
            version=version,
            **meta,
        ))
