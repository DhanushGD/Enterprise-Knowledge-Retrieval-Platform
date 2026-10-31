"""Steps 5+6 as pipeline stages that move documents through their status ladder:

    new / changed  --parse-->  parsed  --chunk-->  chunked        (indexed in Step 7)
                      |  \\--> parse_failed           \\--> chunk_failed
                      \\--> skipped   (valid file with no body text)

Each stage only touches documents waiting for it, so re-running is cheap and a
failure on one file never blocks the others. Failed documents are retried next run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ingestion.chunking import chunk_document
from ingestion.parsing import EmptyDocumentError, ParsedDocument, parse_document

PARSE_QUEUE = ["new", "changed", "parse_failed"]
CHUNK_QUEUE = ["parsed", "chunk_failed"]


def parsed_key(source_name: str, path: str) -> str:
    return f"parsed/{source_name}/{path}.json"


@dataclass
class StageResult:
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)


class ProcessingService:
    def __init__(self, store, registry, chunking: dict | None = None):
        self.store = store
        self.registry = registry
        self.chunking = {
            "max_tokens": 400, "overlap_tokens": 50, "min_tokens": 60, **(chunking or {})
        }

    def parse_pending(self, limit: int | None = None) -> StageResult:
        result = StageResult()
        for rec in self.registry.list_by_status(PARSE_QUEUE)[:limit]:
            try:
                parsed = parse_document(self.store.get(rec.s3_key), rec.source_path)
                self.store.put(
                    parsed_key(rec.source_name, rec.source_path),
                    parsed.to_json().encode("utf-8"),
                    "application/json",
                )
                self.registry.set_status(rec.doc_id, "parsed")
                result.ok += 1
            except EmptyDocumentError:
                # Nothing to index. Re-queued automatically if the file later gains content,
                # because landing flips it back to "changed".
                self.registry.set_status(rec.doc_id, "skipped")
                result.skipped += 1
            except Exception as exc:
                self.registry.set_status(rec.doc_id, "parse_failed")
                result.failed += 1
                result.failures.append(f"{rec.source_path}: {type(exc).__name__}: {exc}")
        return result

    def chunk_pending(self, limit: int | None = None) -> StageResult:
        result = StageResult()
        for rec in self.registry.list_by_status(CHUNK_QUEUE)[:limit]:
            try:
                raw = self.store.get(parsed_key(rec.source_name, rec.source_path))
                parsed = ParsedDocument.from_json(raw.decode("utf-8"))
                chunks = chunk_document(parsed, **self.chunking)
                if not chunks:
                    raise ValueError("document produced no chunks")
                self.registry.replace_chunks(rec, parsed.title, chunks)
                self.registry.set_status(rec.doc_id, "chunked")
                result.ok += 1
            except Exception as exc:
                self.registry.set_status(rec.doc_id, "chunk_failed")
                result.failed += 1
                result.failures.append(f"{rec.source_path}: {type(exc).__name__}: {exc}")
        return result
