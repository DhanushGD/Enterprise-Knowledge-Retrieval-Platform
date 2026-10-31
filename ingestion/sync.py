"""Step 8: one incremental sync = land -> detect deletions -> parse -> chunk -> index.

Everything is driven by content hashes, so a sync that finds nothing new does almost no
work, and one that finds a single edited page re-embeds only that page.

Deletion is the dangerous part of any sync job: if the source is briefly unreachable and
the listing comes back empty, a naive "delete whatever I didn't see" wipes the index. So:

  * a failed or partial listing (error, or --limit) NEVER deletes anything;
  * an empty listing, or one that would remove more than `max_delete_fraction` of a source
    (with a floor of `delete_floor` documents), is refused unless explicitly forced;
  * deletes are soft in Postgres (row kept with status "deleted" as an audit trail) and
    hard in Qdrant / chunks / object storage, so removed content stops being retrievable.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ingestion.landing import LandingResult, LandingService
from ingestion.pipeline import ProcessingService, StageResult, parsed_key

log = logging.getLogger("rag.sync")


@dataclass
class SourceReport:
    source: str
    landing: LandingResult | None = None
    deleted: int = 0
    deletions_blocked: str | None = None
    delete_failures: list[str] = field(default_factory=list)
    error: str | None = None            # the source could not be listed at all


@dataclass
class SyncReport:
    sources: list[SourceReport]
    parse: StageResult
    chunk: StageResult
    index: StageResult
    seconds: float

    @property
    def has_problems(self) -> bool:
        source_problem = any(
            s.error or s.deletions_blocked or s.delete_failures
            or (s.landing and s.landing.failed)
            for s in self.sources
        )
        return bool(source_problem or self.parse.failed or self.chunk.failed or self.index.failed)


class SyncService:
    def __init__(
        self,
        connectors,
        store,
        registry,
        index,
        indexing,
        taxonomy: dict,
        chunking: dict | None = None,
        max_delete_fraction: float = 0.3,
        delete_floor: int = 5,
    ):
        self.connectors = connectors
        self.store = store
        self.registry = registry
        self.index = index
        self.indexing = indexing
        self.taxonomy = taxonomy
        self.chunking = chunking
        self.max_delete_fraction = max_delete_fraction
        self.delete_floor = delete_floor

    def run(self, limit: int | None = None, allow_mass_delete: bool = False) -> SyncReport:
        began = time.monotonic()
        reports = [self._sync_source(c, limit, allow_mass_delete) for c in self.connectors]

        processing = ProcessingService(self.store, self.registry, self.chunking)
        parse = processing.parse_pending()
        chunk = processing.chunk_pending()
        indexed = self.indexing.index_pending()
        return SyncReport(reports, parse, chunk, indexed, round(time.monotonic() - began, 1))

    def land(self, source_name: str, limit: int | None = None,
             allow_mass_delete: bool = False) -> SourceReport:
        """Land one source and apply its deletions, without parsing/chunking/indexing.
        This is the first stage of `run()` for a single source; an orchestrator (Airflow) calls
        it once per source and then runs the shared stages separately."""
        for connector in self.connectors:
            if connector.name == source_name:
                return self._sync_source(connector, limit, allow_mass_delete)
        raise ValueError(f"No source named '{source_name}'")

    # ------------------------------------------------------------------ per source
    def _sync_source(self, connector, limit, allow_mass_delete) -> SourceReport:
        report = SourceReport(connector.name)
        started = datetime.now(timezone.utc)
        try:
            result = LandingService(connector, self.store, self.registry, self.taxonomy).run(
                limit, log=False
            )
        except Exception as exc:
            # Could not even list the source (clone failed, bucket unreachable...).
            # We know nothing about what exists, so we must not delete anything.
            report.error = f"{type(exc).__name__}: {exc}"
            log.error("source %s could not be listed: %s", connector.name, report.error)
            return report

        report.landing = result
        self._apply_deletions(connector.name, result, limit, allow_mass_delete, report)
        result.deleted = report.deleted
        self.registry.log_run(connector.name, result, started, datetime.now(timezone.utc))
        log.info(
            "source %s: new=%d changed=%d unchanged=%d failed=%d deleted=%d",
            connector.name, result.new, result.changed, result.unchanged,
            result.failed, report.deleted,
        )
        return report

    def _apply_deletions(self, name, result: LandingResult, limit, allow_mass_delete,
                         report: SourceReport) -> None:
        if limit is not None or not result.complete:
            return                                       # partial listing: can't know what vanished

        known = [r for r in self.registry.list_by_source(name) if r.status != "deleted"]
        gone = [r for r in known if r.doc_id not in result.seen]
        if not gone:
            return

        allowed = max(self.delete_floor, int(self.max_delete_fraction * len(known)))
        if not allow_mass_delete and (not result.seen or len(gone) > allowed):
            report.deletions_blocked = (
                f"refused to delete {len(gone)} of {len(known)} documents "
                f"(safe limit {allowed}, listing returned {len(result.seen)}); "
                f"re-run with --force-deletes if this is intended"
            )
            log.warning("source %s: %s", name, report.deletions_blocked)
            return

        for rec in gone:
            try:
                self.index.delete_document(rec.doc_id)          # stop retrieval first
                self.registry.mark_deleted(rec.doc_id)
                self.store.delete([rec.s3_key, parsed_key(rec.source_name, rec.source_path)])
                report.deleted += 1
            except Exception as exc:                            # retried on the next sync
                report.delete_failures.append(f"{rec.source_path}: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------- helpers

def run_with_retries(fn, attempts: int = 3, base_delay: float = 30.0,
                     sleep=time.sleep, no_retry: tuple = ()):
    """Call fn(); on an unexpected exception wait base_delay * 4**n and try again.
    Exceptions listed in `no_retry` propagate immediately."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except no_retry:
            raise
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = base_delay * 4 ** (attempt - 1)
            log.warning("sync attempt %d/%d failed (%s: %s); retrying in %.0fs",
                        attempt, attempts, type(exc).__name__, exc, delay)
            sleep(delay)


def format_report(report: SyncReport) -> str:
    lines = []
    for s in report.sources:
        if s.error:
            lines.append(f"[{s.source}] COULD NOT LIST SOURCE: {s.error.splitlines()[0]}")
            continue
        r = s.landing
        lines.append(
            f"[{s.source}] new={r.new} changed={r.changed} unchanged={r.unchanged} "
            f"failed={r.failed} deleted={s.deleted}"
        )
        for failure in r.failures[:5]:
            lines.append(f"    ! {failure}")
        if s.deletions_blocked:
            lines.append(f"    ! {s.deletions_blocked}")
        for failure in s.delete_failures[:5]:
            lines.append(f"    ! delete failed: {failure}")
    for label, stage in (("parse", report.parse), ("chunk", report.chunk), ("index", report.index)):
        lines.append(f"{label:6} ok={stage.ok} skipped={stage.skipped} failed={stage.failed}")
        for failure in stage.failures[:5]:
            lines.append(f"    ! {failure}")
    lines.append(f"finished in {report.seconds}s - " + ("PROBLEMS, see above" if report.has_problems else "all clean"))
    return "\n".join(lines)
