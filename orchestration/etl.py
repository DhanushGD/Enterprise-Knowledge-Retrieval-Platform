"""Step 8b: the sync pipeline as separately runnable, separately retryable stages.

    preflight -> sources -> land (one per source) -> parse -> chunk -> index -> verify

`python scripts/run_sync.py` runs all of this in one process. An orchestrator (Airflow, see
orchestration/dags/ekrp_sync.py) wants the same work as small steps it can retry, time out and
show individually. So each stage is a plain function here, and this module doubles as a CLI:

    python -m orchestration.etl land --source gitlab-handbook-sample
    python -m orchestration.etl all            # every stage in order, no Airflow needed

CLI contract (what the Airflow wrapper relies on):
  * logs go to stderr; the LAST line on stdout is one JSON object describing the outcome
  * exit 0  = success
  * exit 10 = failed, worth retrying (service down, source unreachable, another sync running)
  * exit 20 = failed, retrying cannot help (a human must act: blocked mass deletion, broken index)

Every stage that changes data takes the same Postgres advisory lock as `run_sync.py`, so an
Airflow run and a manual run can never write at the same time.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __name__ == "__main__":      # allow `python orchestration/etl.py` as well as `-m orchestration.etl`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.errors import SyncAlreadyRunning
from ingestion.pipeline import StageResult

log = logging.getLogger("rag.etl")

EXIT_OK, EXIT_RETRY, EXIT_FATAL = 0, 10, 20
STAGES = ("preflight", "sources", "land", "parse", "chunk", "index", "verify")
STUCK_STATUSES = ("parse_failed", "chunk_failed", "index_failed")
LIVE_STATUSES_EXCLUDED = ("deleted", "skipped")        # not expected to be in the index
DEFAULT_MAX_FAILED_FRACTION = 0.05


class EtlError(RuntimeError):
    """A stage failed in a way that a retry may fix."""


class EtlFatal(EtlError):
    """A stage failed and retrying will not help; a person needs to look."""


# ------------------------------------------------------------------------ helpers

def _failed_too_much(failed: int, total: int, max_fraction: float) -> bool:
    return total > 0 and failed / total > max_fraction


def _stage_summary(name: str, result: StageResult, max_fraction: float) -> dict:
    out = {"stage": name, "ok": result.ok, "skipped": result.skipped, "failed": result.failed,
           "failures": result.failures[:10]}
    handled = result.ok + result.skipped + result.failed
    if _failed_too_much(result.failed, handled, max_fraction):
        # Failed documents stay queued (*_failed) and are retried on the next attempt, so a
        # retry is useful when the cause was transient (model/DB hiccup).
        raise EtlError(
            f"{name}: {result.failed} of {handled} documents failed "
            f"(limit {max_fraction:.0%}); first: {result.failures[0] if result.failures else '?'}"
        )
    return out


# ------------------------------------------------------------------------ stages

def preflight(service) -> dict:
    """Fail early and clearly when Postgres, Qdrant or object storage is down, instead of
    letting a later stage report a confusing half-failure."""
    checks = {
        "postgres": service.registry.status_counts,
        "qdrant": service.index.healthy,
        "object_storage": service.store.ensure,
    }
    down = {}
    for name, check in checks.items():
        try:
            check()
        except Exception as exc:
            down[name] = f"{type(exc).__name__}: {exc}"
    if down:
        raise EtlError("infrastructure not reachable: " + "; ".join(f"{k} ({v})" for k, v in down.items()))
    return {"stage": "preflight", "services": sorted(checks)}


def sources(service) -> dict:
    return {"stage": "sources", "sources": [c.name for c in service.connectors]}


def land(service, source: str, limit: int | None = None, force_deletes: bool = False,
         max_fraction: float = DEFAULT_MAX_FAILED_FRACTION) -> dict:
    with service.registry.sync_lock():
        report = service.land(source, limit=limit, allow_mass_delete=force_deletes)

    if report.error:
        # Could not even list the source. Nothing was deleted (see SyncService); retry later.
        raise EtlError(f"source '{source}' could not be listed: {report.error.splitlines()[0]}")
    if report.deletions_blocked:
        raise EtlFatal(f"source '{source}': {report.deletions_blocked}. Trigger the DAG with "
                       f"force_deletes=true if this deletion is really intended.")
    if report.delete_failures:
        raise EtlError(f"source '{source}': {len(report.delete_failures)} deletions failed, "
                       f"first: {report.delete_failures[0]}")

    r = report.landing
    handled = r.new + r.changed + r.unchanged + r.failed
    if _failed_too_much(r.failed, handled, max_fraction):
        raise EtlError(f"source '{source}': {r.failed} of {handled} documents failed to land, "
                       f"first: {r.failures[0] if r.failures else '?'}")
    return {"stage": "land", "source": source, "new": r.new, "changed": r.changed,
            "unchanged": r.unchanged, "failed": r.failed, "deleted": report.deleted,
            "failures": list(r.failures[:10])}


def _processing(service):
    from ingestion.pipeline import ProcessingService

    return ProcessingService(service.store, service.registry, service.chunking)


def parse(service, max_fraction: float = DEFAULT_MAX_FAILED_FRACTION) -> dict:
    with service.registry.sync_lock():
        return _stage_summary("parse", _processing(service).parse_pending(), max_fraction)


def chunk(service, max_fraction: float = DEFAULT_MAX_FAILED_FRACTION) -> dict:
    with service.registry.sync_lock():
        return _stage_summary("chunk", _processing(service).chunk_pending(), max_fraction)


def index(service, max_fraction: float = DEFAULT_MAX_FAILED_FRACTION) -> dict:
    with service.registry.sync_lock():
        return _stage_summary("index", service.indexing.index_pending(), max_fraction)


def verify(service, max_fraction: float = DEFAULT_MAX_FAILED_FRACTION) -> dict:
    """Last gate: is what is searchable consistent with what Postgres says is indexed?"""
    counts = service.registry.status_counts()
    stuck = {s: counts[s] for s in STUCK_STATUSES if counts.get(s)}
    live = sum(n for status, n in counts.items() if status not in LIVE_STATUSES_EXCLUDED)
    indexed = counts.get("indexed", 0)
    out = {"stage": "verify", "documents": counts, "vectors": None,
           "corpus_version": service.registry.corpus_version()}

    if live == 0:
        raise EtlFatal("the registry holds no live documents; refusing to call an empty index healthy")
    if indexed == 0:
        raise EtlFatal(f"no document is indexed (statuses: {counts})")

    stuck_total = sum(stuck.values())
    if _failed_too_much(stuck_total, live, max_fraction):
        raise EtlFatal(f"{stuck_total} of {live} documents are stuck in a failed state {stuck}; "
                       f"limit {max_fraction:.0%}")

    vectors = service.index.count()
    expected = service.registry.indexed_chunk_count()
    out["vectors"], out["expected_vectors"] = vectors, expected
    if not stuck_total and vectors != expected:
        # With nothing failing, Qdrant must hold exactly the chunks of the indexed documents.
        raise EtlFatal(f"vector index has {vectors} points but the registry expects {expected} "
                       f"(a partial write or a deleted document that is still searchable?)")
    if stuck_total:
        out["warning"] = f"{stuck_total} documents are stuck and will be retried: {stuck}"
        log.warning(out["warning"])
    return out


# ------------------------------------------------------------------------ CLI

def _build_service():
    from ingestion.runner import build_sync_service

    service, _registry = build_sync_service()
    return service


def _max_fraction() -> float:
    from core.settings import company

    return float((company.get("etl") or {}).get("max_failed_fraction", DEFAULT_MAX_FAILED_FRACTION))


def run_stage(stage: str, service, *, source: str | None = None, limit: int | None = None,
              force_deletes: bool = False, max_fraction: float = DEFAULT_MAX_FAILED_FRACTION) -> dict:
    if stage == "preflight":
        return preflight(service)
    if stage == "sources":
        return sources(service)
    if stage == "land":
        if not source:
            raise ValueError("land needs --source")
        return land(service, source, limit, force_deletes, max_fraction)
    if stage == "parse":
        return parse(service, max_fraction)
    if stage == "chunk":
        return chunk(service, max_fraction)
    if stage == "index":
        return index(service, max_fraction)
    if stage == "verify":
        return verify(service, max_fraction)
    raise ValueError(f"unknown stage '{stage}'")


def run_all(service, *, limit: int | None = None, force_deletes: bool = False,
            max_fraction: float = DEFAULT_MAX_FAILED_FRACTION) -> dict:
    """Every stage in order, in one process: the Airflow DAG without Airflow."""
    steps = [preflight(service)]
    for name in sources(service)["sources"]:
        steps.append(land(service, name, limit, force_deletes, max_fraction))
    for stage in (parse, chunk, index):
        steps.append(stage(service, max_fraction))
    steps.append(verify(service, max_fraction))
    return {"stage": "all", "steps": steps}


def classify(exc: BaseException) -> int:
    if isinstance(exc, EtlFatal):
        return EXIT_FATAL
    if isinstance(exc, ValueError):          # bad arguments / unknown source: retrying cannot fix it
        return EXIT_FATAL
    return EXIT_RETRY                         # EtlError, SyncAlreadyRunning, connection errors, ...


def main(argv: list[str] | None = None, service_factory=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m orchestration.etl")
    parser.add_argument("stage", choices=STAGES + ("all",))
    parser.add_argument("--source", help="source name (stage 'land')")
    parser.add_argument("--limit", type=int, default=0,
                        help="land at most N documents per source; disables deletions (0 = no limit)")
    parser.add_argument("--force-deletes", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        service = (service_factory or _build_service)()
        common = dict(limit=args.limit or None, force_deletes=args.force_deletes,
                      max_fraction=_max_fraction())
        result = (run_all(service, **common) if args.stage == "all"
                  else run_stage(args.stage, service, source=args.source, **common))
    except SyncAlreadyRunning as exc:
        return _finish({"ok": False, "stage": args.stage, "error": f"SyncAlreadyRunning: {exc}"},
                       EXIT_RETRY)
    except Exception as exc:
        log.exception("stage %s failed", args.stage)
        return _finish({"ok": False, "stage": args.stage, "error": f"{type(exc).__name__}: {exc}"},
                       classify(exc))
    return _finish({"ok": True, **result}, EXIT_OK)


def _finish(payload: dict, code: int) -> int:
    print(json.dumps(payload, default=str))         # must stay the last stdout line
    return code


if __name__ == "__main__":
    sys.exit(main())
