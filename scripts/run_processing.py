"""Steps 5+6: parse landed documents, then chunk them.

Run:  python scripts/run_processing.py
      python scripts/run_processing.py --limit 10
      python scripts/run_processing.py --rechunk   # re-split every document with the current
                                                   # chunking settings (then run run_indexing.py)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import SqlRegistry, init_db
from core.settings import company
from core.storage import RawStore
from ingestion.pipeline import ProcessingService

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, help="process at most N documents per stage")
parser.add_argument("--rechunk", action="store_true",
                    help="re-chunk all documents (use after changing chunking settings)")
args = parser.parse_args()

init_db()
registry = SqlRegistry()
if args.rechunk:
    moved = sum(registry.reset_status(s, "parsed")
                for s in ("chunked", "indexed", "chunk_failed", "index_failed"))
    print(f"Queued {moved} documents for re-chunking")
service = ProcessingService(RawStore(), registry, company.get("chunking"))

exit_code = 0
for stage, run in (("parse", service.parse_pending), ("chunk", service.chunk_pending)):
    result = run(args.limit)
    print(f"{stage:6} ok={result.ok}  skipped={result.skipped}  failed={result.failed}")
    for failure in result.failures[:10]:
        print(f"  ! {failure}")
    if result.failed:
        exit_code = 1

print("\nDocuments by status:", registry.status_counts())
print("Chunk stats:", registry.chunk_stats())
sys.exit(exit_code)
