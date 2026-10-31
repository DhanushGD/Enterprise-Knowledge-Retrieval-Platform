"""Step 8: one full incremental sync - land, detect deletions, parse, chunk, index.

Run:  python scripts/run_sync.py
      python scripts/run_sync.py --source gitlab-handbook-sample
      python scripts/run_sync.py --limit 20        # quick trial; never deletes anything
      python scripts/run_sync.py --force-deletes   # allow removing a large share of a source

Safe to run repeatedly and from several terminals: a Postgres advisory lock stops two
syncs from overlapping. Exit code is 1 if anything needs attention.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import SyncAlreadyRunning
from ingestion.runner import run_sync_once
from ingestion.sync import format_report

parser = argparse.ArgumentParser()
parser.add_argument("--source", help="only sync the source with this name")
parser.add_argument("--limit", type=int, help="process at most N documents per source (disables deletes)")
parser.add_argument("--force-deletes", action="store_true", help="bypass the mass-deletion safety check")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

try:
    report = run_sync_once(args.limit, args.source, args.force_deletes)
except SyncAlreadyRunning:
    sys.exit("Another sync is already running; try again when it finishes.")

print("\n" + format_report(report))
sys.exit(1 if report.has_problems else 0)
