"""Steps 3+4: pull documents from the configured source into the raw zone.

Run:  python scripts/run_landing.py              # everything in company.yaml
      python scripts/run_landing.py --limit 20   # quick trial
      python scripts/run_landing.py --source gitlab-docs-sample
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectors.factory import build_connectors
from core.db import SqlRegistry, init_db
from core.settings import ROOT, company, settings
from core.storage import RawStore
from ingestion.landing import LandingService

parser = argparse.ArgumentParser()
parser.add_argument("--source", help="only run the connector with this name")
parser.add_argument("--limit", type=int, help="stop after N documents per source")
args = parser.parse_args()

init_db()
store, registry = RawStore(), SqlRegistry()
clone_root = ROOT / settings.resolved_clone_dir   # absolute paths win over ROOT

connectors = build_connectors(company["source"], ROOT, clone_root)
if args.source:
    connectors = [c for c in connectors if c.name == args.source]
    if not connectors:
        sys.exit(f"No source named '{args.source}' in config/company.yaml")

exit_code = 0
for connector in connectors:
    print(f"\n== {connector.name} ==")
    try:
        result = LandingService(connector, store, registry, company["metadata"]).run(args.limit)
    except Exception as exc:  # e.g. git clone failed, bucket unreachable
        print(f"  FAILED to run: {type(exc).__name__}: {exc}")
        exit_code = 1
        continue
    print(
        f"  new={result.new}  changed={result.changed}  "
        f"unchanged={result.unchanged}  failed={result.failed}"
    )
    for failure in result.failures[:10]:
        print(f"  ! {failure}")
    if result.failed:
        exit_code = 1

print("\nDocuments by status:", registry.status_counts())
sys.exit(exit_code)
