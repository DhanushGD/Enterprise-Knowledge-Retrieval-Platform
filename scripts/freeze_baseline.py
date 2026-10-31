"""Freeze a finished full run as the baseline the release gate compares against.

Run:  python scripts/freeze_baseline.py                          # newest report in eval/reports/
      python scripts/freeze_baseline.py eval/reports/20261031-082942.json
Writes eval/baseline.json (commit it). No new evaluation is run.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.answer_eval import baseline_from_report

if len(sys.argv) > 1:
    source = Path(sys.argv[1])
else:
    reports = sorted((ROOT / "eval" / "reports").glob("*.json"))
    if not reports:
        sys.exit("No reports in eval/reports/. Run scripts/run_eval.py first.")
    source = reports[-1]

try:
    baseline = baseline_from_report(json.loads(source.read_text(encoding="utf-8")))
except ValueError as exc:
    sys.exit(f"Cannot freeze {source.name}: {exc}")

target = ROOT / "eval" / "baseline.json"
target.write_text(json.dumps(baseline, indent=1), encoding="utf-8")
m = baseline["metrics"]
print(f"baseline written from {source.name} -> {target.relative_to(ROOT)}")
print(f"  answered {m['answer_rate']:.0%}, cited right page {m['cited_expected_rate']:.0%}, "
      f"facts {m['fact_recall']:.0%}, grounded {m['grounded']:.0%}, leaks {m['leaks']}")
