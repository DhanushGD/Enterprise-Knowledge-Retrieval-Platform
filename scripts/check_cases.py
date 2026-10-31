"""For each eval question: which indexed pages match its label, and who can see them?

Use this to fix labels before trusting retrieval numbers.
Run:  python scripts/check_cases.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.db import SqlRegistry
from core.settings import company
from eval.retrieval_eval import load_cases

docs = SqlRegistry().list_by_status(["indexed"])
cases = load_cases(ROOT / "eval" / "retrieval_cases.yaml")
roles = company["roles"]


def can_see(doc, role):
    cfg = roles[role]
    return doc.access_level in cfg["allowed_access_levels"] and doc.department in cfg["allowed_departments"]


for case in cases:
    wanted = [e.lower() for e in case.expected]
    matches = [d for d in docs if any(w in d.source_path.lower() for w in wanted)]
    print(f"\n{case.question}\n  label: {case.expected}")
    if not matches:
        print("  !! NO INDEXED PAGE MATCHES - wrong label, or not in the 200-document sample")
    for d in matches[:4]:
        who = ",".join(r for r in roles if can_see(d, r)) or "nobody"
        print(f"  {d.source_path}  [{d.access_level}/{d.department}]  visible to: {who}")
