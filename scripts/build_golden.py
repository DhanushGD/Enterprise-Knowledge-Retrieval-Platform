"""Draft golden-set candidates with the local model (then review them by hand).

Run:  python scripts/build_golden.py --count 30
Writes eval/golden_candidates.yaml. Open it, read each question against its page, delete the bad
ones, fix the rest, and paste the keepers into eval/golden.yaml. Takes ~15 s per candidate on a CPU.

Only pages an admin can see and that are public-or-internal are sampled; the generated case
inherits the default roles (agent + admin), so pages an agent may not see become permission tests.
"""
import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from core.db import SqlRegistry
from eval.golden_gen import PROMPT, parse_candidate, validate_candidate
from serving.llm import LLMError, build_llm

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=30, help="candidates to produce")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", default=str(ROOT / "eval" / "golden_candidates.yaml"))
args = parser.parse_args()

registry = SqlRegistry()
rng = random.Random(args.seed)
docs = registry.list_by_status(["indexed"])
rng.shuffle(docs)

llm = build_llm(num_ctx=2048)
try:
    llm.warm_up()
except LLMError as exc:
    sys.exit(f"LLM problem: {exc}")

known = {c["question"].lower() for c in (yaml.safe_load((ROOT / "eval" / "golden.yaml").read_text()) or {}).get("cases", [])}
out, tried = [], 0
for doc in docs:
    if len(out) >= args.count:
        break
    chunks = [c for c in registry.get_chunks(doc.doc_id) if 90 <= c.token_count <= 300]
    if not chunks:
        continue
    chunk = rng.choice(chunks)
    tried += 1
    raw = "".join(t for t in llm.stream(
        [{"role": "user", "content": PROMPT.format(passage=chunk.text[:1500])}], 160) if isinstance(t, str))
    cand = validate_candidate(parse_candidate(raw), chunk.text)
    status = "kept" if cand and cand["question"].lower() not in known else "dropped"
    print(f"[{tried}] {status:7} {doc.source_path}", flush=True)
    if status == "kept":
        slug = re.sub(r"[^a-z0-9]+", "-", Path(doc.source_path).stem.lower()).strip("-")[:40]
        out.append({"id": f"gen-{slug}", "question": cand["question"], "expected_docs": doc.source_path,
                    "facts": cand["facts"], "note": f"DRAFT - section: {chunk.section or '-'}"})

header = ("# DRAFTS written by a 3B model. Read each question against its page before trusting it.\n"
          "# Delete bad ones, fix the rest, then move them into eval/golden.yaml.\n")
Path(args.out).write_text(header + yaml.safe_dump({"cases": out}, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")
print(f"\n{len(out)} candidates from {tried} attempts -> {args.out}")
