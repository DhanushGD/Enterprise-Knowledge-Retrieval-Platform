"""Step 12: evaluate the whole pipeline on the golden set, and (optionally) gate a release.

Run:  python scripts/run_eval.py --no-llm              # retrieval + refusal gate + permissions; ~1 min
      python scripts/run_eval.py                       # full: real answers from the local model (~15-20 min on CPU)
      python scripts/run_eval.py --modes dense hybrid  # compare retrieval strategies end to end
      python scripts/run_eval.py --gate                # exit 1 if a threshold or the baseline is violated
      python scripts/run_eval.py --write-baseline      # freeze today's numbers as the reference
      python scripts/run_eval.py --limit 5 --show-failures

--no-llm swaps the model for one that is "down": answerable questions end as llm_unavailable
(with the retrieved pages), off-topic ones still end as abstained (the relevance gate), so
retrieval, the gate and permissions are all measured without waiting for generation.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from core.db import SqlRegistry
from core.settings import company
from eval.answer_eval import (Visibility, evaluate_case, format_failures, format_metrics, gate,
                              load_golden, summarize)
from serving.generation import SYSTEM_PROMPT, AnswerService
from serving.llm import LLMError, LLMUnavailable, build_llm
from serving.retrieval import MODES, build_retriever

EVAL = ROOT / "eval"
parser = argparse.ArgumentParser()
parser.add_argument("--golden", default=str(EVAL / "golden.yaml"))
parser.add_argument("--no-llm", action="store_true", help="skip generation (fast)")
parser.add_argument("--modes", nargs="+", choices=MODES,
                    help="retrieval strategies to compare (default: the configured one)")
parser.add_argument("--limit", type=int, help="run only the first N cases")
parser.add_argument("--only", help="run only cases whose id contains this text")
parser.add_argument("--gate", action="store_true", help="exit 1 if the release gate fails")
parser.add_argument("--write-baseline", action="store_true")
parser.add_argument("--show-failures", action="store_true")
parser.add_argument("--thresholds", default=str(EVAL / "thresholds.yaml"))
args = parser.parse_args()


class DownLLM:
    def stream(self, messages, max_tokens):
        raise LLMUnavailable("evaluation run without a model (--no-llm)")
        yield                                                  # pragma: no cover


cases = load_golden(args.golden, EVAL / "out_of_scope.yaml")
if args.only:
    cases = [c for c in cases if args.only in c.id]
if args.limit:
    cases = cases[: args.limit]
if not cases:
    sys.exit("No cases selected.")

registry = SqlRegistry()
visibility = Visibility(registry.list_by_status(["indexed"]), company["roles"])
if not visibility.docs:
    sys.exit("No indexed documents. Run scripts/run_sync.py first.")

modes = args.modes or [None]
retrieval_cfg = company.get("retrieval", {}) or {}
llm_used = not args.no_llm
gen = company.get("generation", {}) or {}
retriever = build_retriever(rerank=("hybrid_rerank" in modes) or retrieval_cfg.get("rerank", False))
llm = DownLLM() if args.no_llm else build_llm(num_ctx=gen.get("num_ctx", 2048))
service = AnswerService(
    retriever, llm, context_chunks=gen.get("context_chunks", 3),
    max_chunk_chars=gen.get("max_chunk_chars", 800), max_answer_tokens=gen.get("max_answer_tokens", 220),
    min_dense_score=gen.get("min_dense_score", 0.5), require_citations=gen.get("require_citations", True),
)
if llm_used:
    print("Warming up the model ...", flush=True)
    try:
        llm.warm_up()
    except LLMError as exc:
        sys.exit(f"LLM problem: {exc}\n(use --no-llm to evaluate without generation)")

runs = sum(len(c.roles) for c in cases)
print(f"{len(cases)} cases, {runs} runs per configuration, "
      f"{'with the local model' if llm_used else 'retrieval only'}\n", flush=True)

all_results, columns = {}, {}
for mode in modes:
    label = mode or "configured"
    service.retrieval_mode = mode
    results, began, done = [], time.time(), 0
    for case in cases:
        for role in case.roles:
            done += 1
            r = evaluate_case(service, case, role, visibility, SYSTEM_PROMPT, llm_used)
            results.append(r)
            flag = "ok " if r.behaved and not r.problems else "!! "
            print(f"[{label}] {done:>3}/{runs} {flag}{case.id} ({role}) {r.status or r.error} "
                  f"{r.total_ms:,.0f} ms", flush=True)
    all_results[label] = results
    columns[label] = summarize(results, llm_used)
    print(f"  {label}: {time.time() - began:,.0f}s\n")

print(format_metrics(columns))
primary = next(iter(columns))
if args.show_failures or args.gate:
    print("\n" + format_failures(all_results[primary]))

reports = EVAL / "reports"
reports.mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
report = {"when": stamp, "llm_used": llm_used, "model": getattr(llm, "model", None),
          "config": {"retrieval": retrieval_cfg, "generation": gen},
          "metrics": columns,
          "results": {k: [asdict(r) for r in v] for k, v in all_results.items()}}
(reports / f"{stamp}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
print(f"\nreport: eval/reports/{stamp}.json")

baseline_path = EVAL / "baseline.json"
metrics = columns[primary]
if args.write_baseline:
    if not llm_used:
        sys.exit("A baseline needs the full run (without --no-llm), or the answer metrics are empty.")
    baseline_path.write_text(json.dumps({"when": stamp, "model": getattr(llm, "model", None),
                                         "metrics": metrics}, indent=1), encoding="utf-8")
    print(f"baseline written: {baseline_path.relative_to(ROOT)}  (commit it)")

if args.gate:
    thresholds = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    if baseline and baseline.get("metrics", {}).get("answer_rate") is not None and not llm_used:
        baseline = {"metrics": {k: v for k, v in baseline["metrics"].items()
                                if k in ("retrieved_hit_rate", "oos_refusal_rate", "leaks",
                                         "prompt_leaks", "errors", "bad_labels")}}
    problems = gate(metrics, thresholds, baseline)
    print("\nRELEASE GATE: " + ("FAILED" if problems else "passed"
                                + ("" if baseline else "  (no baseline yet - run --write-baseline)")))
    for p in problems:
        print(f"  - {p}")
    sys.exit(1 if problems else 0)
