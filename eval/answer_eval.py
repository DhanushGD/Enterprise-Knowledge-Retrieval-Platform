"""Step 12: end-to-end answer evaluation and the release gate.

Retrieval eval (eval/retrieval_eval.py) asks "did the right page come back?". This asks the
questions a user cares about, on the whole pipeline (retrieve -> generate -> validate):

  * Did it answer when it should, and refuse when it should?        answer_rate, oos_refusal_rate
  * Did it cite the page that actually holds the answer?            cited_expected_rate
  * Does the answer contain the facts a correct answer needs?       fact_recall
  * Is each sentence supported by the sources it cites?             grounded (lexical proxy)
  * Did it ever show a page the role may not see, or leak its prompt?   leaks, prompt_leaks

Everything here is deterministic and free: no LLM is used to *judge*. A 3B model on a CPU is
too unreliable as a judge (it agrees with fluent nonsense), so these are checks that cannot be
sweet-talked. `grounded` is a word-overlap proxy for faithfulness, not a semantic judgement; it
catches invented sentences, not subtle distortions. Treat it as a tripwire.

A golden case lists, per question: which page answers it (`expected_docs`, substrings of
source_path) and short phrases a correct answer must contain (`facts`; "a|b" = either).
The release gate compares the metrics with fixed thresholds and with a saved baseline.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from eval.retrieval_eval import ANSWERABLE, NO_DOC, Case, classify
from serving.generation import REFUSAL

ANSWERABLE_KIND, OUT_OF_SCOPE, INJECTION = "answerable", "out_of_scope", "injection"
KINDS = (ANSWERABLE_KIND, OUT_OF_SCOPE, INJECTION)
DEFAULT_ANSWERABLE_ROLES = ("agent", "admin")      # one that is restricted, one that sees everything


# ------------------------------------------------------------------ golden set

@dataclass
class GoldenCase:
    id: str
    question: str
    kind: str = ANSWERABLE_KIND
    roles: list[str] = field(default_factory=list)
    expected_docs: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    forbidden_docs: list[str] = field(default_factory=list)
    note: str = ""


def load_golden(path: str | Path, out_of_scope_path: str | Path | None = None) -> list[GoldenCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cases, seen = [], set()
    for i, item in enumerate(raw.get("cases", []), start=1):
        kind = item.get("kind", ANSWERABLE_KIND)
        if kind not in KINDS:
            raise ValueError(f"case {item.get('id', i)}: kind must be one of {KINDS}, got {kind!r}")
        expected = item.get("expected_docs", [])
        expected = [expected] if isinstance(expected, str) else list(expected)
        if kind == ANSWERABLE_KIND and not expected:
            raise ValueError(f"case {item.get('id', i)}: answerable cases need expected_docs")
        roles = item.get("roles") or (list(DEFAULT_ANSWERABLE_ROLES) if kind == ANSWERABLE_KIND
                                      else ["agent"])
        case = GoldenCase(
            id=str(item.get("id") or f"case-{i}"), question=item["question"], kind=kind,
            roles=list(roles), expected_docs=expected, facts=list(item.get("facts", [])),
            forbidden_docs=list(item.get("forbidden_docs", [])), note=item.get("note", ""),
        )
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)

    if out_of_scope_path:
        extra = yaml.safe_load(Path(out_of_scope_path).read_text(encoding="utf-8")) or {}
        known = {c.question.strip().lower() for c in cases}
        for n, question in enumerate(extra.get("questions", []), start=1):
            if question.strip().lower() not in known:
                cases.append(GoldenCase(id=f"oos-{n}", question=question, kind=OUT_OF_SCOPE,
                                        roles=["agent"]))
    return cases


# ------------------------------------------------------------------ text helpers

_STOP = set("""a an the and or but if then else of to in on at by for from with without about as is are
was were be been being this that these those it its into over under than so such not no yes can could
should would will may might do does did done have has had you your we our they their them he she his
her i me my there here what which who whom when where why how also just only more most any all each
other some use used using via per etc""".split())
_CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_WORD = re.compile(r"[a-z0-9][a-z0-9_\-]{2,}")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for part in parts:
        part = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", part).strip()
        if len(content_words(_CITE.sub("", part))) >= 3:
            out.append(part)
    return out


def grounding(answer_text: str, sources) -> tuple[float | None, int, int]:
    """(mean support, unsupported sentences, sentences). A sentence's support is the share of
    its content words found in the sources it cites (all sources if it cites none)."""
    by_n = {s.n: content_words(s.context or s.snippet) for s in sources}
    everything = set().union(*by_n.values()) if by_n else set()
    scores = []
    for sentence in split_sentences(answer_text):
        cited = {int(n) for m in _CITE.findall(sentence) for n in re.findall(r"\d+", m)}
        pool = set().union(*(by_n[n] for n in cited if n in by_n)) if cited & set(by_n) else everything
        words = content_words(_CITE.sub("", sentence))
        scores.append(len(words & pool) / len(words))
    if not scores:
        return None, 0, 0
    return sum(scores) / len(scores), sum(1 for s in scores if s < 0.5), len(scores)


def fact_recall(text: str, facts: list[str]) -> float | None:
    if not facts:
        return None
    lowered = text.lower()
    hit = sum(any(alt.strip().lower() in lowered for alt in fact.split("|")) for fact in facts)
    return hit / len(facts)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _longest_shared_run(a: list[str], b: list[str]) -> int:
    positions: dict[str, list[int]] = {}
    for j, word in enumerate(b):
        positions.setdefault(word, []).append(j)
    best = 0
    for i in range(len(a)):
        for j in positions.get(a[i], ()):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best


LEAK_RUN = 7        # this many consecutive words in common with the prompt counts as repeating it


def leaks_prompt(text: str, system_prompt: str, allowed: tuple[str, ...] = (REFUSAL,)) -> bool:
    """True when the answer repeats a stretch of the system prompt. The fixed refusal sentence
    appears in the prompt on purpose and is allowed."""
    for phrase in allowed:
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
        system_prompt = re.sub(re.escape(phrase), " ", system_prompt, flags=re.IGNORECASE)
    return _longest_shared_run(_words(text), _words(system_prompt)) >= LEAK_RUN


# ------------------------------------------------------------------ one case

@dataclass
class CaseResult:
    case: GoldenCase
    role: str
    visibility: str = "n/a"             # answerable | hidden | no_doc (answerable-kind cases only)
    status: str = ""
    reason: str | None = None
    text: str = ""
    retrieved_hit: bool | None = None   # expected page was among the sources shown to the model
    cited_hit: bool | None = None       # ... and among the pages the answer cites
    fact_recall: float | None = None
    grounded: float | None = None
    unsupported: int = 0
    sentences: int = 0
    leaked: list[str] = field(default_factory=list)
    prompt_leak: bool = False
    error: str | None = None
    total_ms: float = 0.0
    behaved: bool = True                # did the pipeline do the right thing for this case?
    problems: list[str] = field(default_factory=list)


class Visibility:
    """Which paths a role may see, from the document registry (the ground truth for ACLs)."""

    def __init__(self, docs, roles: dict):
        self.docs, self.roles = list(docs), roles
        self.by_path: dict[str, list] = {}
        for d in self.docs:
            self.by_path.setdefault(d.source_path, []).append(d)

    def allowed(self, path: str, role: str) -> bool:
        cfg = self.roles[role]
        docs = self.by_path.get(path)
        if not docs:
            return False                                   # unknown to the registry: fail closed
        return all(d.access_level in cfg["allowed_access_levels"]
                   and d.department in cfg["allowed_departments"] for d in docs)


def _matches(path: str, wanted: list[str]) -> bool:
    return any(w.lower() in path.lower() for w in wanted)


def evaluate_case(service, case: GoldenCase, role: str, visibility: Visibility,
                  system_prompt: str, llm_used: bool = True) -> CaseResult:
    res = CaseResult(case=case, role=role)
    if case.kind == ANSWERABLE_KIND:
        res.visibility = classify(Case(case.question, case.expected_docs), visibility.docs,
                                  visibility.roles[role])
        if res.visibility == NO_DOC:
            res.behaved = False
            res.problems.append("label matches no indexed page - fix expected_docs")
            return res

    began = time.perf_counter()
    try:
        answer = service.answer(case.question, role)
    except Exception as exc:                               # one broken case must not stop the run
        res.error = f"{type(exc).__name__}: {exc}"
        res.behaved = False
        res.problems.append(res.error)
        return res
    res.total_ms = answer.timings.get("total_ms") or round((time.perf_counter() - began) * 1000, 1)
    res.status, res.reason, res.text = answer.status, answer.reason, answer.text

    shown = [s.source_path for s in answer.sources]
    cited = [s.source_path for s in answer.citations]
    res.leaked = sorted({p for p in shown if not visibility.allowed(p, role)}
                        | {p for p in shown if _matches(p, case.forbidden_docs)})
    res.prompt_leak = leaks_prompt(answer.text, system_prompt)
    if res.leaked:
        res.problems.append(f"showed pages this role must not see: {res.leaked}")
    if res.prompt_leak:
        res.problems.append("the answer repeats the system prompt")

    if case.kind == ANSWERABLE_KIND and res.visibility == ANSWERABLE:
        res.retrieved_hit = any(_matches(p, case.expected_docs) for p in shown)
        res.cited_hit = any(_matches(p, case.expected_docs) for p in cited)
        if not res.retrieved_hit:
            res.problems.append(f"expected page not retrieved (got {shown[:3]})")
        if llm_used:
            if answer.status != "answered":
                res.problems.append(f"expected an answer, got {answer.status}"
                                    + (f" ({answer.reason})" if answer.reason else ""))
            else:
                res.fact_recall = fact_recall(answer.text, case.facts)
                res.grounded, res.unsupported, res.sentences = grounding(answer.text, answer.sources)
                if not res.cited_hit:
                    res.problems.append(f"cited the wrong page(s): {cited}")
                if res.fact_recall is not None and res.fact_recall < 1:
                    res.problems.append(f"missing facts (recall {res.fact_recall:.0%})")
            res.behaved = answer.status == "answered" and bool(res.retrieved_hit) and not res.leaked
        else:
            res.behaved = bool(res.retrieved_hit) and not res.leaked
    elif case.kind == ANSWERABLE_KIND:                     # HIDDEN from this role
        res.behaved = not res.leaked
    elif case.kind == OUT_OF_SCOPE:
        res.behaved = answer.status == "abstained" and not res.leaked
        if answer.status != "abstained":
            res.problems.append(f"should have refused, got {answer.status}")
    else:                                                  # INJECTION
        res.behaved = not res.leaked and not res.prompt_leak
    return res


# ------------------------------------------------------------------ aggregates

def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _rate(flags):
    flags = list(flags)
    return sum(flags) / len(flags) if flags else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


def summarize(results: list[CaseResult], llm_used: bool = True) -> dict:
    ok = [r for r in results if not r.error]
    answerable = [r for r in ok if r.case.kind == ANSWERABLE_KIND and r.visibility == ANSWERABLE]
    oos = [r for r in ok if r.case.kind == OUT_OF_SCOPE]
    answered_ans = [r for r in answerable if r.status == "answered"]
    statuses: dict[str, int] = {}
    for r in ok:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    sentences = sum(r.sentences for r in answered_ans)
    latency = [r.total_ms for r in ok if r.status in ("answered", "ungrounded")]

    m = {
        "cases_run": len(results),
        "answerable_run": len(answerable),
        "retrieved_hit_rate": _rate(r.retrieved_hit for r in answerable),
        "oos_refusal_rate": _rate(r.status == "abstained" for r in oos),
        "leaks": sum(len(r.leaked) for r in results),
        "prompt_leaks": sum(r.prompt_leak for r in results),
        "errors": sum(1 for r in results if r.error),
        "bad_labels": sum(1 for r in results if r.visibility == NO_DOC),
        "answer_rate": None, "cited_expected_rate": None, "fact_recall": None,
        "grounded": None, "unsupported_sentence_rate": None, "ungrounded_rate": None,
    }
    if llm_used:
        m.update({
            "answer_rate": _rate(r.status == "answered" for r in answerable),
            "cited_expected_rate": _rate(bool(r.cited_hit) for r in answerable),
            "fact_recall": _mean(r.fact_recall for r in answered_ans),
            "grounded": _mean(r.grounded for r in answered_ans),
            "unsupported_sentence_rate": (sum(r.unsupported for r in answered_ans) / sentences
                                          if sentences else None),
            "ungrounded_rate": _rate(r.status == "ungrounded" for r in ok),
        })
    m["latency_p50_ms"] = percentile(latency, 0.5)
    m["latency_p95_ms"] = percentile(latency, 0.95)
    m["status_counts"] = statuses
    return m


# ------------------------------------------------------------------ the gate

def gate(metrics: dict, thresholds: dict, baseline: dict | None = None) -> list[str]:
    """Human-readable violations; an empty list means the release may proceed.

    thresholds: {"min": {metric: floor}, "max": {metric: ceiling}, "regression_tolerance": 0.05}
    A metric that was not measured (None, e.g. answer metrics in --no-llm mode) is skipped.
    Against a baseline, a "min" metric may not fall, and a "max" metric may not rise, by more
    than the tolerance (absolute, in the metric's own units).
    """
    problems = []
    floors, ceilings = thresholds.get("min", {}), thresholds.get("max", {})
    tol = float(thresholds.get("regression_tolerance", 0.05))

    for name, floor in floors.items():
        value = metrics.get(name)
        if value is not None and value < floor:
            problems.append(f"{name} = {value:.3f} is below the minimum {floor}")
    for name, ceiling in ceilings.items():
        value = metrics.get(name)
        if value is not None and value > ceiling:
            problems.append(f"{name} = {value:.3f} is above the maximum {ceiling}")

    if baseline:
        base = baseline.get("metrics", baseline)
        for name in floors:
            now, before = metrics.get(name), base.get(name)
            if now is not None and before is not None and now < before - tol:
                problems.append(f"{name} regressed: {before:.3f} -> {now:.3f} (tolerance {tol})")
        for name in ceilings:
            now, before = metrics.get(name), base.get(name)
            if now is not None and before is not None and now > before + tol:
                problems.append(f"{name} regressed: {before:.3f} -> {now:.3f} (tolerance {tol})")
    return problems


def baseline_from_report(report: dict, column: str | None = None) -> dict:
    """Turn a saved run report (eval/reports/*.json) into the baseline the gate compares against."""
    if not report.get("llm_used"):
        raise ValueError("this report was made with --no-llm; a baseline needs a full run")
    columns = report["metrics"]
    column = column or next(iter(columns))
    if column not in columns:
        raise ValueError(f"no column {column!r} in the report (have {list(columns)})")
    return {"when": report.get("when"), "model": report.get("model"), "metrics": columns[column]}


# ------------------------------------------------------------------ reporting

_ROWS = [
    ("retrieved_hit_rate", "right page retrieved", "pct"),
    ("answer_rate", "answered when it should", "pct"),
    ("cited_expected_rate", "cited the right page", "pct"),
    ("fact_recall", "required facts present", "pct"),
    ("grounded", "grounded (word-overlap proxy)", "pct"),
    ("unsupported_sentence_rate", "unsupported sentences", "pct"),
    ("ungrounded_rate", "'ungrounded' fallbacks", "pct"),
    ("oos_refusal_rate", "off-topic refused", "pct"),
    ("leaks", "PERMISSION LEAKS", "int"),
    ("prompt_leaks", "PROMPT LEAKS", "int"),
    ("errors", "errors", "int"),
    ("bad_labels", "bad golden labels", "int"),
    ("latency_p50_ms", "latency p50 (ms)", "ms"),
    ("latency_p95_ms", "latency p95 (ms)", "ms"),
]


def _fmt(value, kind):
    if value is None:
        return "-"
    return {"pct": f"{100 * value:.0f}%", "int": f"{int(value)}", "ms": f"{value:,.0f}"}[kind]


def format_metrics(columns: dict[str, dict]) -> str:
    """One column per configuration (e.g. retrieval mode)."""
    names = list(columns)
    width = max(12, *(len(n) for n in names))
    lines = [f"{'':34}" + "".join(f"{n:>{width + 2}}" for n in names)]
    for key, label, kind in _ROWS:
        lines.append(f"{label:34}" + "".join(f"{_fmt(columns[n].get(key), kind):>{width + 2}}"
                                             for n in names))
    first = columns[names[0]]
    lines.append(f"\n{first['cases_run']} runs ({first['answerable_run']} answerable); "
                 f"statuses: {first['status_counts']}")
    return "\n".join(lines)


def format_failures(results: list[CaseResult], limit: int = 40) -> str:
    bad = [r for r in results if r.problems]
    if not bad:
        return "No problems."
    lines = [f"{len(bad)} runs with problems:"]
    for r in bad[:limit]:
        lines.append(f"\n[{r.case.id}] as {r.role}: {r.case.question}")
        lines += [f"    - {p}" for p in r.problems]
    if len(bad) > limit:
        lines.append(f"\n... and {len(bad) - limit} more (see the JSON report)")
    return "\n".join(lines)
