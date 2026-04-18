"""
evaluation/Router/eval_router.py
=================================
Evaluates the production routing path: decompose_with_zeroshot().

Two evaluation modes (run together by default):

  1. Classification accuracy (test_cases.json, 60 queries)
     - Calls decompose_with_zeroshot() on each query
     - Derives overall route via route_from_subtasks()
     - Reports per-class Precision / Recall / F1 + confusion matrix

  2. Decomposition quality (decomp_cases.json, 10 complex queries)
     - Route validity  : all sub-task routes in {sql, text}
     - Atomicity       : each sub-task is single-intent
     - Coverage        : key concepts from original appear in sub-tasks
     - No raw SQL      : sub_query fields are natural language only

Usage:
  python evaluation/Router/eval_router.py
  python evaluation/Router/eval_router.py --only-classification
  python evaluation/Router/eval_router.py --only-decomp
  python evaluation/Router/eval_router.py --output path/to/out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROUTER_DIR       = Path(__file__).resolve().parent
TEST_CASES_FILE  = ROUTER_DIR / "test_cases.json"
DECOMP_CASES_FILE = ROUTER_DIR / "decomp_cases.json"
DEFAULT_OUTPUT   = PROJECT_ROOT / "evaluation" / "results" / "router_eval.json"

CLASSES: list[str] = ["sql", "text", "hybrid"]

_RAW_SQL_RE = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|FROM|WHERE|GROUP\s+BY|ORDER\s+BY)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClassResult:
    case_id: str
    query: str
    expected: str
    predicted: str
    correct: bool
    difficulty: str
    subtasks: list[dict]
    error: str | None = None


@dataclass
class PerClassMetrics:
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "label":     self.label,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "support":   self.support,
            "precision": round(self.precision, 4),
            "recall":    round(self.recall,    4),
            "f1":        round(self.f1,        4),
        }


@dataclass
class DecompResult:
    case_id: str
    query: str
    subtasks: list[dict]
    route_validity: float
    atomicity: float
    coverage: float
    no_raw_sql: float
    count_ok: bool
    quality_score: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cohen_kappa(confusion: dict, classes: list[str]) -> float:
    """Multi-class Cohen's kappa (κ)."""
    n = sum(confusion[e][p] for e in classes for p in classes)
    if n == 0:
        return 0.0
    p_o = sum(confusion[c][c] for c in classes) / n
    p_e = sum(
        (sum(confusion[k][p] for p in classes) / n) *
        (sum(confusion[e][k] for e in classes) / n)
        for k in classes
    )
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-12 else 1.0


def _mcc_multiclass(confusion: dict, classes: list[str]) -> float:
    """Gorodkin multi-class Matthews Correlation Coefficient."""
    n = sum(confusion[e][p] for e in classes for p in classes)
    if n == 0:
        return 0.0
    c   = sum(confusion[k][k] for k in classes)
    t_k = [sum(confusion[k][p] for p in classes) for k in classes]   # row sums (actual)
    p_k = [sum(confusion[e][k] for e in classes) for k in classes]   # col sums (predicted)
    num      = c * n - sum(t * p for t, p in zip(t_k, p_k))
    denom_sq = (n * n - sum(p ** 2 for p in p_k)) * (n * n - sum(t ** 2 for t in t_k))
    return num / denom_sq ** 0.5 if denom_sq > 0 else 0.0


def _compute_metrics(results: list[ClassResult]) -> dict:
    pc = {c: PerClassMetrics(label=c) for c in CLASSES}
    confusion = {e: {p: 0 for p in CLASSES} for e in CLASSES}

    for r in results:
        if r.error:
            continue
        exp  = r.expected
        pred = r.predicted if r.predicted in CLASSES else "text"

        if exp in pc:
            if pred == exp:
                pc[exp].tp += 1
            else:
                pc[exp].fn += 1
                if pred in pc:
                    pc[pred].fp += 1

        if exp in confusion:
            confusion[exp][pred] = confusion[exp].get(pred, 0) + 1

    valid = [r for r in results if not r.error]
    n = len(valid)
    accuracy    = sum(r.correct for r in valid) / n if n else 0.0
    macro_p     = sum(m.precision for m in pc.values()) / len(CLASSES)
    macro_r     = sum(m.recall    for m in pc.values()) / len(CLASSES)
    macro_f1    = sum(m.f1        for m in pc.values()) / len(CLASSES)
    weighted_f1 = (sum(m.f1 * m.support for m in pc.values()) / n) if n else 0.0
    kappa       = _cohen_kappa(confusion, CLASSES)
    mcc         = _mcc_multiclass(confusion, CLASSES)

    by_diff: dict[str, dict] = {}
    for diff in ("easy", "medium", "hard"):
        sub = [r for r in valid if r.difficulty == diff]
        if sub:
            by_diff[diff] = {
                "n": len(sub),
                "correct": sum(r.correct for r in sub),
                "accuracy": round(sum(r.correct for r in sub) / len(sub), 4),
            }

    return {
        "n_cases":         n,
        "n_errors":        sum(1 for r in results if r.error),
        "accuracy":        round(accuracy,    4),
        "macro_precision": round(macro_p,     4),
        "macro_recall":    round(macro_r,     4),
        "macro_f1":        round(macro_f1,    4),
        "weighted_f1":     round(weighted_f1, 4),
        "cohen_kappa":     round(kappa,       4),
        "mcc":             round(mcc,         4),
        "per_class":        {c: pc[c].as_dict() for c in CLASSES},
        "confusion_matrix": confusion,
        "by_difficulty":   by_diff,
    }


def _atomicity(route: str, sub_query: str, sql_kws: set[str], text_kws: set[str]) -> float:
    if route not in {"sql", "text"}:
        return 0.0
    tokens    = set(re.findall(r"\b\w+\b", sub_query.lower()))
    sql_hits  = len(tokens & sql_kws)
    text_hits = len(tokens & text_kws)
    if sql_hits == 0 and text_hits == 0:
        return 0.7   # ambiguous but route was declared
    if route == "sql":
        return 1.0 if sql_hits >= text_hits else 0.4
    return 1.0 if text_hits >= sql_hits else 0.4


def _coverage(key_concepts: list[str], subtasks: list[dict]) -> float:
    if not key_concepts:
        return 1.0
    combined = " ".join(t.get("sub_query", "") for t in subtasks).lower()
    return sum(1 for c in key_concepts if c.lower() in combined) / len(key_concepts)


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def run_classification(cases: list[dict]) -> list[ClassResult]:
    from backend.router.router import QueryRouter
    router = QueryRouter()
    results = []

    for c in cases:
        qid   = c["id"]
        query = c["query"]
        exp   = c["expected_route"]
        diff  = c.get("difficulty", "unknown")

        try:
            subtasks  = router.decompose_with_zeroshot(query)
            predicted = router.route_from_subtasks(subtasks)
            task_dicts = [{"sub_query": t.sub_query, "route": t.route} for t in subtasks]
            results.append(ClassResult(
                case_id=qid, query=query, expected=exp,
                predicted=predicted, correct=(predicted == exp),
                difficulty=diff, subtasks=task_dicts,
            ))
        except Exception as exc:
            results.append(ClassResult(
                case_id=qid, query=query, expected=exp,
                predicted="error", correct=False,
                difficulty=diff, subtasks=[], error=str(exc),
            ))

        mark = "+" if results[-1].correct else ("-" if not results[-1].error else "!")
        sub_info = " | ".join(
            f"{t['route']}:{t['sub_query'][:30]}"
            for t in results[-1].subtasks
        ) or results[-1].error or ""
        print(f"  [{mark}] {qid:<12} exp={exp:<7} pred={results[-1].predicted:<7}  {sub_info[:70]}")

    return results


def print_classification_report(results: list[ClassResult], metrics: dict) -> None:
    m = metrics
    n_correct = sum(r.correct for r in results if not r.error)

    print(f"\n  {'='*65}")
    print(f"  CLASSIFICATION REPORT")
    print(f"  {'='*65}")
    print(f"  Accuracy          : {m['accuracy']:.4f}  ({n_correct}/{m['n_cases']})")
    if m["n_errors"]:
        print(f"  Errors skipped    : {m['n_errors']}")
    print(f"  Macro Precision   : {m['macro_precision']:.4f}")
    print(f"  Macro Recall      : {m['macro_recall']:.4f}")
    print(f"  Macro F1          : {m['macro_f1']:.4f}")
    print(f"  Weighted F1       : {m['weighted_f1']:.4f}")
    print(f"  Cohen's κ         : {m['cohen_kappa']:.4f}")
    print(f"  Matthews CC (MCC) : {m['mcc']:.4f}")

    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'Support':>8}")
    print(f"  {'-'*65}")
    for cls in CLASSES:
        cm = m["per_class"][cls]
        supp = cm["tp"] + cm["fn"]
        print(f"  {cls:<12} {cm['precision']:>10.4f} {cm['recall']:>8.4f} {cm['f1']:>8.4f}"
              f" {cm['tp']:>5} {cm['fp']:>5} {cm['fn']:>5} {supp:>8}")
    print(f"  {'-'*65}")
    print(f"  {'weighted avg':<12} {'':>10} {'':>8} {m['weighted_f1']:>8.4f}"
          f" {'':>5} {'':>5} {'':>5} {m['n_cases']:>8}")

    # --- Absolute confusion matrix ---
    print(f"\n  Confusion Matrix  (rows = actual, cols = predicted)")
    col_w = 9
    label = "actual \\ pred"
    header = f"  {label:<14}" + "".join(f"{c:>{col_w}}" for c in CLASSES) + f"{'| total':>9}"
    sep    = f"  {'-'*56}"
    print(header)
    print(sep)
    col_totals = {p: 0 for p in CLASSES}
    for exp in CLASSES:
        row   = m["confusion_matrix"].get(exp, {})
        total = sum(row.get(p, 0) for p in CLASSES)
        for p in CLASSES:
            col_totals[p] += row.get(p, 0)
        cols  = "".join(f"{row.get(p, 0):>{col_w}}" for p in CLASSES)
        print(f"  {exp:<14}{cols}  | {total:>4}")
    print(sep)
    grand = sum(col_totals.values())
    print(f"  {'total':<14}" + "".join(f"{col_totals[p]:>{col_w}}" for p in CLASSES) + f"  | {grand:>4}")

    # --- Normalised confusion matrix (% of actual) ---
    print(f"\n  Normalised Confusion Matrix  (row % of actual class)")
    print(header.replace("| total", ""))
    print(sep)
    for exp in CLASSES:
        row   = m["confusion_matrix"].get(exp, {})
        total = sum(row.get(p, 0) for p in CLASSES)
        if total:
            cols = "".join(f"{row.get(p, 0)/total*100:>{col_w-1}.1f}%" for p in CLASSES)
        else:
            cols = "".join(f"{'N/A':>{col_w}}" for _ in CLASSES)
        print(f"  {exp:<14}{cols}")

    # --- By difficulty ---
    if m.get("by_difficulty"):
        print(f"\n  Accuracy by Difficulty")
        print(f"  {'Level':<10} {'n':>5} {'Correct':>9} {'Accuracy':>10}")
        print(f"  {'-'*36}")
        for diff, s in m["by_difficulty"].items():
            bar = "#" * int(s["accuracy"] * 20)
            print(f"  {diff:<10} {s['n']:>5} {s['correct']:>9} {s['accuracy']:>10.4f}  {bar}")


# ---------------------------------------------------------------------------
# Decomposition quality evaluation
# ---------------------------------------------------------------------------

def run_decomp_quality(decomp_cases: list[dict], sql_kws: set[str], text_kws: set[str]) -> list[DecompResult]:
    from backend.router.router import QueryRouter
    router = QueryRouter()
    results = []

    for c in decomp_cases:
        qid          = c["id"]
        query        = c["query"]
        min_s        = c.get("min_subtasks", 1)
        max_s        = c.get("max_subtasks", 4)
        key_concepts = c.get("key_concepts", [])

        try:
            subtasks   = router.decompose_with_zeroshot(query)
            task_dicts = [{"sub_query": t.sub_query, "route": t.route} for t in subtasks]
        except Exception as exc:
            results.append(DecompResult(
                case_id=qid, query=query, subtasks=[],
                route_validity=0.0, atomicity=0.0, coverage=0.0,
                no_raw_sql=0.0, count_ok=False, quality_score=0.0,
                error=str(exc),
            ))
            continue

        route_val  = (sum(1 for t in subtasks if t.route in {"sql","text"}) / len(subtasks)
                      if subtasks else 0.0)
        atom_scores = [_atomicity(t.route, t.sub_query, sql_kws, text_kws) for t in subtasks]
        atomicity  = sum(atom_scores) / len(atom_scores) if atom_scores else 0.0
        coverage   = _coverage(key_concepts, task_dicts)
        no_raw     = (sum(1 for t in subtasks if not _RAW_SQL_RE.search(t.sub_query)) / len(subtasks)
                      if subtasks else 1.0)
        count_ok   = min_s <= len(subtasks) <= max_s
        quality    = (route_val + atomicity + coverage + no_raw) / 4

        results.append(DecompResult(
            case_id=qid, query=query, subtasks=task_dicts,
            route_validity=round(route_val, 4),
            atomicity=round(atomicity, 4),
            coverage=round(coverage, 4),
            no_raw_sql=round(no_raw, 4),
            count_ok=count_ok,
            quality_score=round(quality, 4),
        ))

    return results


def print_decomp_report(results: list[DecompResult]) -> None:
    for r in results:
        if r.error:
            print(f"\n  [!] {r.case_id}  ERROR: {r.error}")
            print(f"      {r.query[:80]}")
            continue
        cnt = "+" if r.count_ok else "-"
        print(f"\n  {r.case_id}  quality={r.quality_score:.2f}  count[{cnt}]={len(r.subtasks)}")
        print(f"  Query : {r.query[:75]}")
        for i, t in enumerate(r.subtasks, 1):
            raw_flag = "  [RAW-SQL!]" if _RAW_SQL_RE.search(t["sub_query"]) else ""
            print(f"    {i}. [{t['route']:<5}] {t['sub_query'][:65]}{raw_flag}")
        print(f"  route_validity={r.route_validity:.2f}  atomicity={r.atomicity:.2f}"
              f"  coverage={r.coverage:.2f}  no_raw_sql={r.no_raw_sql:.2f}")

    valid = [r for r in results if not r.error]
    if valid:
        n = len(valid)
        print(f"\n  {'-'*55}")
        print(f"  Cases evaluated : {n}")
        print(f"  Avg quality     : {sum(r.quality_score    for r in valid)/n:.4f}")
        print(f"  Avg route_valid : {sum(r.route_validity   for r in valid)/n:.4f}")
        print(f"  Avg atomicity   : {sum(r.atomicity        for r in valid)/n:.4f}")
        print(f"  Avg coverage    : {sum(r.coverage         for r in valid)/n:.4f}")
        print(f"  Avg no_raw_sql  : {sum(r.no_raw_sql       for r in valid)/n:.4f}")
        print(f"  Count-ok rate   : {sum(r.count_ok         for r in valid)/n:.1%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate decompose_with_zeroshot routing")
    parser.add_argument("--only-classification", action="store_true")
    parser.add_argument("--only-decomp",         action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(TEST_CASES_FILE)  as f: test_data   = json.load(f)
    with open(DECOMP_CASES_FILE) as f: decomp_data = json.load(f)

    cases        = test_data["test_cases"]
    sql_kws      = set(test_data.get("sql_keywords",  []))
    text_kws     = set(test_data.get("text_keywords", []))
    decomp_cases = decomp_data["test_cases"]

    output: dict = {}

    # -- Classification --
    if not args.only_decomp:
        print(f"\n{'='*63}")
        print(f"  Classification  (decompose_with_zeroshot -> route_from_subtasks)")
        print(f"{'='*63}")
        cls_results = run_classification(cases)
        cls_metrics = _compute_metrics(cls_results)
        print_classification_report(cls_results, cls_metrics)
        output["classification"] = {
            "metrics": cls_metrics,
            "per_case": [
                {"id": r.case_id, "query": r.query, "expected": r.expected,
                 "predicted": r.predicted, "correct": r.correct,
                 "difficulty": r.difficulty, "subtasks": r.subtasks,
                 "error": r.error}
                for r in cls_results
            ],
        }

    # -- Decomposition quality --
    if not args.only_classification:
        print(f"\n{'='*63}")
        print(f"  Decomposition Quality  (decompose_with_zeroshot)")
        print(f"{'='*63}")
        decomp_results = run_decomp_quality(decomp_cases, sql_kws, text_kws)
        print_decomp_report(decomp_results)
        valid = [r for r in decomp_results if not r.error]
        output["decomposition"] = {
            "per_case": [
                {"id": r.case_id, "query": r.query, "subtasks": r.subtasks,
                 "route_validity": r.route_validity, "atomicity": r.atomicity,
                 "coverage": r.coverage, "no_raw_sql": r.no_raw_sql,
                 "count_ok": r.count_ok, "quality_score": r.quality_score,
                 "error": r.error}
                for r in decomp_results
            ],
            "summary": {
                "avg_quality":       round(sum(r.quality_score  for r in valid)/len(valid), 4) if valid else 0,
                "avg_route_validity":round(sum(r.route_validity for r in valid)/len(valid), 4) if valid else 0,
                "avg_atomicity":     round(sum(r.atomicity      for r in valid)/len(valid), 4) if valid else 0,
                "avg_coverage":      round(sum(r.coverage       for r in valid)/len(valid), 4) if valid else 0,
                "avg_no_raw_sql":    round(sum(r.no_raw_sql     for r in valid)/len(valid), 4) if valid else 0,
                "count_ok_rate":     round(sum(r.count_ok       for r in valid)/len(valid), 4) if valid else 0,
            } if valid else {},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
