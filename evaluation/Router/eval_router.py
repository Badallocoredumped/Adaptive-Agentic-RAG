"""
evaluation/Router/eval_router.py
=================================
CLASSIFICATION mode  (--mode classification)
  Compares three single-route classifiers on test_cases.json:
    keyword  -> router.route()                              (rule-based, no model)
    semantic -> router.route_with_semantic()                (embedding cosine, no LLM)
    zeroshot -> decompose_with_zeroshot() + route_from_subtasks()  (LLM)

  Per-router output  : evaluation/Router/results/classification_{router}.json
  Comparison output  : evaluation/Router/results/classification_comparison.json

DECOMPOSITION mode  (--mode decomposition)
  Evaluates decompose_with_zeroshot() quality on decomp_cases.json:
    route_validity · atomicity · coverage · no_raw_sql

  Output : evaluation/Router/results/decomposition.json

Usage:
  python evaluation/Router/eval_router.py                    # both modes
  python evaluation/Router/eval_router.py --mode classification
  python evaluation/Router/eval_router.py --mode decomposition
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROUTER_DIR        = Path(__file__).resolve().parent
TEST_CASES_FILE   = ROUTER_DIR / "test_cases.json"
DECOMP_CASES_FILE = ROUTER_DIR / "decomp_cases.json"
RESULTS_DIR       = ROUTER_DIR / "results"

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
# Metric helpers
# ---------------------------------------------------------------------------

def _cohen_kappa(confusion: dict, classes: list[str]) -> float:
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
    n = sum(confusion[e][p] for e in classes for p in classes)
    if n == 0:
        return 0.0
    c   = sum(confusion[k][k] for k in classes)
    t_k = [sum(confusion[k][p] for p in classes) for k in classes]
    p_k = [sum(confusion[e][k] for e in classes) for k in classes]
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
        return 0.7
    if route == "sql":
        return 1.0 if sql_hits >= text_hits else 0.4
    return 1.0 if text_hits >= sql_hits else 0.4


def _coverage(key_concepts: list[str], subtasks: list[dict]) -> float:
    if not key_concepts:
        return 1.0
    combined = " ".join(t.get("sub_query", "") for t in subtasks).lower()
    return sum(1 for c in key_concepts if c.lower() in combined) / len(key_concepts)


# ---------------------------------------------------------------------------
# Classification runners
# ---------------------------------------------------------------------------

def _run_single_route(cases: list[dict], route_fn) -> list[ClassResult]:
    """Generic runner: calls route_fn(query) -> str for every case."""
    results = []
    for c in cases:
        qid, query, exp = c["id"], c["query"], c["expected_route"]
        diff = c.get("difficulty", "unknown")
        try:
            predicted = route_fn(query)
            results.append(ClassResult(
                case_id=qid, query=query, expected=exp,
                predicted=predicted, correct=(predicted == exp),
                difficulty=diff, subtasks=[],
            ))
        except Exception as exc:
            results.append(ClassResult(
                case_id=qid, query=query, expected=exp,
                predicted="error", correct=False,
                difficulty=diff, subtasks=[], error=str(exc),
            ))
        mark = "+" if results[-1].correct else ("-" if not results[-1].error else "!")
        print(f"  [{mark}] {qid:<12} exp={exp:<7} pred={results[-1].predicted:<7}")
    return results


def run_keyword_classification(cases: list[dict]) -> list[ClassResult]:
    from backend.router.router import QueryRouter
    router = QueryRouter()
    print(f"\n{'='*63}")
    print(f"  KEYWORD  classification  (router.route)")
    print(f"{'='*63}")
    return _run_single_route(cases, router.route)


def run_semantic_classification(cases: list[dict]) -> list[ClassResult]:
    from backend.router.router import QueryRouter
    router = QueryRouter()
    print(f"\n{'='*63}")
    print(f"  SEMANTIC  classification  (route_with_semantic)")
    print(f"{'='*63}")
    return _run_single_route(cases, router.route_with_semantic)


def run_zeroshot_classification(cases: list[dict]) -> list[ClassResult]:
    from backend.router.router import QueryRouter
    router = QueryRouter()
    print(f"\n{'='*63}")
    print(f"  ZEROSHOT  classification  (decompose_with_zeroshot -> route_from_subtasks)")
    print(f"{'='*63}")
    results = []
    for c in cases:
        qid, query, exp = c["id"], c["query"], c["expected_route"]
        diff = c.get("difficulty", "unknown")
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
        print(f"  [{mark}] {qid:<12} exp={exp:<7} pred={results[-1].predicted:<7}  {sub_info[:60]}")
    return results


# ---------------------------------------------------------------------------
# Classification report printers
# ---------------------------------------------------------------------------

def print_classification_report(router_name: str, results: list[ClassResult], metrics: dict) -> None:
    m = metrics
    n_correct = sum(r.correct for r in results if not r.error)
    label = router_name.upper()

    print(f"\n  {'='*65}")
    print(f"  {label} — CLASSIFICATION REPORT")
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
        print(f"  {cls:<12} {cm['precision']:>10.4f} {cm['recall']:>8.4f} {cm['f1']:>8.4f}"
              f" {cm['tp']:>5} {cm['fp']:>5} {cm['fn']:>5} {cm['support']:>8}")

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
        cols = "".join(f"{row.get(p, 0):>{col_w}}" for p in CLASSES)
        print(f"  {exp:<14}{cols}  | {total:>4}")
    print(sep)
    grand = sum(col_totals.values())
    print(f"  {'total':<14}" + "".join(f"{col_totals[p]:>{col_w}}" for p in CLASSES) + f"  | {grand:>4}")

    if m.get("by_difficulty"):
        print(f"\n  Accuracy by Difficulty")
        print(f"  {'Level':<10} {'n':>5} {'Correct':>9} {'Accuracy':>10}")
        print(f"  {'-'*36}")
        for diff, s in m["by_difficulty"].items():
            bar = "#" * int(s["accuracy"] * 20)
            print(f"  {diff:<10} {s['n']:>5} {s['correct']:>9} {s['accuracy']:>10.4f}  {bar}")


def print_three_way_comparison(all_results: dict[str, list[ClassResult]], all_metrics: dict[str, dict]) -> None:
    routers = list(all_results.keys())

    routers_label = "  |  ".join(routers)
    print(f"\n{'='*75}")
    print(f"  COMPARISON  ({routers_label})")
    print(f"{'='*75}")

    metric_rows = [
        ("Accuracy",    "accuracy"),
        ("Macro F1",    "macro_f1"),
        ("Weighted F1", "weighted_f1"),
        ("Cohen's κ",   "cohen_kappa"),
        ("MCC",         "mcc"),
    ]
    print(f"  {'Metric':<18}" + "".join(f"  {r.capitalize():>10}" for r in routers))
    print(f"  {'-'*58}")
    for label, key in metric_rows:
        vals = "".join(f"  {all_metrics[r][key]:>10.4f}" for r in routers)
        print(f"  {label:<18}{vals}")

    print(f"\n  Per-class F1")
    print(f"  {'Class':<12}" + "".join(f"  {r.capitalize():>10}" for r in routers))
    print(f"  {'-'*46}")
    for cls in CLASSES:
        vals = "".join(f"  {all_metrics[r]['per_class'][cls]['f1']:>10.4f}" for r in routers)
        print(f"  {cls:<12}{vals}")

    # Cases where routers disagree
    all_ids = [r.case_id for r in all_results["keyword"]]
    by_id: dict[str, dict[str, ClassResult]] = {
        cid: {rname: next(r for r in all_results[rname] if r.case_id == cid) for rname in routers}
        for cid in all_ids
    }

    disagreements = [
        cid for cid in all_ids
        if len({by_id[cid][r].predicted for r in routers if not by_id[cid][r].error}) > 1
    ]

    print(f"\n  Cases where routers disagree  ({len(disagreements)} / {len(all_ids)})")
    if disagreements:
        header_cols = "".join(f"  {r.capitalize():>10}" for r in routers)
        print(f"  {'ID':<14} {'Expected':<9}{header_cols}  {'Votes correct'}")
        print(f"  {'-'*72}")
        for cid in disagreements:
            exp = by_id[cid][routers[0]].expected
            preds = "".join(
                f"  {by_id[cid][r].predicted:>10}" for r in routers
            )
            correct_flags = " ".join(
                f"{r[0].upper()}{'✓' if by_id[cid][r].correct else '✗'}"
                for r in routers if not by_id[cid][r].error
            )
            print(f"  {cid:<14} {exp:<9}{preds}  {correct_flags}")
    else:
        print("  (all three routers agree on every case)")


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

        route_val   = (sum(1 for t in subtasks if t.route in {"sql", "text"}) / len(subtasks)
                       if subtasks else 0.0)
        atom_scores = [_atomicity(t.route, t.sub_query, sql_kws, text_kws) for t in subtasks]
        atomicity   = sum(atom_scores) / len(atom_scores) if atom_scores else 0.0
        coverage    = _coverage(key_concepts, task_dicts)
        no_raw      = (sum(1 for t in subtasks if not _RAW_SQL_RE.search(t.sub_query)) / len(subtasks)
                       if subtasks else 1.0)
        count_ok    = min_s <= len(subtasks) <= max_s
        quality     = (route_val + atomicity + coverage + no_raw) / 4

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
# Save helpers
# ---------------------------------------------------------------------------

def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved -> {path.relative_to(PROJECT_ROOT)}")


def save_classification_result(router_name: str, results: list[ClassResult], metrics: dict) -> None:
    path = RESULTS_DIR / f"classification_{router_name}.json"
    _save(path, {
        "router": router_name,
        "metrics": metrics,
        "per_case": [
            {"id": r.case_id, "query": r.query, "expected": r.expected,
             "predicted": r.predicted, "correct": r.correct,
             "difficulty": r.difficulty, "subtasks": r.subtasks, "error": r.error}
            for r in results
        ],
    })


def save_comparison(all_results: dict[str, list[ClassResult]], all_metrics: dict[str, dict]) -> None:
    routers = list(all_results.keys())
    all_ids = [r.case_id for r in all_results[routers[0]]]
    by_id   = {
        cid: {rn: next(r for r in all_results[rn] if r.case_id == cid) for rn in routers}
        for cid in all_ids
    }

    disagreements = []
    for cid in all_ids:
        preds = {rn: by_id[cid][rn].predicted for rn in routers if not by_id[cid][rn].error}
        if len(set(preds.values())) > 1:
            disagreements.append({
                "id":       cid,
                "expected": by_id[cid][routers[0]].expected,
                "predictions": preds,
                "correct":  {rn: by_id[cid][rn].correct for rn in routers if not by_id[cid][rn].error},
            })

    path = RESULTS_DIR / "classification_comparison.json"
    _save(path, {
        "routers": routers,
        "metrics_summary": {
            rn: {
                "accuracy":    all_metrics[rn]["accuracy"],
                "macro_f1":    all_metrics[rn]["macro_f1"],
                "weighted_f1": all_metrics[rn]["weighted_f1"],
                "cohen_kappa": all_metrics[rn]["cohen_kappa"],
                "mcc":         all_metrics[rn]["mcc"],
                "per_class_f1": {
                    cls: all_metrics[rn]["per_class"][cls]["f1"] for cls in CLASSES
                },
            }
            for rn in routers
        },
        "n_disagreements": len(disagreements),
        "disagreements":   disagreements,
    })


def save_decomposition(results: list[DecompResult]) -> None:
    valid = [r for r in results if not r.error]
    n = len(valid)
    path = RESULTS_DIR / "decomposition.json"
    _save(path, {
        "router": "zeroshot",
        "summary": {
            "n_cases":           n,
            "avg_quality":       round(sum(r.quality_score  for r in valid) / n, 4) if n else 0,
            "avg_route_validity":round(sum(r.route_validity for r in valid) / n, 4) if n else 0,
            "avg_atomicity":     round(sum(r.atomicity      for r in valid) / n, 4) if n else 0,
            "avg_coverage":      round(sum(r.coverage       for r in valid) / n, 4) if n else 0,
            "avg_no_raw_sql":    round(sum(r.no_raw_sql     for r in valid) / n, 4) if n else 0,
            "count_ok_rate":     round(sum(r.count_ok       for r in valid) / n, 4) if n else 0,
        },
        "per_case": [
            {"id": r.case_id, "query": r.query, "subtasks": r.subtasks,
             "route_validity": r.route_validity, "atomicity": r.atomicity,
             "coverage": r.coverage, "no_raw_sql": r.no_raw_sql,
             "count_ok": r.count_ok, "quality_score": r.quality_score,
             "error": r.error}
            for r in results
        ],
    })


# ---------------------------------------------------------------------------
# Top-level eval orchestrators
# ---------------------------------------------------------------------------

def run_classification_eval(cases: list[dict], routers: list[str] | None = None) -> None:
    if routers is None:
        routers = ["keyword", "semantic", "zeroshot"]

    all_runners = {
        "keyword":  run_keyword_classification,
        "semantic": run_semantic_classification,
        "zeroshot": run_zeroshot_classification,
    }

    all_results: dict[str, list[ClassResult]] = {}
    all_metrics: dict[str, dict] = {}

    for router_name in routers:
        results = all_runners[router_name](cases)
        metrics = _compute_metrics(results)
        print_classification_report(router_name, results, metrics)
        save_classification_result(router_name, results, metrics)
        all_results[router_name] = results
        all_metrics[router_name] = metrics

    if len(routers) > 1:
        print_three_way_comparison(all_results, all_metrics)
        save_comparison(all_results, all_metrics)


def run_decomposition_eval(decomp_cases: list[dict], sql_kws: set[str], text_kws: set[str]) -> None:
    print(f"\n{'='*63}")
    print(f"  DECOMPOSITION QUALITY  (decompose_with_zeroshot)")
    print(f"{'='*63}")
    results = run_decomp_quality(decomp_cases, sql_kws, text_kws)
    print_decomp_report(results)
    save_decomposition(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Router evaluation: classification and decomposition")
    parser.add_argument(
        "--mode",
        choices=["classification", "decomposition", "all"],
        default="all",
        help="classification: compare keyword/semantic/zeroshot as single-route classifiers; "
             "decomposition: evaluate zeroshot query splitting quality; "
             "all: run both (default)",
    )
    parser.add_argument(
        "--router",
        choices=["keyword", "semantic", "zeroshot"],
        nargs="+",
        default=["keyword", "semantic", "zeroshot"],
        help="which router(s) to include in classification eval (default: all three)",
    )
    args = parser.parse_args()

    with open(TEST_CASES_FILE,  encoding="utf-8") as f: test_data   = json.load(f)
    with open(DECOMP_CASES_FILE, encoding="utf-8") as f: decomp_data = json.load(f)

    cases        = test_data["test_cases"]
    sql_kws      = set(test_data.get("sql_keywords",  []))
    text_kws     = set(test_data.get("text_keywords", []))
    decomp_cases = decomp_data["test_cases"]

    if args.mode in ("classification", "all"):
        run_classification_eval(cases, routers=args.router)

    if args.mode in ("decomposition", "all"):
        run_decomposition_eval(decomp_cases, sql_kws, text_kws)

    print(f"\nAll results saved to {RESULTS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
