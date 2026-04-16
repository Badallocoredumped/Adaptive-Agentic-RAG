"""
evaluation/llm_judge.py
========================
Scores raw_results.json and produces every table and figure from the
TableRAG paper (adapted for your system):

  Table 1  — LLM-as-Judge accuracy: Overall / Single-Source / Multi-Source
  Table 2  — Latency: total latency (s) and avg latency per step (s)
  Table 4  — Exact Match accuracy (same layout as Table 1)
  Figure 5 — Execution iteration distribution (<3, 3, 4, 5, >5 steps)
  Figure 6 — Error distribution: Correct / Reasoning Error / Refusal

Usage:
    export GEMINI_API_KEY=your_key
    python evaluation/llm_judge.py

    # Skip LLM scoring (Exact Match only, free):
    python evaluation/llm_judge.py --no-llm

    # Score a specific results file (e.g. a baseline):
    python evaluation/llm_judge.py --results evaluation/results/naive_rag_results.json --system-name NaiveRAG
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Gemini config ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)
SLEEP_BETWEEN = 1.1   # seconds between Gemini calls (rate limit safety)


# =============================================================================
# EXACT MATCH
# =============================================================================

def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

def exact_match(predicted: str, ground_truth: str) -> int:
    return int(_normalise(predicted) == _normalise(ground_truth))


# =============================================================================
# LLM-AS-JUDGE  (exact prompt from TableRAG paper Appendix G)
# =============================================================================

JUDGE_PROMPT = """\
We would like to request your feedback on the performance of the AI assistant \
in response to the user question displayed above according to the gold answer. \
Please use the following listed aspects and their descriptions as evaluation criteria:
- Accuracy and Hallucinations: The assistant's answer is semantically consistent \
with the gold answer; The numerical value and order need to be accurate, and there \
should be no hallucinations.
- Completeness: Referring to the reference answers, the assistant's answer should \
contain all the key points needed to answer the user's question; further elaboration \
on these key points can be omitted.

Please rate whether this answer is suitable for the question. Please note that the \
gold answer can be considered as a correct answer to the question.

The assistant receives an overall score on a scale of 0 OR 1, where 0 means wrong \
and 1 means correct.

Directly output a line indicating the score of the Assistant.
PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS 0 OR 1 BY STRICTLY \
FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[1]]":
<start output>
Rating: [[score]]
<end output>

[Question]
{question}

[Gold Answer]
{gold}

[The Start of Assistant's Predicted Answer]
{predicted}
[The End of Assistant's Predicted Answer]
"""

def llm_judge_score(question: str, gold: str, predicted: str) -> int | None:
    if not GEMINI_API_KEY:
        return None
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold=gold,
        predicted=predicted if predicted else "[No answer provided]",
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 64},
    }
    try:
        resp = requests.post(GEMINI_URL, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        m = re.search(r"\[\[(\d)\]\]", text)
        if m:
            return int(m.group(1))
        m = re.search(r"\b([01])\b", text)
        if m:
            return int(m.group(1))
        print(f"  [WARN] Could not parse judge output: {text!r}")
        return None
    except Exception as e:
        print(f"  [WARN] Gemini API error: {e}")
        return None


# =============================================================================
# ERROR CLASSIFICATION  (matches Figure 6 in paper)
# =============================================================================

def classify_error(result: dict) -> str:
    """
    Classify each result into one of three categories (Figure 6):
      correct          — LLM judge = 1 (or EM = 1 if no LLM score)
      reasoning_error  — answered something wrong
      refusal          — empty answer, system error, or exceeded max iterations
    """
    predicted = str(result.get("predicted", "")).strip()
    has_error  = bool(result.get("error"))
    llm_score  = result.get("llm_judge")
    n_steps    = result.get("metadata", {}).get("n_steps")
    max_steps  = result.get("metadata", {}).get("max_steps", 5)

    if has_error or not predicted:
        return "refusal"

    if n_steps is not None and n_steps >= max_steps:
        return "refusal"

    if llm_score == 1:
        return "correct"
    if llm_score == 0:
        return "reasoning_error"

    # Fall back to exact match when no LLM score available
    return "correct" if result.get("exact_match", 0) == 1 else "reasoning_error"


# =============================================================================
# SCORING LOOP
# =============================================================================

def score_results(results: list[dict], use_llm: bool = True) -> list[dict]:
    total = len(results)
    print(f"\nScoring {total} results...\n")
    scored = []

    for i, r in enumerate(results, 1):
        predicted = str(r.get("predicted", "")).strip()
        gold      = r.get("ground_truth", "")
        question  = r.get("question", "")
        src       = r.get("answer_source", "?")

        em = exact_match(predicted, gold)

        llm_score = None
        if use_llm and not r.get("error") and predicted:
            llm_score = llm_judge_score(question, gold, predicted)
            time.sleep(SLEEP_BETWEEN)

        entry = {**r, "exact_match": em, "llm_judge": llm_score}
        scored.append(entry)

        judge_str = f"llm={llm_score}" if llm_score is not None else "llm=n/a"
        print(f"  [{i:>4}/{total}] {src:7s} em={em} {judge_str} | "
              f"gold='{gold[:25]}' | pred='{predicted[:25]}'")

    for entry in scored:
        entry["error_class"] = classify_error(entry)

    return scored


# =============================================================================
# TABLE 1 / TABLE 4 — Accuracy
# =============================================================================

def _acc(subset: list[dict], key: str) -> float | None:
    vals = [s[key] for s in subset if s.get(key) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals) * 100, 2)

def build_accuracy_tables(scored: list[dict]) -> dict:
    sql    = [s for s in scored if s.get("answer_source") == "sql"]
    text   = [s for s in scored if s.get("answer_source") == "text"]
    hybrid = [s for s in scored if s.get("answer_source") == "hybrid"]
    single = sql + text

    def row(subset):
        return {
            "n"          : len(subset),
            "llm_judge"  : _acc(subset, "llm_judge"),
            "exact_match": _acc(subset, "exact_match"),
        }

    return {
        "overall"      : row(scored),
        "single_source": row(single),
        "multi_source" : row(hybrid),
        "by_source"    : {"sql": row(sql), "text": row(text), "hybrid": row(hybrid)},
    }


# =============================================================================
# TABLE 2 — Latency
# =============================================================================

def build_latency_table(scored: list[dict]) -> dict:
    latencies_ms = [r.get("latency_ms", 0) for r in scored]
    total_s      = sum(latencies_ms) / 1000
    avg_per_q    = total_s / max(len(scored), 1)

    per_step = []
    for r in scored:
        ms = r.get("latency_ms", 0)
        n  = r.get("metadata", {}).get("n_steps") or 1
        per_step.append(ms / 1000 / n)

    avg_per_step = sum(per_step) / max(len(per_step), 1)

    return {
        "total_latency_s"       : round(total_s, 2),
        "avg_latency_per_q_s"   : round(avg_per_q, 3),
        "avg_latency_per_step_s": round(avg_per_step, 3),
        "n_questions"           : len(scored),
    }


# =============================================================================
# FIGURE 5 — Execution Iteration Distribution
# =============================================================================

def build_iteration_distribution(scored: list[dict]) -> dict:
    counts: Counter = Counter()
    for r in scored:
        n = r.get("metadata", {}).get("n_steps") or 1
        if n < 3:
            counts["less_than_3"] += 1
        elif n == 3:
            counts["3"] += 1
        elif n == 4:
            counts["4"] += 1
        elif n == 5:
            counts["5"] += 1
        else:
            counts["more_than_5"] += 1

    total = max(len(scored), 1)
    return {
        k: {"count": v, "pct": round(v / total * 100, 1)}
        for k, v in counts.items()
    }


# =============================================================================
# FIGURE 6 — Error Distribution
# =============================================================================

def build_error_distribution(scored: list[dict]) -> dict:
    counts: Counter = Counter(r.get("error_class", "reasoning_error") for r in scored)
    total = max(len(scored), 1)
    return {
        k: {"count": v, "pct": round(v / total * 100, 1)}
        for k, v in counts.items()
    }


# =============================================================================
# COMBINED COMPARISON TABLE (for comparing multiple systems like Table 1 in paper)
# =============================================================================

def compare_systems(system_summaries: list[dict]) -> str:
    """
    Prints a multi-row Table 1 with one row per system,
    matching the paper's layout exactly.
    """
    W = 80
    lines = [
        "",
        "=" * W,
        "  COMBINED TABLE 1 — LLM-as-Judge Accuracy (%) — All Systems",
        f"  {'Method':<26} {'HybridQA':>10} {'Single-Src':>12} {'Multi-Src':>10} {'Overall':>9}",
        f"  {'-'*26} {'-'*10} {'-'*12} {'-'*10} {'-'*9}",
    ]
    for s in system_summaries:
        name = s["system"][:26]
        acc  = s["accuracy"]
        lines.append(
            f"  {name:<26} "
            f"{'n/a':>10} "
            f"{fmt(acc['single_source']['llm_judge']):>12} "
            f"{fmt(acc['multi_source']['llm_judge']):>10} "
            f"{fmt(acc['overall']['llm_judge']):>9}"
        )
    lines += ["=" * W, ""]
    return "\n".join(lines)


# =============================================================================
# REPORT PRINTER
# =============================================================================

def fmt(val) -> str:
    return f"{val:.2f}" if val is not None else "  n/a"

def print_report(
    accuracy  : dict,
    latency   : dict,
    iterations: dict,
    errors    : dict,
    system_name: str,
) -> str:
    W = 70
    lines = [
        "",
        "=" * W,
        f"  RESULTS — {system_name}",
        f"  Dataset: HybridQA (tables > 100 cells, n={accuracy['overall']['n']})",
        "=" * W,
    ]

    # Table 1
    lines += [
        "",
        "  TABLE 1 — LLM-as-Judge Accuracy (%)",
        f"  {'':18} {'Overall':>10} {'Single-Src':>12} {'Multi-Src':>12}",
        f"  {'-'*18} {'-'*10} {'-'*12} {'-'*12}",
        f"  {system_name[:18]:<18} "
        f"{fmt(accuracy['overall']['llm_judge']):>10} "
        f"{fmt(accuracy['single_source']['llm_judge']):>12} "
        f"{fmt(accuracy['multi_source']['llm_judge']):>12}",
    ]

    # Table 4
    lines += [
        "",
        "  TABLE 4 — Exact Match Accuracy (%)",
        f"  {'':18} {'Overall':>10} {'Single-Src':>12} {'Multi-Src':>12}",
        f"  {'-'*18} {'-'*10} {'-'*12} {'-'*12}",
        f"  {system_name[:18]:<18} "
        f"{fmt(accuracy['overall']['exact_match']):>10} "
        f"{fmt(accuracy['single_source']['exact_match']):>12} "
        f"{fmt(accuracy['multi_source']['exact_match']):>12}",
    ]

    # By source
    lines += [
        "",
        "  Breakdown by source type:",
        f"  {'Source':<10} {'N':>6} {'LLM-Judge%':>12} {'ExactMatch%':>13}",
        f"  {'-'*10} {'-'*6} {'-'*12} {'-'*13}",
    ]
    for src in ["sql", "text", "hybrid"]:
        s = accuracy["by_source"][src]
        lines.append(
            f"  {src:<10} {s['n']:>6} "
            f"{fmt(s['llm_judge']):>12} "
            f"{fmt(s['exact_match']):>13}"
        )

    # Table 2
    lines += [
        "",
        "  TABLE 2 — Latency",
        f"  {'Method':<22} {'Total (s)':>10} {'Avg/Step (s)':>14}",
        f"  {'-'*22} {'-'*10} {'-'*14}",
        f"  {system_name[:22]:<22} "
        f"{latency['total_latency_s']:>10.2f} "
        f"{latency['avg_latency_per_step_s']:>14.3f}",
    ]

    # Figure 5
    lines += ["", "  FIGURE 5 — Execution Iteration Distribution"]
    order  = ["less_than_3", "3", "4", "5", "more_than_5"]
    labels = {
        "less_than_3": "< 3 steps", "3": "3 steps", "4": "4 steps",
        "5": "5 steps",             "more_than_5": "> 5 steps",
    }
    for k in order:
        v   = iterations.get(k, {"count": 0, "pct": 0.0})
        bar = "█" * int(v["pct"] / 2)
        lines.append(f"  {labels[k]:>12}  {bar:<35} {v['pct']:5.1f}%  (n={v['count']})")

    # Figure 6
    lines += ["", "  FIGURE 6 — Error Distribution"]
    err_labels = {
        "correct"        : "Correct",
        "reasoning_error": "Reasoning Error",
        "refusal"        : "Refusal / Exceeded Max Iter",
    }
    for k, label in err_labels.items():
        v   = errors.get(k, {"count": 0, "pct": 0.0})
        bar = "█" * int(v["pct"] / 2)
        lines.append(
            f"  {label:>30}  {bar:<25} {v['pct']:5.1f}%  (n={v['count']})"
        )

    lines += [
        "",
        "=" * W,
        "  Note: Single-Source = sql + text questions",
        "        Multi-Source  = hybrid questions",
        "        LLM judge uses exact prompt from TableRAG paper (Appendix G)",
        "",
    ]
    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=str(RESULTS_DIR / "raw_results.json"),
        help="Path to raw_results.json to score",
    )
    parser.add_argument("--no-llm", action="store_true", default=False)
    parser.add_argument("--system-name", default="Adaptive Agentic RAG")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.")
        print("Run  python evaluation/run_benchmark.py  first.")
        return

    if not args.no_llm and not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not set. Falling back to Exact Match only.\n")
        args.no_llm = True

    with open(results_path) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from {results_path}")

    scored     = score_results(results, use_llm=not args.no_llm)
    accuracy   = build_accuracy_tables(scored)
    latency    = build_latency_table(scored)
    iterations = build_iteration_distribution(scored)
    errors     = build_error_distribution(scored)

    stem         = results_path.stem
    scores_path  = RESULTS_DIR / f"{stem}_scored.json"
    summary_path = RESULTS_DIR / f"{stem}_summary.json"
    report_path  = RESULTS_DIR / f"{stem}_report.txt"

    with open(scores_path, "w") as f:
        json.dump(scored, f, indent=2)

    summary = {
        "system"    : args.system_name,
        "accuracy"  : accuracy,
        "latency"   : latency,
        "iterations": iterations,
        "errors"    : errors,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    report = print_report(accuracy, latency, iterations, errors, args.system_name)
    print(report)

    with open(report_path, "w") as f:
        f.write(report)

    print(f"Scores  → {scores_path}")
    print(f"Summary → {summary_path}")
    print(f"Report  → {report_path}")
    print("\nNext: run baselines with  python evaluation/baselines.py")


if __name__ == "__main__":
    main()
