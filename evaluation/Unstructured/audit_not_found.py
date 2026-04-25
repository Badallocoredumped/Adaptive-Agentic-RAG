"""
audit_not_found.py
==================
Audits the "Not found" rate across all ragbench_* configs.

Steps:
  1. Classify every "Not found" into TRUE_ABSTENTION / FALSE_ABSTENTION / NO_GOLD
  2. Inspect FALSE_ABSTENTION cases in detail
  3. Cross-reference RAGAS scoring of "Not found" responses with NO_GOLD bucket
  4. Summarise abstention behaviour and produce fix recommendations
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_not_found(answer: str) -> bool:
    return "not found" in answer.lower()

def load_rag_results(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_ragas_scores(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ── Step 1: Classify "Not found" responses across all configs ─────────────────

# Focus on the key ragbench_RAG_* configs (not C/T-series internal runs)
TARGET_PREFIXES = ["ragbench_RAG_", "ragbench_BASE_NO_RAG"]

rag_result_files = sorted([
    p for p in RESULTS_DIR.glob("ragbench_*_rag_results.json")
    if any(p.name.startswith(pfx) for pfx in TARGET_PREFIXES)
])

print(f"\n{'='*70}")
print("  STEP 1 — 'Not Found' Classification Across All Configs")
print(f"{'='*70}\n")

# Track per-config and aggregate counts
agg = defaultdict(lambda: {"TRUE_ABSTENTION": 0, "FALSE_ABSTENTION": 0, "NO_GOLD": 0, "total_nf": 0, "total": 0})

# Store FALSE_ABSTENTION cases for Step 2
false_abstentions: list[dict] = []  # {question, contexts, gold_ids, config}

per_config_summary = []

for rfile in rag_result_files:
    cfg_name = rfile.name.replace("_rag_results.json", "")
    data = load_rag_results(rfile)

    nf_items = [r for r in data if is_not_found(r.get("answer", ""))]
    total = len(data)
    total_nf = len(nf_items)

    counts = {"TRUE_ABSTENTION": 0, "FALSE_ABSTENTION": 0, "NO_GOLD": 0}

    for r in nf_items:
        gold_ids = r.get("gold_passage_ids") or []
        gold_hit = r.get("gold_hit")

        if not gold_ids:
            counts["NO_GOLD"] += 1
        elif gold_hit is True or gold_hit is None:
            counts["FALSE_ABSTENTION"] += 1
            # Collect for Step 2
            false_abstentions.append({
                "config": cfg_name,
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "retrieved_contexts": r.get("retrieved_contexts", []),
                "gold_passage_ids": gold_ids,
                "gold_hit": gold_hit,
            })
        else:
            counts["TRUE_ABSTENTION"] += 1

    for k in counts:
        agg[k][k] += counts[k]
    agg["TOTAL"]["total_nf"] += total_nf
    agg["TOTAL"]["total"] += total

    per_config_summary.append((cfg_name, total, total_nf, counts))

    # Print per-config table
    pct = lambda n: f"{n/total_nf*100:.1f}%" if total_nf else "N/A"
    print(f"  Config : {cfg_name}")
    print(f"  Total  : {total} questions | Not Found: {total_nf} ({total_nf/total*100:.1f}%)")
    print(f"    TRUE_ABSTENTION  (retrieval failed)  : {counts['TRUE_ABSTENTION']:>3}  ({pct(counts['TRUE_ABSTENTION'])})")
    print(f"    FALSE_ABSTENTION (retrieval succeeded): {counts['FALSE_ABSTENTION']:>3}  ({pct(counts['FALSE_ABSTENTION'])})")
    print(f"    NO_GOLD          (no gold passage)    : {counts['NO_GOLD']:>3}  ({pct(counts['NO_GOLD'])})")
    print()

# ── Aggregate summary ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  AGGREGATE 'Not Found' Totals (all configs combined)")
print(f"{'='*70}")

grand_total = sum(r[1] for r in per_config_summary)
grand_nf = sum(r[2] for r in per_config_summary)
grand_ta = sum(r[3]["TRUE_ABSTENTION"] for r in per_config_summary)
grand_fa = sum(r[3]["FALSE_ABSTENTION"] for r in per_config_summary)
grand_ng = sum(r[3]["NO_GOLD"] for r in per_config_summary)

def agg_pct(n):
    return f"{n/grand_nf*100:.1f}%" if grand_nf else "N/A"

print(f"  Grand total questions  : {grand_total}")
print(f"  Grand total 'Not Found': {grand_nf}  ({grand_nf/grand_total*100:.1f}% overall)")
print(f"    TRUE_ABSTENTION   : {grand_ta:>4}  ({agg_pct(grand_ta)})")
print(f"    FALSE_ABSTENTION  : {grand_fa:>4}  ({agg_pct(grand_fa)})  ← FIXABLE")
print(f"    NO_GOLD           : {grand_ng:>4}  ({agg_pct(grand_ng)})  ← Methodological issue")

# ── Step 2: Inspect FALSE_ABSTENTION cases ────────────────────────────────────
print(f"\n\n{'='*70}")
print("  STEP 2 — FALSE_ABSTENTION Case Inspection")
print(f"  (gold_hit=True/None but LLM still said 'Not found')")
print(f"{'='*70}\n")

# Deduplicate by question (same question may appear in multiple configs)
seen_questions = {}
for fa in false_abstentions:
    q = fa["question"]
    if q not in seen_questions:
        seen_questions[q] = {
            "question": q,
            "answer": fa["answer"],
            "retrieved_contexts": fa["retrieved_contexts"],
            "gold_passage_ids": fa["gold_passage_ids"],
            "gold_hit": fa["gold_hit"],
            "configs": [],
        }
    seen_questions[q]["configs"].append(fa["config"])

unique_fa = list(seen_questions.values())
print(f"  Total unique FALSE_ABSTENTION questions: {len(unique_fa)}\n")

for i, case in enumerate(unique_fa, 1):
    print(f"  ── Case {i}/{len(unique_fa)} {'─'*45}")
    print(f"  Question  : {case['question']}")
    print(f"  Answer    : {case['answer']}")
    print(f"  gold_hit  : {case['gold_hit']}")
    print(f"  Gold IDs  : {case['gold_passage_ids']}")
    print(f"  Configs   : {', '.join(case['configs'])}")
    print(f"  Retrieved contexts ({len(case['retrieved_contexts'])} chunks):")
    for j, ctx in enumerate(case["retrieved_contexts"][:5], 1):
        print(f"    [{j}] {ctx[:300].strip()}{'...' if len(ctx)>300 else ''}")
    if len(case["retrieved_contexts"]) > 5:
        print(f"    ... +{len(case['retrieved_contexts'])-5} more chunks")
    print()

# ── Step 3: RAGAS scoring of "Not found" responses ───────────────────────────
print(f"\n{'='*70}")
print("  STEP 3 — RAGAS Scoring of 'Not Found' Responses")
print(f"{'='*70}\n")

# Build NO_GOLD question set for cross-reference
no_gold_questions = set()
for rfile in rag_result_files:
    data = load_rag_results(rfile)
    for r in data:
        if is_not_found(r.get("answer", "")) and not (r.get("gold_passage_ids") or []):
            no_gold_questions.add(r.get("question", ""))

ragas_files = sorted([
    p for p in RESULTS_DIR.glob("ragbench_*_ragas_scores.json")
    if any(p.name.startswith(pfx) for pfx in TARGET_PREFIXES)
])

METRIC_KEYS = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]

for rfile in ragas_files:
    cfg_name = rfile.name.replace("_ragas_scores.json", "")
    data = load_ragas_scores(rfile)
    if data is None or "per_question" not in data:
        continue

    per_q = data["per_question"]
    nf_rows = [row for row in per_q if is_not_found(str(row.get("response", row.get("answer", ""))))]

    if not nf_rows:
        continue

    print(f"  Config: {cfg_name}  |  'Not Found' rows scored: {len(nf_rows)}")

    # Categorise by NO_GOLD membership
    no_gold_rows = [r for r in nf_rows if r.get("user_input", r.get("question", "")) in no_gold_questions]
    gold_rows    = [r for r in nf_rows if r.get("user_input", r.get("question", "")) not in no_gold_questions]

    def avg(rows, key):
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        return sum(vals)/len(vals) if vals else float("nan")

    print(f"    ┌─ All 'Not Found' rows (n={len(nf_rows)})")
    for m in METRIC_KEYS:
        print(f"    │   {m:<24}: {avg(nf_rows, m):.4f}")

    if no_gold_rows:
        print(f"    ├─ NO_GOLD subset (n={len(no_gold_rows)}) — correct abstentions being penalised")
        for m in METRIC_KEYS:
            print(f"    │   {m:<24}: {avg(no_gold_rows, m):.4f}")

    if gold_rows:
        print(f"    └─ Non-NO_GOLD 'Not Found' rows (n={len(gold_rows)}) — incorrect abstentions")
        for m in METRIC_KEYS:
            print(f"        {m:<24}: {avg(gold_rows, m):.4f}")
    print()

# ── Step 4: Prompt / abstention behaviour (from source code) ──────────────────
print(f"\n{'='*70}")
print("  STEP 4 — Prompt / LLM Abstention Behaviour (Static Analysis)")
print(f"{'='*70}")
print("""
  Source: evaluation/Unstructured/run_ragbench_eval.py  (lines 265-275)

  ANSWER_PROMPT:
  ──────────────────────────────────────────────────────────────────
  Answer the following question in 2-3 sentences using ONLY the context below.
  Be specific and ground every claim in the provided context.
  If the answer cannot be found in the context, reply exactly: "Not found."

  Context:
  {context}

  Question: {question}

  Answer:
  ──────────────────────────────────────────────────────────────────

  Findings:
  • "Not found." is a HARDCODED INSTRUCTION in the system prompt.
  • The model is told to reply EXACTLY "Not found." — no partial answers allowed.
  • There is NO similarity score cutoff or pre-LLM abstention gate.
  • The instruction is binary: either full 2-3 sentence answer OR "Not found."
  • This is extremely conservative: the model is given no middle ground to
    say "Based on the context, the answer appears to be X, although I'm
    uncertain." This drives up FALSE_ABSTENTION.
""")

# ── Step 5: Fix Recommendations ───────────────────────────────────────────────
print(f"\n{'='*70}")
print("  STEP 5 — Fix Recommendations")
print(f"{'='*70}\n")

# Corrected not-found rate excluding NO_GOLD questions
# Focus on the best config: RAG_K15_BGE_BASE
k15_file = RESULTS_DIR / "ragbench_RAG_K15_BGE_BASE_rag_results.json"
if k15_file.exists():
    k15_data = load_rag_results(k15_file)
    k15_total = len(k15_data)
    k15_nf_all = [r for r in k15_data if is_not_found(r.get("answer", ""))]
    k15_no_gold = [r for r in k15_nf_all if not (r.get("gold_passage_ids") or [])]
    k15_nf_corrected = [r for r in k15_nf_all if r.get("gold_passage_ids")]  # exclude NO_GOLD

    print(f"  Best config (RAG_K15_BGE_BASE) — Not-Found Rate Analysis:")
    print(f"    Raw not-found rate          : {len(k15_nf_all)/k15_total*100:.1f}%  ({len(k15_nf_all)}/{k15_total})")
    print(f"    NO_GOLD questions           : {len(k15_no_gold)}  (unanswerable — 'Not found' is correct)")
    print(f"    Corrected not-found rate    : {len(k15_nf_corrected)/(k15_total - len(k15_no_gold))*100:.1f}%"
          f"  ({len(k15_nf_corrected)}/{k15_total-len(k15_no_gold)}) — after excluding NO_GOLD")

print("""
  ── Recommendation 1: PROMPT FIX (addresses FALSE_ABSTENTION) ──────────────

  PROBLEM: Current prompt is maximally conservative. The binary instruction
  "if not found, say EXACTLY 'Not found.'" leaves the LLM no room to
  synthesise a partial or hedged answer from context that contains the
  information but not in the exact phrasing the model expects.

  REVISED PROMPT (drop-in replacement for ANSWER_PROMPT in run_ragbench_eval.py):
  ────────────────────────────────────────────────────────────────────────────
  Answer the following question in 2-3 sentences using the context below.
  Ground every claim in the context. Do NOT add outside knowledge.

  If the context contains relevant information — even partial — attempt an
  answer and note any uncertainty.
  Only reply exactly "Not found." if the context contains NO information
  whatsoever that is relevant to the question.

  Context:
  {context}

  Question: {question}

  Answer:
  ────────────────────────────────────────────────────────────────────────────
  This change lowers the bar for answering, which should reduce FALSE_ABSTENTION
  while keeping TRUE_ABSTENTION intact (genuinely irrelevant context).

  ── Recommendation 2: EVALUATION FIX (addresses NO_GOLD bias) ──────────────

  PROBLEM: NO_GOLD questions — where gold_passage_ids is empty — have no
  retrievable answer in the corpus. RAGAS scores "Not found." as 0 for
  answer_relevancy and faithfulness, which artificially inflates the
  penalty and suppresses overall scores.

  FIX: Exclude NO_GOLD questions from not_found_rate calculations.
  In run_ragbench_eval.py, compute two rates:

    not_found_rate_raw      = not_found / len(valid)
    answerable              = [r for r in valid if r.get("gold_passage_ids")]
    not_found_rate_adjusted = sum(nf for nf in answerable if "not found" in answer) / len(answerable)

  Document NO_GOLD questions as "unanswerable" in the benchmark metadata.
  Their "Not found." response should be scored 1.0, not 0.0.

  ── Recommendation 3: HYBRID (best path forward) ────────────────────────────

  Apply BOTH fixes simultaneously:
  1. Relax the prompt to reduce FALSE_ABSTENTION → fewer incorrect abstentions
  2. Exclude NO_GOLD from not_found_rate → corrected benchmark metric
  3. Optionally: tag NO_GOLD questions in per_question output with
     "unanswerable": True so RAGAS or downstream tools can skip them.

  Expected outcome:
  • not_found_rate drops significantly (NO_GOLD removed + prompt fix)
  • faithfulness and answer_relevancy rise (no zero-scores for correct abstentions)
  • Gold hit rate unaffected (retrieval step unchanged)
""")

print(f"{'='*70}")
print("  AUDIT COMPLETE")
print(f"{'='*70}\n")
