"""
BIRD Benchmark Harness

QUICK START — run from anywhere in the repo:

  # Generate predictions AND score them in one command
  python bird_eval.py --ablation tablerag_react --evaluate

  # Quick smoke test (20 questions, auto-score)
  python bird_eval.py --ablation full_single_pass --limit 20 --evaluate

  # Score an existing predictions file without re-running the model
  python bird_eval.py --ablation tablerag_react --score_only

  # Include BIRD's evidence hints in the prompt
  python bird_eval.py --ablation tablerag_react --use_evidence --evaluate

  # Only simple questions
  python bird_eval.py --ablation tablerag_single_pass --difficulty simple --evaluate

Ablation modes:
  full_single_pass     — Full schema + single LLM call (no TableRAG, no ReAct)
  tablerag_single_pass — TableRAG schema pruning + single SQL agent call
  tablerag_react       — TableRAG schema pruning + multi-turn ReAct agent
"""

import json
import subprocess
import sys
import argparse
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Locate project root robustly — works regardless of which directory you run from
# ---------------------------------------------------------------------------
def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    while current != current.parent:
        if (current / "backend").is_dir():
            return current
        current = current.parent
    raise RuntimeError("Cannot find project root. Make sure 'backend/' exists in an ancestor directory.")


project_root = _find_project_root(Path(__file__))
sys.path.insert(0, str(project_root))

from backend import config
from backend.sql.react_agent import run_react_sql_agent
from backend.sql.database import get_live_schema
from backend.sql.table_rag import build_schema_index, retrieve_relevant_schema
from backend.sql.sql_agent import run_sql_agent
from backend.sql import database as _db_module
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Fixed paths (all derived from project root)
# ---------------------------------------------------------------------------
BIRD_DIR    = project_root / "evaluation" / "Structured" / "bird"
EVAL_SCRIPT = BIRD_DIR / "evaluation" / "evaluation_ex.py"
DEV_JSON    = BIRD_DIR / "dev.json"
DEV_TABLES  = BIRD_DIR / "dev_tables.json"
DB_ROOT     = BIRD_DIR / "dev_databases" / "dev_databases"
OUTPUT_DIR  = project_root / "evaluation" / "Structured"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_sql(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        s = s.rsplit("```", 1)[0]
    s = s.replace("\n", " ").replace("\r", " ")
    if ";" in s:
        s = s.split(";")[0]
    s = s.strip()
    return s if s else "SELECT 1"


def load_schema_map() -> dict:
    with open(DEV_TABLES, "r", encoding="utf-8") as f:
        data = json.load(f)
    schema_map = {}
    for db in data:
        db_id  = db["db_id"]
        tables = db["table_names_original"]
        cols   = db["column_names_original"]
        lines  = []
        for i, tbl in enumerate(tables):
            tcols = [c[1] for c in cols if c[0] == i]
            lines.append(f"Table: {tbl} | Columns: {', '.join(tcols)}")
        schema_map[db_id] = "\n".join(lines)
    return schema_map


def run_single_pass_baseline(question: str, schema: str, evidence: str = "") -> str:
    llm = ChatOpenAI(model=config.SQL_OPENAI_MODEL, temperature=0.0)
    ev  = f"\nEvidence/Hints:\n{evidence}" if evidence else ""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert SQLite developer. Write a valid SQLite query for the question below.\n"
         "STRICT RULE: Return ONLY the raw SQL — no markdown, no fences, no explanations.\n\n"
         "Schema:\n{schema}{ev}"),
        ("human", "{q}"),
    ])
    resp = (prompt | llm).invoke({"schema": schema, "ev": ev, "q": question})
    return clean_sql(str(resp.content))


def make_stem(ablation: str, use_evidence: bool, difficulty: str, limit) -> str:
    parts = [ablation]
    if use_evidence:
        parts.append("evidence")
    if difficulty != "all":
        parts.append(difficulty)
    if limit:
        parts.append(f"limit{limit}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------
def generate_predictions(args, stem: str):
    predictions_path = OUTPUT_DIR / f"predictions_bird_{stem}.json"
    gold_path        = OUTPUT_DIR / f"gold_bird_{stem}.sql"

    print(f"\n{'='*60}")
    print(f"  Mode      : {args.ablation}")
    print(f"  Evidence  : {'ON' if args.use_evidence else 'OFF'}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Limit     : {args.limit or 'all'}")
    print(f"{'='*60}\n")

    with open(DEV_JSON, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if args.difficulty != "all":
        queries = [q for q in queries if q.get("difficulty") == args.difficulty]
        print(f"After difficulty filter: {len(queries)} questions")

    if args.limit:
        queries = queries[:args.limit]

    total = len(queries)
    print(f"Questions to process: {total}\n")

    schema_map = load_schema_map()

    # Write gold SQL
    with open(gold_path, "w", encoding="utf-8") as f:
        for q in queries:
            sql = q["SQL"].replace("\n", " ").strip()
            f.write(f"{sql}\t{q['db_id']}\n")

    # Group by DB to amortise index builds
    by_db = defaultdict(list)
    for idx, q in enumerate(queries):
        by_db[q["db_id"]].append((idx, q))

    results = ["SELECT 1"] * total
    saved_threshold = config.SQL_SCHEMA_THRESHOLD
    config.SQL_SCHEMA_THRESHOLD = None  # disable threshold for BIRD (small DBs)

    processed = 0
    for db_id, items in by_db.items():
        print(f"--- DB: {db_id} ({len(items)} questions) ---")

        config.SQLITE_PATH = str(DB_ROOT / db_id / f"{db_id}.sqlite")
        _db_module._sqlite_local.conn = None

        if args.ablation in ("tablerag_single_pass", "tablerag_react"):
            info = get_live_schema()
            if info.tables:
                build_schema_index(info)

        full_schema = schema_map.get(db_id, "")

        for orig_idx, entry in items:
            processed += 1
            question = entry["question"]
            evidence = entry.get("evidence", "") if args.use_evidence else ""
            prompt_q = f"{question}\nHint: {evidence}" if evidence else question
            diff     = entry.get("difficulty", "?")

            print(f"  [{processed}/{total}] ({diff}) {question[:75]}{'...' if len(question) > 75 else ''}")

            try:
                if args.ablation == "full_single_pass":
                    sql = run_single_pass_baseline(question, full_schema, evidence)

                elif args.ablation == "tablerag_single_pass":
                    schema_list = retrieve_relevant_schema(prompt_q, top_k=config.SQL_TOP_K)
                    schema_ctx  = "\n".join(schema_list) if schema_list else full_schema
                    result = run_sql_agent(query=prompt_q, schema_context=schema_ctx)
                    sql = clean_sql(result.get("sql", "SELECT 1"))

                elif args.ablation == "tablerag_react":
                    schema_list = retrieve_relevant_schema(prompt_q, top_k=config.SQL_TOP_K)
                    schema_ctx  = "\n".join(schema_list) if schema_list else full_schema
                    result = run_react_sql_agent(query=prompt_q, schema_context=schema_ctx)
                    raw    = result.get("sql")
                    sql    = clean_sql(raw) if raw else "SELECT 1"

            except Exception as e:
                print(f"    -> Error: {e}")
                sql = "SELECT 1"

            results[orig_idx] = sql

    config.SQL_SCHEMA_THRESHOLD = saved_threshold

    # Write predictions in BIRD format
    payload = [
        f"{sql}\t----- bird -----\t{queries[i]['db_id']}"
        for i, sql in enumerate(results)
    ]
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nPredictions saved : {predictions_path.name}")
    print(f"Gold SQL saved    : {gold_path.name}")
    return predictions_path, gold_path


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def run_evaluation(predictions_path: Path, gold_path: Path, stem: str):
    results_path = OUTPUT_DIR / f"results_bird_{stem}.txt"

    print(f"\n{'='*60}")
    print("  Running BIRD EX evaluation ...")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--predicted_sql_path", str(predictions_path),
        "--ground_truth_path",  str(gold_path),
        "--db_root_path",       str(DB_ROOT) + "/",
        "--diff_json_path",     str(DEV_JSON),
        "--num_cpus",           "1",
        "--meta_time_out",      "30.0",
        "--sql_dialect",        "SQLite",
        "--output_log_path",    str(results_path),
    ]

    subprocess.run(cmd, check=False)
    print(f"\nResults saved to: {results_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BIRD Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bird_eval.py --ablation tablerag_react --evaluate
  python bird_eval.py --ablation full_single_pass --limit 20 --evaluate
  python bird_eval.py --ablation tablerag_react --score_only
  python bird_eval.py --ablation tablerag_react --use_evidence --difficulty simple --evaluate
        """,
    )
    parser.add_argument(
        "--ablation",
        choices=["full_single_pass", "tablerag_single_pass", "tablerag_react"],
        required=True,
        help="Pipeline variant to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N questions (useful for quick tests).",
    )
    parser.add_argument(
        "--use_evidence",
        action="store_true",
        help="Include BIRD's evidence/hint field in each question prompt.",
    )
    parser.add_argument(
        "--difficulty",
        choices=["all", "simple", "moderate", "challenging"],
        default="all",
        help="Filter to a specific difficulty tier (default: all).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Override SQL_TOP_K for TableRAG retrieval.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Automatically score predictions after generating them.",
    )
    parser.add_argument(
        "--score_only",
        action="store_true",
        help="Skip prediction generation; score the existing predictions file.",
    )

    args = parser.parse_args()

    if args.top_k is not None:
        config.SQL_TOP_K = args.top_k

    stem             = make_stem(args.ablation, args.use_evidence, args.difficulty, args.limit)
    predictions_path = OUTPUT_DIR / f"predictions_bird_{stem}.json"
    gold_path        = OUTPUT_DIR / f"gold_bird_{stem}.sql"

    if args.score_only:
        if not predictions_path.exists():
            print(f"ERROR: No predictions file found at:\n  {predictions_path}")
            print("Run without --score_only first to generate predictions.")
            sys.exit(1)
        if not gold_path.exists():
            print(f"ERROR: No gold SQL file found at:\n  {gold_path}")
            print("Run without --score_only first to generate the gold file.")
            sys.exit(1)
        run_evaluation(predictions_path, gold_path, stem)
        return

    predictions_path, gold_path = generate_predictions(args, stem)

    if args.evaluate:
        run_evaluation(predictions_path, gold_path, stem)
    else:
        print(
            f"\nTo score, run:\n"
            f"  python bird_eval.py --ablation {args.ablation}"
            + (" --use_evidence" if args.use_evidence else "")
            + (f" --difficulty {args.difficulty}" if args.difficulty != "all" else "")
            + (f" --limit {args.limit}" if args.limit else "")
            + " --score_only"
        )


if __name__ == "__main__":
    main()
