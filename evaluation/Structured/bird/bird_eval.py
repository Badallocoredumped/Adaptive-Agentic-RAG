"""
BIRD Benchmark Harness — mirrors spider_eval.py structure.

Ablation modes:
  full_single_pass     — No TableRAG, no ReAct. Full schema passed as plain text.
  tablerag_single_pass — TableRAG schema pruning + single-pass SQL agent (no ReAct).
  tablerag_react       — TableRAG schema pruning + ReAct loop (multi-turn agent).

BIRD-specific options:
  --use_evidence       — Append BIRD's evidence/hint field to each question prompt.
  --difficulty         — Run only questions of a given difficulty level.

Output:
  predictions_bird_<stem>.json  — BIRD evaluator-compatible predictions (JSON array).
  gold_bird_<stem>.sql          — Matching ground-truth SQL file (tab-separated).

Score with BIRD's official evaluator:
  python bird/evaluation/evaluation_ex.py \\
      --predicted_sql_path predictions_bird_<stem>.json \\
      --ground_truth_path gold_bird_<stem>.sql \\
      --db_root_path bird/dev_databases/dev_databases/ \\
      --diff_json_path bird/dev.json \\
      --num_cpus 1 --meta_time_out 30.0 --sql_dialect SQLite \\
      --output_log_path results_bird_<stem>.txt
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Ensure backend is importable from anywhere within the repo.
# Walk up from this file until we find the directory that contains backend/.
def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    while current != current.parent:
        if (current / "backend").is_dir():
            return current
        current = current.parent
    raise RuntimeError(
        "Could not find project root (expected a parent directory containing 'backend/')."
    )

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

BIRD_DIR = project_root / "evaluation" / "Structured" / "bird"
DEV_JSON_PATH = BIRD_DIR / "dev.json"
DEV_TABLES_PATH = BIRD_DIR / "dev_tables.json"
DB_ROOT_PATH = BIRD_DIR / "dev_databases" / "dev_databases"
OUTPUT_DIR = project_root / "evaluation" / "Structured"


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def clean_sql(raw_sql: str) -> str:
    """Strip markdown fences, collapse newlines, and isolate a single statement."""
    cleaned = raw_sql.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    if ";" in cleaned:
        cleaned = cleaned.split(";")[0]
    cleaned = cleaned.strip()
    return cleaned if cleaned else "SELECT 1"


# ---------------------------------------------------------------------------
# Ablation implementations
# ---------------------------------------------------------------------------

def run_single_pass_baseline(question: str, schema_context: str, evidence: str = "") -> str:
    """Naive single-turn LLM generation: no TableRAG, no ReAct."""
    llm = ChatOpenAI(model=config.SQL_OPENAI_MODEL, temperature=0.0)
    evidence_section = f"\nEvidence/Hints:\n{evidence}" if evidence else ""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert SQLite developer. Given the following database schema, "
         "write a valid SQLite query to answer the user's question. "
         "STRICT RULE: Return ONLY the raw SQL query. No markdown, no code fences, no explanations.\n\n"
         "Schema:\n{schema_context}{evidence_section}"),
        ("human", "{query}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "schema_context": schema_context,
        "evidence_section": evidence_section,
        "query": question,
    })
    return clean_sql(str(response.content))


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

def load_schema_map() -> dict:
    """Parse dev_tables.json into {db_id: schema_string}."""
    with open(DEV_TABLES_PATH, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    schema_map = {}
    for db in tables_data:
        db_id = db["db_id"]
        tables = db["table_names_original"]
        cols = db["column_names_original"]

        schema_lines = []
        for i, table_name in enumerate(tables):
            table_cols = [col[1] for col in cols if col[0] == i]
            schema_lines.append(f"Table: {table_name} | Columns: {', '.join(table_cols)}")

        schema_map[db_id] = "\n".join(schema_lines)

    return schema_map


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BIRD Benchmark Harness — mirrors spider_eval.py"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["full_single_pass", "tablerag_single_pass", "tablerag_react"],
        required=True,
        help="Pipeline variant to benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of questions processed (e.g. 20 for a quick smoke test).",
    )
    parser.add_argument(
        "--use_evidence",
        action="store_true",
        default=False,
        help="Append BIRD's evidence/hint field to each question prompt.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["all", "simple", "moderate", "challenging"],
        default="all",
        help="Restrict evaluation to one BIRD difficulty tier.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Override config.SQL_TOP_K for TableRAG schema retrieval.",
    )
    args = parser.parse_args()

    # Override top_k if provided
    if args.top_k is not None:
        config.SQL_TOP_K = args.top_k

    # Build a descriptive stem for output filenames
    parts = [args.ablation]
    if args.use_evidence:
        parts.append("evidence")
    if args.difficulty != "all":
        parts.append(args.difficulty)
    if args.limit:
        parts.append(f"limit{args.limit}")

    file_stem = "_".join(parts)
    predictions_path = OUTPUT_DIR / f"predictions_bird_{file_stem}.json"
    gold_sql_path = OUTPUT_DIR / f"gold_bird_{file_stem}.sql"

    # -----------------------------------------------------------------------
    print(f"Starting BIRD benchmark: {args.ablation.upper()}")
    print(f"  Evidence hints : {'ON' if args.use_evidence else 'OFF'}")
    print(f"  Difficulty     : {args.difficulty}")
    print(f"  SQL_TOP_K      : {config.SQL_TOP_K}")
    if args.limit:
        print(f"\n  ⚠️  LIMIT: first {args.limit} questions only\n")

    # 1. Load questions
    with open(DEV_JSON_PATH, "r", encoding="utf-8") as f:
        dev_queries = json.load(f)

    if args.difficulty != "all":
        dev_queries = [q for q in dev_queries if q.get("difficulty") == args.difficulty]
        print(f"After difficulty filter: {len(dev_queries)} questions")

    if args.limit:
        dev_queries = dev_queries[: args.limit]

    total = len(dev_queries)
    print(f"Total questions to process: {total}")

    # 2. Load schema map
    schema_map = load_schema_map()

    # 3. Write gold SQL file for this evaluation run
    with open(gold_sql_path, "w", encoding="utf-8") as f:
        for entry in dev_queries:
            sql = entry["SQL"].replace("\n", " ").strip()
            f.write(f"{sql}\t{entry['db_id']}\n")
    print(f"Gold SQL written to: {gold_sql_path.name}")

    # 4. Group by db_id to amortise schema index builds
    queries_by_db: dict[str, list] = defaultdict(list)
    for idx, entry in enumerate(dev_queries):
        queries_by_db[entry["db_id"]].append((idx, entry))

    results_array = ["SELECT 1"] * total

    # Disable cosine threshold during BIRD eval — same rationale as Spider:
    # the threshold was tuned for large production DBs; BIRD DBs are self-contained
    # and all tables may be relevant for JOINs.
    _original_threshold = config.SQL_SCHEMA_THRESHOLD
    config.SQL_SCHEMA_THRESHOLD = None

    # 5. Process database by database
    total_processed = 0
    for db_id, items in queries_by_db.items():
        print(f"\n--- DB: {db_id} ({len(items)} questions) ---")

        db_path = DB_ROOT_PATH / db_id / f"{db_id}.sqlite"
        config.SQLITE_PATH = str(db_path)
        _db_module._sqlite_local.conn = None  # drop cached connection

        full_schema_context = schema_map.get(db_id, "")

        if args.ablation in ("tablerag_single_pass", "tablerag_react"):
            schema_info = get_live_schema()
            if schema_info.tables:
                build_schema_index(schema_info)

        for original_idx, entry in items:
            total_processed += 1
            question = entry["question"]
            evidence = entry.get("evidence", "") if args.use_evidence else ""
            difficulty = entry.get("difficulty", "?")

            # Build the prompt question; evidence is appended as a hint when enabled
            prompt_question = f"{question}\nHint: {evidence}" if evidence else question

            print(
                f"[{total_processed}/{total}] ({difficulty}) "
                f"{question[:80]}{'...' if len(question) > 80 else ''}"
            )

            try:
                if args.ablation == "full_single_pass":
                    final_sql = run_single_pass_baseline(
                        question, full_schema_context, evidence
                    )

                elif args.ablation == "tablerag_single_pass":
                    pruned_schema_list = retrieve_relevant_schema(
                        prompt_question, top_k=config.SQL_TOP_K
                    )
                    pruned_schema = (
                        "\n".join(pruned_schema_list)
                        if pruned_schema_list
                        else full_schema_context
                    )
                    result = run_sql_agent(
                        query=prompt_question, schema_context=pruned_schema
                    )
                    final_sql = clean_sql(result.get("sql", "SELECT 1"))

                elif args.ablation == "tablerag_react":
                    pruned_schema_list = retrieve_relevant_schema(
                        prompt_question, top_k=config.SQL_TOP_K
                    )
                    pruned_schema = (
                        "\n".join(pruned_schema_list)
                        if pruned_schema_list
                        else full_schema_context
                    )
                    result = run_react_sql_agent(
                        query=prompt_question, schema_context=pruned_schema
                    )
                    generated_sql = result.get("sql")
                    final_sql = clean_sql(generated_sql) if generated_sql else "SELECT 1"

            except Exception as exc:
                print(f"  -> Error: {exc}")
                final_sql = "SELECT 1"

            results_array[original_idx] = final_sql

    config.SQL_SCHEMA_THRESHOLD = _original_threshold

    # 6. Write predictions in BIRD evaluator format (JSON array of tagged strings)
    predictions = [
        f"{sql}\t----- bird -----\t{dev_queries[i]['db_id']}"
        for i, sql in enumerate(results_array)
    ]
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # 7. Print summary and ready-to-run evaluation command
    print(f"\nFinished! Processed {total_processed} questions.")
    print(f"  Predictions : {predictions_path.name}")
    print(f"  Gold SQL    : {gold_sql_path.name}")
    print(
        f"\nRun EX evaluation:\n"
        f"  python bird/evaluation/evaluation_ex.py \\\n"
        f"      --predicted_sql_path {predictions_path} \\\n"
        f"      --ground_truth_path {gold_sql_path} \\\n"
        f"      --db_root_path {DB_ROOT_PATH}/ \\\n"
        f"      --diff_json_path {DEV_JSON_PATH} \\\n"
        f"      --num_cpus 1 --meta_time_out 30.0 --sql_dialect SQLite \\\n"
        f"      --output_log_path {OUTPUT_DIR / f'results_bird_{file_stem}.txt'}"
    )


if __name__ == "__main__":
    main()
