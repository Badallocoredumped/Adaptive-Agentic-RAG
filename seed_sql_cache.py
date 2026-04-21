"""
Seed the FAISS SQL cache with hand-written few-shot question→SQL pairs,
then run test queries through run_table_rag_pipeline() to verify cache hits.

Usage:
    python seed_sql_cache.py                   # seed + run all test queries
    python seed_sql_cache.py --seed-only       # only seed, skip test queries
    python seed_sql_cache.py --test-only       # skip seeding, only run tests
    python seed_sql_cache.py --clear           # wipe cache before seeding
    python seed_sql_cache.py --build-schema    # (re)build TableRAG schema index
    python seed_sql_cache.py --shots fintech   # use fintech few-shots (default)
    python seed_sql_cache.py --shots generic   # use generic few-shots

    Recommended first-time setup:
        python seed_sql_cache.py --clear --build-schema

Edit your few-shot examples in:
    data/fintech_few_shots.json   ← fintech database (default)
    data/sql_few_shots.json       ← generic / old database
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Make sure the project root is on sys.path ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import config
from backend.sql.sql_cache import SQLCache
from backend.sql.sql_agent import run_table_rag_pipeline
from backend.sql.table_rag import get_schema_texts

# ── Few-shot file map ──────────────────────────────────────────────────────
FEW_SHOT_FILES = {
    "fintech": PROJECT_ROOT / "data" / "fintech_few_shots.json",
    "generic": PROJECT_ROOT / "data" / "sql_few_shots.json",
}

# ── Fintech test queries — probe cache hits / misses ──────────────────────
FINTECH_TEST_QUERIES = [
    # ── Expected CACHE HIT (near-identical to seeded questions) ───────────
    "What is the total balance of all active accounts?",                    # ~seed
    "How many accounts does each customer hold?",                           # ~seed
    "Show customers in Cairo who are active",                               # ~seed
    "Which branch has the highest number of customers?",                    # ~seed
    "What is the average salary for each employee role?",                   # ~seed
    "Total outstanding loan balance by loan type",                          # ~seed
    "Show defaulted loans with customer name and balance",                  # ~seed
    "Customers with more than one active loan",                             # ~seed
    "Average credit score per risk tier",                                   # ~seed
    "How many KYC records have each status?",                               # ~seed
    # ── Expected CACHE MISS (new questions, full pipeline runs) ───────────
    "What is the total interest collected from loan payments this year?",
    "Which asset had the biggest price drop last week?",
    "Show customers whose KYC expires within the next 30 days",
    "Which relationship manager's customers have the highest total loan balance?",
    "Show all loan payments made via Mobile App",
]

# ── Generic test queries (old database) ───────────────────────────────────
GENERIC_TEST_QUERIES = [
    "What is the total revenue from completed orders?",
    "How many orders has each customer placed?",
    "Show customers from Cairo",
    "Average salary by department",
    "Which customers have more than 2 completed orders?",
]


def _load_few_shots(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] Few-shots file not found: {path}")
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        shots = json.load(f)
    # Filter out comment-only entries
    shots = [s for s in shots if "question" in s and "sql" in s]
    print(f"[SEED] Loaded {len(shots)} few-shot pairs from {path.name}")
    return shots


def seed_cache(shots_path: Path, clear: bool = False) -> SQLCache:
    """Load few-shots from JSON and add them to the FAISS SQL cache."""
    cache = SQLCache()

    if clear:
        for path in [cache.index_path, cache.metadata_path]:
            if path.exists():
                path.unlink()
                print(f"[SEED] Deleted old cache file: {path.name}")

    loaded = cache.load_cache()
    if not loaded:
        print("[SEED] No existing cache found — starting fresh.")
        cache.initialize_cache()
    else:
        print(f"[SEED] Loaded existing cache ({cache.index.ntotal} entries).")

    shots = _load_few_shots(shots_path)

    print("\n── Seeding few-shot pairs ──────────────────────────────────────────")
    for i, shot in enumerate(shots, 1):
        question = shot["question"].strip()
        sql      = shot["sql"].strip()
        cache.add_to_cache(question=question, sql=sql, schema=None)
        print(f"  [{i:02d}] Q: {question}")
        print(f"       SQL: {sql[:85]}{'...' if len(sql) > 85 else ''}")

    cache.save_cache()
    print(f"\n[SEED] ✅  Cache saved — {cache.index.ntotal} total entries.\n")
    return cache


def build_schema() -> None:
    """(Re)build the TableRAG schema FAISS index from the live PostgreSQL database.

    Always wipes any existing index files first so the result reflects the
    current live schema rather than whatever was indexed previously.
    """
    from backend import config
    from backend.sql.database import get_live_schema
    from backend.sql.table_rag import build_schema_index

    index_path = config.INDEX_DIR / "schema.faiss"
    texts_path = config.INDEX_DIR / "schema_texts.json"

    # Always delete stale files — the whole point of calling this is a fresh build.
    for p in [index_path, texts_path]:
        if p.exists():
            p.unlink()
            print(f"[SCHEMA] Deleted old index file: {p.name}")

    print("[SCHEMA] Reading live schema from PostgreSQL ...")
    schema_info = get_live_schema()
    if not schema_info.tables:
        print("[SCHEMA] ❌  No tables found in the database. Is the DB seeded?")
        return

    print(f"[SCHEMA] Found {len(schema_info.tables)} tables: {', '.join(sorted(schema_info.tables.keys()))}")
    build_schema_index(schema_info)
    import json
    with texts_path.open("r", encoding="utf-8") as f:
        texts = json.load(f)
    print(f"[SCHEMA] ✅  Schema index built — {len(texts)} entries indexed.")
    for t in texts:
        print(f"          {t}")
    print()


def run_tests(queries: list[str]) -> None:
    """Run queries through the full pipeline and show cache hit/miss."""
    print("── Running test queries ────────────────────────────────────────────\n")

    hits   = 0
    misses = 0

    for i, query in enumerate(queries, 1):
        print(f"[{i:02d}] {query!r}")
        print(f"{'─' * 70}")
        result = run_table_rag_pipeline(query)

        path  = result.get("path", "?").upper()
        sql   = result.get("sql") or "n/a"
        rows  = result.get("result", [])
        error = result.get("error")
        lat   = result.get("latency", 0.0)

        is_hit = path == "FAST"
        label  = "⚡ CACHE HIT" if is_hit else "🤖 FULL PIPELINE"
        if is_hit:
            hits += 1
        else:
            misses += 1

        print(f"  Path    : {path}  {label}")
        print(f"  SQL     : {sql[:90]}{'...' if len(sql) > 90 else ''}")
        print(f"  Latency : {lat:.2f}s")
        if error:
            print(f"  ❌ Error : {error}")
        else:
            print(f"  ✅ Rows  : {len(rows)} returned")
            for row in rows[:3]:
                print(f"           {row}")
            if len(rows) > 3:
                print(f"           ... and {len(rows) - 3} more")
        print()

    print(f"── Summary ─────────────────────────────────────────────────────────")
    print(f"  Total queries : {len(queries)}")
    print(f"  Cache hits    : {hits}  ⚡")
    print(f"  Full pipeline : {misses}  🤖")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed SQL cache and run test queries.")
    parser.add_argument(
        "--shots", choices=["fintech", "generic"], default="fintech",
        help="Which few-shot file to use (default: fintech)"
    )
    parser.add_argument("--seed-only",     action="store_true", help="Only seed, skip tests")
    parser.add_argument("--test-only",     action="store_true", help="Only test, skip seeding")
    parser.add_argument("--clear",         action="store_true", help="Clear SQL cache before seeding")
    parser.add_argument("--build-schema",  action="store_true", help="(Re)build TableRAG schema index from live DB")
    args = parser.parse_args()

    shots_path   = FEW_SHOT_FILES[args.shots]
    test_queries = FINTECH_TEST_QUERIES if args.shots == "fintech" else GENERIC_TEST_QUERIES

    print(f"[CONFIG] DB           : {config.SQLITE_PATH or config.DATABASE_URL or 'PostgreSQL'}")
    print(f"[CONFIG] Few-shots    : {shots_path.name}")
    print(f"[CONFIG] Router mode  : {config.ROUTER_MODE}")
    print(f"[CONFIG] Build schema : {args.build_schema}\n")

    if args.build_schema:
        build_schema()

    if not args.test_only:
        seed_cache(shots_path, clear=args.clear)

    if not args.seed_only:
        run_tests(test_queries)


if __name__ == "__main__":
    main()
