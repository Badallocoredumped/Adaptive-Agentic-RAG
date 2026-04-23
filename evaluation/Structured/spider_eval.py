import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict



# Ensure the backend module can be imported
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from backend import config
from backend.sql.react_agent import run_react_sql_agent
from backend.sql.database import get_live_schema
from backend.sql.table_rag import build_schema_index, retrieve_relevant_schema
from backend.sql.sql_agent import run_sql_agent
from backend.sql import database as _db_module

# Import LangChain tools for the single-pass baselines
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

SPIDER_DIR = Path(__file__).resolve().parent
DEV_JSON_PATH = SPIDER_DIR / "dev.json"
TABLES_JSON_PATH = SPIDER_DIR / "tables.json"

def clean_sql(raw_sql: str) -> str:
    """Strips markdown fences, removes all newlines, and isolates a single query."""
    cleaned = raw_sql.strip()
    
    # 1. Remove Markdown fences
    if cleaned.startswith("```"):
        # split safely and take the content between fences
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
        
    # 2. Destroy ALL newlines and carriage returns (The cause of Error A)
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    
    # 3. Force a single query (The cause of Error B)
    if ";" in cleaned:
        cleaned = cleaned.split(";")[0]
        
    cleaned = cleaned.strip()
    
    # 4. Safe fallback
    return cleaned if cleaned else "SELECT 1"

def run_single_pass_baseline(query: str, schema_context: str) -> str:
    """A naive, single-turn LLM generation without ReAct or tool calling."""
    llm = ChatOpenAI(model=config.SQL_OPENAI_MODEL, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert SQLite developer. Given the following database schema, "
         "write a valid SQLite query to answer the user's question. "
         "STRICT RULE: Return ONLY the raw SQL query. No markdown, no code fences, no explanations.\n\n"
         "Schema:\n{schema_context}"),
        ("human", "{query}")
    ])
    chain = prompt | llm
    response = chain.invoke({"schema_context": schema_context, "query": query})
    return clean_sql(str(response.content))

def load_schema_map() -> dict:
    """Parses tables.json into a dictionary mapping db_id to its full schema."""
    with open(TABLES_JSON_PATH, "r", encoding="utf-8") as f:
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

def main():
    parser = argparse.ArgumentParser(description="Spider Benchmark Harness")
    parser.add_argument(
        "--ablation", 
        type=str, 
        choices=["full_single_pass", "tablerag_single_pass", "tablerag_react"],
        required=True,
        help="Choose which version of the system to benchmark."
    )
    parser.add_argument(
        "--limit", 
        type=int,
        default=None,
        help="Limit the number of queries to process (e.g., 10, 20, 50)."
    )
    args = parser.parse_args()
    
    # Append the limit to the filename if one was provided
    file_suffix = f"{args.ablation}_limit{args.limit}" if args.limit else args.ablation
    output_filename = SPIDER_DIR / f"predictions_{file_suffix}.txt"
    
    print(f"Starting benchmark mode: {args.ablation.upper()}")

    # 1. Load data
    with open(DEV_JSON_PATH, "r", encoding="utf-8") as f:
        dev_queries = json.load(f)
        
    if args.limit:
        print(f"\n⚠️  LIMIT ENABLED: Limiting dataset to the first {args.limit} queries. ⚠️\n")
        dev_queries = dev_queries[:args.limit]
        
    schema_map = load_schema_map()

    # 2. Group queries by db_id for efficiency
    queries_by_db = defaultdict(list)
    for idx, entry in enumerate(dev_queries):
        # Store original index to maintain output order required by the official evaluator
        queries_by_db[entry["db_id"]].append((idx, entry["question"]))

    results_array = ["SELECT 1;"] * len(dev_queries)

    # 3. Process database by database
    total_processed = 0
    for db_id, questions in queries_by_db.items():
        print(f"\n--- Processing DB: {db_id} ({len(questions)} questions) ---")
        
        # Override system SQLite path for this specific database
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        config.SQLITE_PATH = str(db_path)
        # Reset the cached thread-local connection so execute_query opens the new DB
        _db_module._sqlite_local.conn = None
        full_schema_context = schema_map.get(db_id, "")

        # For tablerag ablations, build a fresh FAISS schema index from this DB
        if args.ablation in ("tablerag_single_pass", "tablerag_react"):
            schema_info = get_live_schema()
            if schema_info.tables:
                build_schema_index(schema_info)

        for original_idx, question in questions:
            total_processed += 1
            print(f"[{total_processed}/{len(dev_queries)}] Q: {question}")
            
            try:
                if args.ablation == "full_single_pass":
                    # Baseline: No pruning, no agent
                    final_sql = run_single_pass_baseline(question, full_schema_context)
                    
                elif args.ablation == "tablerag_single_pass":
                    # Ablation 1: TableRAG pruning + single-pass SQL agent (no ReAct loop).
                    pruned_schema_list = retrieve_relevant_schema(question, top_k=config.SQL_TOP_K)
                    pruned_schema = "\n".join(pruned_schema_list) if pruned_schema_list else full_schema_context
                    result = run_sql_agent(query=question, schema_context=pruned_schema)
                    generated_sql = result.get("sql")
                    final_sql = clean_sql(generated_sql)
                    
                elif args.ablation == "tablerag_react":
                    # Ablation 2: Pruned schema + LangGraph ReAct agent
                    pruned_schema_list = retrieve_relevant_schema(question, top_k=config.SQL_TOP_K)
                    pruned_schema = "\n".join(pruned_schema_list) if pruned_schema_list else full_schema_context
                    
                    result = run_react_sql_agent(query=question, schema_context=pruned_schema)
                    generated_sql = result.get("sql")
                    final_sql = clean_sql(generated_sql) if generated_sql else "SELECT 1;"

            except Exception as e:
                print(f"  -> Error: {e}")
                final_sql = "SELECT 1;"

            results_array[original_idx] = final_sql

    # 4. Write aligned predictions
    with open(output_filename, "w", encoding="utf-8") as out_file:
        for sql in results_array:
            out_file.write(f"{sql}\n")
            
    print(f"\nFinished! Evaluated {total_processed} queries.")
    print(f"Predictions saved to {output_filename.name}")

if __name__ == "__main__":
    main()