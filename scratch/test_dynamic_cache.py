import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.sql.sql_agent import run_table_rag_pipeline

def test_cache_refiner():
    print("=" * 60)
    print("TESTING DYNAMIC FEW-SHOT CACHE HIT")
    print("=" * 60)
    
    # 1. First, we send a query that is VERY similar to a seed query.
    # Seed query in fintech_few_shots.json: "What is the total balance across all active accounts?"
    # We will ask for "frozen" accounts instead to force the LLM to refine the cached SQL.
    test_query = "What is the total balance across all frozen accounts?"
    
    print(f"\nUser Question: {test_query}\n")
    print("Watch the logs below carefully:")
    print("1. It should say '⚡ CACHE HIT'")
    print("2. It should run the LLM SQL refiner.")
    print("3. It should adjust the SQL to look for 'frozen' instead of 'active' accounts.\n")
    
    # Run the pipeline
    result = run_table_rag_pipeline(test_query)
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Path Taken : {result['path']}")
    print(f"Final SQL  : {result['sql']}")
    print(f"Rows found : {len(result['result'])}")
    if result['result']:
        print(f"Top Row    : {result['result'][0]}")

if __name__ == "__main__":
    test_cache_refiner()
