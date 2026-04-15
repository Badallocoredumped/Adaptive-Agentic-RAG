import sqlite3
import pandas as pd
import json
import os
from datasets import load_dataset

os.makedirs("data/index", exist_ok=True)
DB_PATH = "data/hybridqa_eval.db"
SCHEMA_OUTPUT_PATH = "data/index/hybridqa_schemas.json"

print("Loading HybridQA from Colab Cache...")
dataset = load_dataset("hybrid_qa", split="train", trust_remote_code=True)

conn = sqlite3.connect(DB_PATH)
valid_tables = set()
schemas = []

print("Filtering tables and building database...")
for item in dataset:
    table_id = item["table_id"]
    if table_id in valid_tables:
        continue
        
    table_dict = item["table"]
    headers = table_dict.get("header", [])
    
    # FIX: Hugging Face uses 'data' instead of 'rows'
    rows = table_dict.get("data", [])
    
    # TableRAG Paper Rule: Only tables with > 100 cells
    if len(headers) * len(rows) <= 100:
        continue
        
    valid_tables.add(table_id)
    df = pd.DataFrame(rows, columns=headers)
    df.columns = [str(c).replace(' ', '_').replace('-', '_').replace('"', '').lower() for c in df.columns]
    
    try:
        df.to_sql(table_id, conn, if_exists='replace', index=False)
    except Exception:
        continue
    
    columns_schema = []
    for col in df.columns:
        example_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
        columns_schema.append([col, str(df[col].dtype), str(example_val)])
        
    schemas.append({"table_name": table_id, "columns": columns_schema})

conn.close()

with open(SCHEMA_OUTPUT_PATH, 'w') as f:
    json.dump(schemas, f, indent=2)

print("✅ Finished processing!")

# 3. Download the files to your local Windows machine
from google.colab import files
files.download(DB_PATH)
files.download(SCHEMA_OUTPUT_PATH)