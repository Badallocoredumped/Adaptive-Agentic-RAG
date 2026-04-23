#!/bin/bash

# DO NOT CHANGE THIS
db_root_path='../llm/dev_data/dev_20240627/dev_databases/'
num_cpus=16
meta_time_out=30.0
# DO NOT CHANGE THIS

# Get the engine name from run_gpt.sh
engine_line=$(grep "engine=" ../llm/run/run_gpt.sh)
engine_name=$(echo "$engine_line" | cut -d'=' -f2 | tr -d "'" | tr -d '"' | awk '{print $1}')  # Remove comments

# Choose the SQL dialect from run_gpt.sh or use default
sql_dialect_line=$(grep "sql_dialect=" ../llm/run/run_gpt.sh)
sql_dialect=$(echo "$sql_dialect_line" | cut -d'=' -f2 | tr -d "'" | tr -d '"')

# Get chain of thought setting
cot_line=$(grep "cot=" ../llm/run/run_gpt.sh)
cot=$(echo "$cot_line" | cut -d'=' -f2 | tr -d "'" | tr -d '"')

echo "Looking for prediction file from model: $engine_name with dialect: $sql_dialect"

# Determine correct file pattern based on chain of thought setting
cot_suffix=""
if [ "$cot" == "True" ]; then
  cot_suffix="_cot"
fi

# Determine the script directory and project root for better path handling
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd )"

# Use the correct path format with proper case for SQL dialect
predicted_sql_path='/Users/apple/developement/bird_bench_shiva/bird-bench/llm/run/exp_result/sql_output_kg/predict_dev_inf-2-0-32b-sql_cot_PostgreSQL.json'

# Check if file exists, if not try alternative paths
if [ ! -f "$predicted_sql_path" ]; then
  echo "File not found at $predicted_sql_path, trying alternative paths..."
  
  # Try to find the file using find command
  found_files=$(find "$PROJECT_ROOT" -name "predict_mini_dev_${engine_name}${cot_suffix}_${sql_dialect}*.json" -print -quit)
  
  if [ -n "$found_files" ]; then
    predicted_sql_path="$found_files"
    echo "Found alternative file at: $predicted_sql_path"
  fi
fi

# Convert SQL dialect to proper case for display
echo "Using prediction file: $predicted_sql_path"
echo "Using SQL dialect: $sql_dialect"

# DO NOT CHANGE THIS
# Extract the base filename without extension
base_name=$(basename "$predicted_sql_path" .json)
# Define the output log path
output_log_path="$PROJECT_ROOT/eval_result/${base_name}.txt"
mkdir -p "$PROJECT_ROOT/eval_result"

case $sql_dialect in
  "SQLite")
    diff_json_path="../sqlite/mini_dev_sqlite.jsonl"
    ground_truth_path="../sqlite/mini_dev_sqlite_gold.sql"
    ;;
  "PostgreSQL")
    diff_json_path="/Users/apple/developement/bird_bench_shiva/bird-bench/llm/dev_data/financial_5/dev_financial.json"
    ground_truth_path="/Users/apple/developement/bird_bench_shiva/bird-bench/llm/dev_data/financial_5/dev_financial.sql"
    ;;
  "MySQL")
    diff_json_path="../mysql/mini_dev_mysql.jsonl"
    ground_truth_path="../mysql/mini_dev_mysql_gold.sql"
    ;;
  *)
    echo "Invalid SQL dialect: $sql_dialect"
    exit 1
    ;;
esac
# DO NOT CHANGE THIS

# Output the set paths
echo "Differential JSON Path: $diff_json_path"
echo "Ground Truth Path: $ground_truth_path"
echo "Predicted SQL Path: $predicted_sql_path"

# Check if prediction file exists
if [ ! -f "$predicted_sql_path" ]; then
  echo "ERROR: Predicted SQL file not found at: $predicted_sql_path"
  echo "Please check if the file exists and the path is correct"
  echo "Available prediction files:"
  find "$PROJECT_ROOT" -name "predict_mini_dev_*.json" | sort
  exit 1
fi

# Check file size
file_size=$(wc -c < "$predicted_sql_path")
if [ "$file_size" -eq 0 ]; then
  echo "ERROR: The prediction file exists but is empty"
  exit 1
fi

# Validate JSON format before proceeding
echo "Validating JSON file format..."
if ! python3 -c "import json; json.load(open('$predicted_sql_path'));" 2>/dev/null; then
  echo "WARNING: The prediction file exists but has JSON format issues."
  echo "Attempting to read with more robust methods in the evaluation script."
fi

echo "Starting to compare with knowledge for ex, sql_dialect: ${sql_dialect}"
python3 -u "$SCRIPT_DIR/evaluation_ex.py" --db_root_path "${db_root_path}" --predicted_sql_path "${predicted_sql_path}" \
--ground_truth_path "${ground_truth_path}" --num_cpus ${num_cpus} --output_log_path "${output_log_path}" \
--diff_json_path "${diff_json_path}" --meta_time_out ${meta_time_out} --sql_dialect "${sql_dialect}"


# echo "Starting to compare with knowledge for R-VES, sql_dialect: ${sql_dialect}"
# python3 -u "$SCRIPT_DIR/evaluation_ves.py" --db_root_path "${db_root_path}" --predicted_sql_path "${predicted_sql_path}" \
# --ground_truth_path "${ground_truth_path}" --num_cpus ${num_cpus} --output_log_path "${output_log_path}" \
# --diff_json_path "${diff_json_path}" --meta_time_out ${meta_time_out} --sql_dialect "${sql_dialect}"


# echo "Starting to compare with knowledge for soft-f1, sql_dialect: ${sql_dialect}"
# python3 -u "$SCRIPT_DIR/evaluation_f1.py" --db_root_path "${db_root_path}" --predicted_sql_path "${predicted_sql_path}" \
# --ground_truth_path "${ground_truth_path}" --num_cpus ${num_cpus} --output_log_path "${output_log_path}" \
# --diff_json_path "${diff_json_path}" --meta_time_out ${meta_time_out} --sql_dialect "${sql_dialect}"
