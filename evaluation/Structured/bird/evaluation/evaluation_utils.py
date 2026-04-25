import json
import psycopg2
import pymysql
import sqlite3
import os

def load_jsonl(file_path):
    """
    Load a JSON or JSONL file robustly across different platforms.
    Handles both regular JSON and line-by-line JSONL formats.
    Also handles different line endings (CRLF vs LF) and encodings.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # First try to load as a regular JSON file with different encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return json.load(file)
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError:
            # If JSON parsing fails, it could be JSONL or malformed
            break
    
    # If regular JSON loading failed, try line-by-line JSONL approach
    data = []
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                for line in file:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            # Skip malformed lines
                            continue
            if data:  # If we got any data, return it
                return data
        except UnicodeDecodeError:
            continue
    
    # As a last resort, try to read the entire file as one JSON string
    # with manual preprocessing
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                content = file.read()
                # Remove any BOM characters and normalize line endings
                content = content.replace('\r\n', '\n').strip()
                if content:
                    return json.loads(content)
        except Exception:
            continue
    
    # If all attempts failed
    raise ValueError(f"Failed to load JSON from {file_path}. File may be corrupted or in an unsupported format.")

def load_json(dir):
    """Legacy function, now uses the more robust load_jsonl function"""
    return load_jsonl(dir)


# psycopg2   2.9.9
def connect_postgresql():
    # Open database connection
    # Connect to the database
    db = psycopg2.connect(
         "dbname=**** user=**** host=*** password="" port=5432"
    )
    return db


# PyMySQL  1.1.1
def connect_mysql():
    # Open database connection
    # Connect to the database"
    try:
        # First try with unix_socket at /var/run/mysqld/mysqld.sock (Linux)
        db = pymysql.connect(
            host="localhost",
            user="root",
            password="li123911",
            database="BIRD",
            unix_socket="/var/run/mysqld/mysqld.sock"
        )
        return db
    except pymysql.err.OperationalError:
        try:
            # Then try with unix_socket at /tmp/mysql.sock (macOS)
            db = pymysql.connect(
                host="localhost",
                user="root",
                password="li123911",
                database="BIRD",
                unix_socket="/tmp/mysql.sock"
            )
            return db
        except pymysql.err.OperationalError:
            # Finally, try without unix_socket (Windows or other configs)
            db = pymysql.connect(
                host="localhost",
                user="root",
                password="li123911",
                database="BIRD",
                port=3306,
            )
            return db


def connect_db(sql_dialect, db_path):
    if sql_dialect == "SQLite":
        conn = sqlite3.connect(db_path)
    elif sql_dialect == "MySQL":
        conn = connect_mysql()
    elif sql_dialect == "PostgreSQL":
        conn = connect_postgresql()
    else:
        raise ValueError("Unsupported SQL dialect")
    return conn


def execute_sql(predicted_sql, ground_truth, db_path, sql_dialect, calculate_func):
    conn = connect_db(sql_dialect, db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    conn.close()
    res = calculate_func(predicted_res, ground_truth_res)
    return res


def package_sqls(
    sql_path, db_root_path, mode="pred"
):
    clean_sqls = []
    db_path_list = []
    if mode == "pred":
        # use chain of thought
        try:
            sql_data = load_jsonl(sql_path)
            
            # Handle both dictionary and list formats
            if isinstance(sql_data, dict):
                sql_items = sql_data.items()
            elif isinstance(sql_data, list):
                sql_items = enumerate(sql_data)
            else:
                raise ValueError(f"Unexpected JSON format in {sql_path}")
                
            for _, sql_str in sql_items:
                if isinstance(sql_str, str):
                    try:
                        sql, db_name = sql_str.split("\t----- bird -----\t")
                    except ValueError:
                        sql = sql_str.strip()
                        db_name = "financial"  # Default database when no db_name is provided
                elif isinstance(sql_str, dict) and "sql" in sql_str:
                    # Handle case where predictions are saved as objects
                    sql = sql_str["sql"]
                    db_name = sql_str.get("db_id", "financial")
                else:
                    # Default for empty or malformed entries
                    sql = " "
                    db_name = "financial"               
                
                # Clean SQL strings that might have escaped quotes or encodings
                sql = sql.strip()
                
                # Handle empty results with a placeholder
                if not sql or len(sql) < 2:
                    sql = "SELECT 'empty' AS result"
                
                clean_sqls.append(sql)
                db_path_list.append(db_root_path + db_name + "/" + db_name + ".sqlite")
                
            print(f"Successfully processed {len(clean_sqls)} prediction entries")
        except Exception as e:
            print(f"Error loading prediction file: {e}")
            raise

    elif mode == "gt":
        try:
            sqls = open(sql_path)
            sql_txt = sqls.readlines()
            for idx, sql_str in enumerate(sql_txt):
                try:
                    sql, db_name = sql_str.strip().split("\t")
                    clean_sqls.append(sql)
                    db_path_list.append(db_root_path + db_name + "/" + db_name + ".sqlite")
                except ValueError:
                    print(f"Warning: Malformed ground truth line {idx}: {sql_str}")
                    continue
            print(f"Successfully processed {len(clean_sqls)} ground truth entries")
        except Exception as e:
            print(f"Error loading ground truth file: {e}")
            raise

    return clean_sqls, db_path_list


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x["sql_idx"])


def print_data(score_lists, count_lists, metric="F1 Score",result_log_file=None):
    levels = ["simple", "moderate", "challenging", "total"]
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format("count", *count_lists))

    print(
        f"======================================    {metric}    ====================================="
    )
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format(metric, *score_lists))
    
     # Log to file in append mode
    if result_log_file is not None:
        with open(result_log_file, "a") as log_file:
            log_file.write(f"start calculate {metric}\n")
            log_file.write("{:20} {:20} {:20} {:20} {:20}\n".format("", *levels))
            log_file.write(
                "{:20} {:<20} {:<20} {:<20} {:<20}\n".format("count", *count_lists)
            )
            log_file.write(
                f"======================================    {metric}   =====================================\n"
            )
            log_file.write(
                "{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}\n".format(
                    metric, *score_lists
                )
            )
            log_file.write(
                "===========================================================================================\n"
            )
            log_file.write(f"Finished {metric} evaluation for {metric} on Mini Dev set\n")
            log_file.write("\n")
