import sys
import argparse
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
from evaluation_utils import (
    load_jsonl,
    execute_sql,
    package_sqls,
    sort_results,
    print_data,
)


def result_callback(result):
    exec_result.append(result)


def calculate_ex(predicted_res, ground_truth_res):
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def execute_model(
    predicted_sql, ground_truth, db_place, idx, meta_time_out, sql_dialect
):
    try:
        res = func_timeout(
            meta_time_out,
            execute_sql,
            args=(predicted_sql, ground_truth, db_place, sql_dialect, calculate_ex),
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f"timeout",)]
        res = 0
    except Exception as e:
        result = [(f"error",)]  # possibly len(query) > 512 or not executable
        res = 0
    result = {"sql_idx": idx, "res": res}
    return result


def run_sqls_parallel(
    sqls, db_places, num_cpus=1, meta_time_out=30.0, sql_dialect="SQLite"
):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):

        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(
                predicted_sql,
                ground_truth,
                db_places[i],
                i,
                meta_time_out,
                sql_dialect,
            ),
            callback=result_callback,
        )
    pool.close()
    pool.join()


def compute_acc_by_diff(exec_results, diff_json_path):
    num_queries = len(exec_results)
    results = [res["res"] for res in exec_results]
    
    try:
        contents = load_jsonl(diff_json_path)
        print(f"Successfully loaded diff file with {len(contents)} entries")
    except Exception as e:
        print(f"Error loading diff file: {e}")
        print("Falling back to simple accuracy calculation without difficulty breakdown")
        
        # Calculate overall accuracy without difficulty breakdown
        all_acc = sum(results) / num_queries if num_queries > 0 else 0
        count_lists = [0, 0, 0, num_queries]
        
        # Return placeholder values for difficulty breakdown
        return (all_acc * 100, all_acc * 100, all_acc * 100, all_acc * 100, count_lists)
    
    # Handle partial evaluation (when we're testing with limited questions)
    is_partial = num_queries < len(contents)
    if is_partial:
        print(f"TESTING MODE: Evaluating only {num_queries} of {len(contents)} questions")
        contents = contents[:num_queries]

    simple_results, moderate_results, challenging_results = [], [], []

    for i, content in enumerate(contents):
        if i >= len(exec_results):
            # This should not happen in normal cases, but let's be safe
            print(f"Warning: Not enough results ({len(exec_results)}) for dataset size ({len(contents)})")
            break
        
        # Add safeguard for missing difficulty field
        difficulty = content.get("difficulty", "moderate")  # Default to moderate if missing
            
        if difficulty == "simple":
            simple_results.append(exec_results[i])
        elif difficulty == "moderate":
            moderate_results.append(exec_results[i])
        elif difficulty == "challenging":
            challenging_results.append(exec_results[i])
        else:
            print(f"Warning: Unknown difficulty level '{difficulty}' for question {i}, treating as moderate")
            moderate_results.append(exec_results[i])

    # Guard against division by zero
    simple_acc = sum([res["res"] for res in simple_results]) / max(len(simple_results), 1) if simple_results else 0
    moderate_acc = sum([res["res"] for res in moderate_results]) / max(len(moderate_results), 1) if moderate_results else 0 
    challenging_acc = sum([res["res"] for res in challenging_results]) / max(len(challenging_results), 1) if challenging_results else 0
    all_acc = sum(results) / num_queries if num_queries > 0 else 0
    
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]
    return (
        simple_acc * 100,
        moderate_acc * 100,
        challenging_acc * 100,
        all_acc * 100,
        count_lists,
    )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--predicted_sql_path", type=str, required=True, default=""
    )
    args_parser.add_argument("--ground_truth_path", type=str, required=True, default="")
    args_parser.add_argument("--db_root_path", type=str, required=True, default="")
    args_parser.add_argument("--num_cpus", type=int, default=1)
    args_parser.add_argument("--meta_time_out", type=float, default=30.0)
    args_parser.add_argument("--diff_json_path", type=str, default="")
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args_parser.add_argument("--output_log_path", type=str, default="SQLite")
    args = args_parser.parse_args()
    exec_result = []

    pred_queries, db_paths = package_sqls(
        args.predicted_sql_path,
        args.db_root_path,
        mode='pred'
    )
    # generate ground truth sqls:
    gt_queries, db_paths_gt = package_sqls(
        args.ground_truth_path,
        args.db_root_path,
        mode="gt",
    )

    # Handle case where prediction file has fewer queries than ground truth
    if len(pred_queries) < len(gt_queries):
        print(f"WARNING: Prediction file contains only {len(pred_queries)} queries, but ground truth has {len(gt_queries)}")
        print(f"Will evaluate only the first {len(pred_queries)} queries")
        gt_queries = gt_queries[:len(pred_queries)]
        db_paths_gt = db_paths_gt[:len(pred_queries)]

    query_pairs = list(zip(pred_queries, gt_queries))

    run_sqls_parallel(
        query_pairs,
        db_places=db_paths_gt,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
        sql_dialect=args.sql_dialect,
    )
    exec_result = sort_results(exec_result)
    print("start calculate EX")
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(
        exec_result, args.diff_json_path
    )
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc] 
    print_data(score_lists, count_lists, metric="EX", result_log_file=args.output_log_path)
    print(
        "==========================================================================================="
    )
    print(f"Finished EX evaluation for {args.sql_dialect} on Mini Dev set")
    print("\n\n")
