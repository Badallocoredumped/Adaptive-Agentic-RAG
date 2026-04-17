"""
evaluation/Unstructured/run_t_series.py
Run all T-series configs sequentially, storing separate result files for each.

Usage:
    python evaluation/Unstructured/run_t_series.py
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER     = SCRIPT_DIR / "run_config_eval.py"

T_CONFIGS = ["T2A", "T2B", "T3A", "T3B", "T4A", "T4B", "T5A", "T5B"]

summary = []

for cfg_name in T_CONFIGS:
    print(f"\n{'#'*62}")
    print(f"  Starting {cfg_name}  ({T_CONFIGS.index(cfg_name)+1}/{len(T_CONFIGS)})")
    print(f"{'#'*62}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(RUNNER), cfg_name],
        check=False,
    )
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    summary.append((cfg_name, status, elapsed))
    print(f"\n[{cfg_name}] {status}  —  {elapsed/60:.1f} min")

print(f"\n{'='*62}")
print("  T-SERIES RUN COMPLETE")
print(f"{'='*62}")
for name, status, elapsed in summary:
    print(f"  {name:<6}  {status:<25}  {elapsed/60:.1f} min")
print(f"{'='*62}\n")
