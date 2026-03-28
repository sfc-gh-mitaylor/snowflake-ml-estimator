"""
Benchmark Status & Utilities.

All benchmark execution happens on Snowflake SPCS via ML Jobs.
Use submit_jobs.py to run benchmarks:
    python -m src.submit_jobs submit --pool CPU_X64_S_TEST
    python -m src.submit_jobs submit-all
    python -m src.submit_jobs status

This module provides status checking and combination generation utilities.

Usage:
    python -m src.runner status
"""
import argparse
import itertools
import sys
from typing import List, Tuple, Set
import os

from .config import BenchmarkConfig, DEFAULT_CONFIG
from .estimators import EstimatorFactory, DEFAULT_FACTORY


Combination = Tuple[str, str, str, int, int]  # (model, task_type, pool, cols, rows)


def get_snowflake_session():
    """Create Snowflake session from connection name."""
    from snowflake.snowpark import Session
    conn_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
    session = Session.builder.config("connection_name", conn_name).create()
    session.sql("USE DATABASE ML_ESTIMATOR").collect()
    session.sql("USE SCHEMA PUBLIC").collect()
    return session


def get_tested_combinations(session, table_name: str) -> Set[Combination]:
    """Query existing results to find already-tested combinations."""
    try:
        df = session.table(table_name).select(
            "MODEL_CLASS", "TASK_TYPE", "COMPUTE_POOL", "N_COLS_SAMPLED", "N_ROWS_SAMPLED"
        ).distinct().to_pandas()
        
        return set(
            (row["MODEL_CLASS"], row["TASK_TYPE"], row["COMPUTE_POOL"], 
             row["N_COLS_SAMPLED"], row["N_ROWS_SAMPLED"])
            for _, row in df.iterrows()
        )
    except Exception:
        return set()


def generate_all_combinations(
    factory: EstimatorFactory,
    config: BenchmarkConfig,
) -> List[Combination]:
    """Generate all (model, task_type, pool, cols, rows) combinations."""
    combinations = []
    
    for model_name in factory.list_available():
        task_type = factory.get_task_type(model_name)
        row_limit = factory.get_row_limit(model_name)
        
        for pool, cols, rows in itertools.product(
            config.grid_pools, config.grid_cols, config.grid_rows
        ):
            if row_limit and rows > row_limit:
                continue
            combinations.append((model_name, task_type, pool, cols, rows))
    
    return combinations


def show_status(session, config: BenchmarkConfig, factory: EstimatorFactory):
    """Show current benchmark coverage status."""
    all_combos = generate_all_combinations(factory, config)
    tested = get_tested_combinations(session, config.results_table_name)
    remaining = [c for c in all_combos if c not in tested]
    
    print("\n" + "=" * 60)
    print("BENCHMARK STATUS")
    print("=" * 60)
    print(f"Total possible combinations: {len(all_combos)}")
    print(f"Already tested:              {len(tested)}")
    print(f"Remaining:                   {len(remaining)}")
    print(f"Coverage:                    {len(tested)/len(all_combos)*100:.1f}%")
    
    print("\nBy task type:")
    for task_type in ["classification", "regression", "clustering", "anomaly_detection"]:
        task_combos = [c for c in all_combos if c[1] == task_type]
        task_tested = [c for c in tested if c[1] == task_type]
        print(f"  {task_type}: {len(task_tested)}/{len(task_combos)}")
    
    print("\nBy compute pool:")
    for pool in config.grid_pools:
        pool_combos = [c for c in all_combos if c[2] == pool]
        pool_tested = [c for c in tested if c[2] == pool]
        print(f"  {pool}: {len(pool_tested)}/{len(pool_combos)}")


def main():
    parser = argparse.ArgumentParser(
        description="ML Benchmark Status (benchmarks run on Snowflake via submit_jobs.py)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show benchmark coverage status")

    run_parser = subparsers.add_parser("run", help="(REMOVED) Use submit_jobs.py instead")
    run_parser.add_argument("--max-combos", type=int, default=50)
    run_parser.add_argument("--runs", type=int, default=3)
    run_parser.add_argument("--task-type", type=str, default=None)
    run_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "run":
        print("ERROR: Local benchmark execution has been removed.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Benchmarks MUST run on Snowflake SPCS via ML Jobs.", file=sys.stderr)
        print("Use submit_jobs.py instead:", file=sys.stderr)
        print("", file=sys.stderr)
        print("  python -m src.submit_jobs submit --pool CPU_X64_S_TEST", file=sys.stderr)
        print("  python -m src.submit_jobs submit-all", file=sys.stderr)
        print("  python -m src.submit_jobs status", file=sys.stderr)
        sys.exit(1)

    elif args.command == "status":
        session = get_snowflake_session()
        show_status(session, DEFAULT_CONFIG, DEFAULT_FACTORY)


if __name__ == "__main__":
    main()
