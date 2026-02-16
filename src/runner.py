"""
Benchmark Runner - Executes ML benchmarks and records results.

Usage:
    python -m src.runner --help
    python -m src.runner run --max-combos 10
    python -m src.runner status
"""
import argparse
import itertools
import time
from datetime import datetime
from typing import List, Tuple, Set, Optional
import os

import numpy as np
import pandas as pd

from .config import BenchmarkConfig, DEFAULT_CONFIG, CREDIT_RATES, calculate_credits
from .estimators import EstimatorFactory, DEFAULT_FACTORY
from .data_generator import generate_data


Combination = Tuple[str, str, str, int, int]  # (model, task_type, pool, cols, rows)


def get_snowflake_session():
    """Create Snowflake session from connection name."""
    from snowflake.snowpark import Session
    conn_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
    return Session.builder.config("connection_name", conn_name).create()


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


def run_single_benchmark(
    model_name: str,
    task_type: str,
    n_cols: int,
    n_rows: int,
    factory: EstimatorFactory,
) -> float:
    """Run a single model fit and return duration in seconds."""
    X, y = generate_data(task_type, n_samples=n_rows, n_features=n_cols)
    
    if n_cols < X.shape[1]:
        X = X[:, :n_cols]
    
    model = factory.create(model_name)
    
    start = time.perf_counter()
    if task_type == "clustering":
        model.fit(X)
    else:
        model.fit(X, y)
    duration = time.perf_counter() - start
    
    return duration


def run_benchmarks(
    session,
    combinations: List[Combination],
    config: BenchmarkConfig,
    factory: EstimatorFactory,
    runs_per_combo: int = 3,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run benchmarks for given combinations and save results."""
    results = []
    total = len(combinations) * runs_per_combo
    completed = 0
    
    print(f"\nRunning {len(combinations)} combinations x {runs_per_combo} runs = {total} total")
    print("=" * 60)
    
    for model_name, task_type, pool, n_cols, n_rows in combinations:
        for run_id in range(1, runs_per_combo + 1):
            completed += 1
            print(f"[{completed}/{total}] {model_name} | {pool} | {n_cols}c x {n_rows:,}r | run {run_id}")
            
            if dry_run:
                duration = np.random.uniform(1, 10)
            else:
                try:
                    duration = run_single_benchmark(
                        model_name, task_type, n_cols, n_rows, factory
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    duration = -1.0
            
            credits = calculate_credits(pool, duration) if duration > 0 else 0.0
            
            results.append({
                "MODEL_CLASS": model_name,
                "TASK_TYPE": task_type,
                "COMPUTE_POOL": pool,
                "RUN_ID": run_id,
                "N_COLS_SAMPLED": n_cols,
                "N_ROWS_SAMPLED": n_rows,
                "DURATION_SECONDS": round(duration, 4),
                "ESTIMATED_CREDITS": round(credits, 6),
                "START_TIMESTAMP": datetime.now().isoformat(),
            })
            
            if duration > 0:
                print(f"  -> {duration:.2f}s, {credits:.6f} credits")
    
    df = pd.DataFrame(results)
    
    if not dry_run and len(df) > 0:
        snow_df = session.create_dataframe(df)
        snow_df.write.mode("append").save_as_table(config.results_table_name)
        print(f"\nSaved {len(df)} results to {config.results_table_name}")
    
    return df


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
    for task_type in ["classification", "regression", "clustering"]:
        task_combos = [c for c in all_combos if c[1] == task_type]
        task_tested = [c for c in tested if c[1] == task_type]
        print(f"  {task_type}: {len(task_tested)}/{len(task_combos)}")
    
    print("\nBy compute pool:")
    for pool in config.grid_pools:
        pool_combos = [c for c in all_combos if c[2] == pool]
        pool_tested = [c for c in tested if c[2] == pool]
        print(f"  {pool}: {len(pool_tested)}/{len(pool_combos)}")


def main():
    parser = argparse.ArgumentParser(description="ML Benchmark Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--max-combos", type=int, default=50, help="Max combinations to run")
    run_parser.add_argument("--runs", type=int, default=3, help="Runs per combination")
    run_parser.add_argument("--task-type", choices=["classification", "regression", "clustering"])
    run_parser.add_argument("--dry-run", action="store_true", help="Simulate without running")
    
    subparsers.add_parser("status", help="Show benchmark coverage status")
    
    args = parser.parse_args()
    
    session = get_snowflake_session()
    config = DEFAULT_CONFIG
    factory = DEFAULT_FACTORY
    
    if args.command == "status":
        show_status(session, config, factory)
    
    elif args.command == "run":
        all_combos = generate_all_combinations(factory, config)
        tested = get_tested_combinations(session, config.results_table_name)
        remaining = [c for c in all_combos if c not in tested]
        
        if args.task_type:
            remaining = [c for c in remaining if c[1] == args.task_type]
        
        to_run = remaining[:args.max_combos]
        
        if not to_run:
            print("No remaining combinations to run!")
            return
        
        run_benchmarks(session, to_run, config, factory, args.runs, args.dry_run)


if __name__ == "__main__":
    main()
