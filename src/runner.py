"""
Benchmark Status — query Snowflake for coverage stats.

All benchmark execution happens on Snowflake SPCS via ML Jobs.
Use submit_jobs.py to run benchmarks:
    python -m src.submit_jobs submit --pool CPU_X64_S_TEST
    python -m src.submit_jobs submit-all
    python -m src.submit_jobs status
"""
import argparse
import sys
import os


RESULTS_TABLE = "ML_ESTIMATOR.PUBLIC.ML_BENCHMARK_RESULTS"


def get_snowflake_session():
    from snowflake.snowpark import Session
    conn_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
    session = Session.builder.config("connection_name", conn_name).create()
    session.sql("USE DATABASE ML_ESTIMATOR").collect()
    session.sql("USE SCHEMA PUBLIC").collect()
    return session


def show_status(session):
    df = session.sql(f"""
        SELECT TASK_TYPE, COMPUTE_POOL, MODEL_CLASS,
               COUNT(DISTINCT N_COLS_SAMPLED || '-' || N_ROWS_SAMPLED) as COMBOS,
               COUNT(*) as RUNS,
               ROUND(AVG(DURATION_SECONDS), 2) as AVG_SEC
        FROM {RESULTS_TABLE}
        WHERE DURATION_SECONDS > 0
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """).to_pandas()

    total_runs = df["RUNS"].sum()
    total_combos = df["COMBOS"].sum()

    print("\n" + "=" * 60)
    print("BENCHMARK STATUS")
    print("=" * 60)
    print(f"Total successful runs: {total_runs:,}")
    print(f"Model/pool/grid combos covered: {total_combos:,}")
    print(f"Distinct models: {df['MODEL_CLASS'].nunique()}")
    print(f"Distinct pools: {df['COMPUTE_POOL'].nunique()}")

    print("\nBy task type:")
    for task, group in df.groupby("TASK_TYPE"):
        print(f"  {task}: {group['RUNS'].sum():,} runs, {group['MODEL_CLASS'].nunique()} models")

    print("\nBy compute pool:")
    for pool, group in df.groupby("COMPUTE_POOL"):
        print(f"  {pool}: {group['RUNS'].sum():,} runs, {group['MODEL_CLASS'].nunique()} models")


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
        print("Benchmarks MUST run on Snowflake SPCS via ML Jobs.", file=sys.stderr)
        print("Use: python -m src.submit_jobs submit-all", file=sys.stderr)
        sys.exit(1)

    elif args.command == "status":
        session = get_snowflake_session()
        show_status(session)


if __name__ == "__main__":
    main()
