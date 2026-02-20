"""
Submit Benchmark Jobs - Fire and forget ML Jobs to Snowflake.

This script submits benchmark jobs to actual Snowflake compute pools.
Once submitted, you can close your laptop - jobs run on Snowflake.

Usage:
    python -m src.submit_jobs --help
    python -m src.submit_jobs submit --pool CPU_X64_S --task-type classification
    python -m src.submit_jobs submit-all
    python -m src.submit_jobs status
    python -m src.submit_jobs status --job-id <id>
"""
import argparse
import os
import json
from datetime import datetime
from pathlib import Path

from snowflake.snowpark import Session
from snowflake.ml.jobs import submit_file, list_jobs, get_job


COMPUTE_POOLS = [
    "CPU_X64_XS_TEST",
    "CPU_X64_S_TEST", 
    "CPU_X64_M_TEST",
    "CPU_X64_SL_TEST",
]

TASK_TYPES = ["classification", "regression", "clustering", "anomaly_detection"]

STAGE_NAME = "ML_ESTIMATOR.PUBLIC.ML_JOBS_STAGE"
JOB_TRACKING_FILE = Path(__file__).parent.parent / ".job_tracking.json"


def get_session():
    """Create Snowflake session."""
    conn_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
    session = Session.builder.config("connection_name", conn_name).create()
    session.sql("USE DATABASE ML_ESTIMATOR").collect()
    session.sql("USE SCHEMA PUBLIC").collect()
    return session


def ensure_stage(session):
    """Create stage for ML Jobs payloads if it doesn't exist."""
    session.sql(f"CREATE STAGE IF NOT EXISTS {STAGE_NAME}").collect()
    print(f"Stage ready: {STAGE_NAME}")


def submit_single_job(session, pool: str, task_type: str, runs: int = 2, max_combos: int = None):
    """Submit a single benchmark job to a compute pool."""
    script_path = Path(__file__).parent / "benchmark_job.py"
    
    args = ["--pool", pool, "--task-type", task_type, "--runs", str(runs)]
    if max_combos:
        args.extend(["--max-combos", str(max_combos)])
    
    print(f"Submitting job: pool={pool}, task_type={task_type}")
    
    job = submit_file(
        str(script_path),
        pool,
        stage_name=STAGE_NAME,
        args=args,
        session=session,
    )
    
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")
    
    return job


def submit_all_jobs(session, runs: int = 2):
    """Submit jobs for all pool × task_type combinations."""
    jobs = []
    
    for pool in COMPUTE_POOLS:
        for task_type in TASK_TYPES:
            try:
                job = submit_single_job(session, pool, task_type, runs)
                jobs.append({
                    "job_id": job.id,
                    "pool": pool,
                    "task_type": task_type,
                    "submitted_at": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"  ERROR submitting {pool}/{task_type}: {e}")
    
    save_job_tracking(jobs)
    
    print("\n" + "=" * 60)
    print(f"Submitted {len(jobs)} jobs")
    print("You can now close your laptop - jobs run on Snowflake")
    print(f"Check status with: python -m src.submit_jobs status")
    print("=" * 60)
    
    return jobs


def save_job_tracking(jobs: list):
    """Save job IDs to tracking file."""
    existing = []
    if JOB_TRACKING_FILE.exists():
        with open(JOB_TRACKING_FILE) as f:
            existing = json.load(f)
    
    existing.extend(jobs)
    
    with open(JOB_TRACKING_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    
    print(f"Job IDs saved to {JOB_TRACKING_FILE}")


def show_status(session, job_id: str = None):
    """Show status of submitted jobs."""
    if job_id:
        job = get_job(job_id, session=session)
        print(f"Job ID: {job.id}")
        print(f"Status: {job.status}")
        print(f"\nLogs:")
        print(job.get_logs())
        return
    
    print("Recent ML Jobs:")
    print("=" * 60)
    
    jobs_df = list_jobs(limit=20, session=session)
    print(jobs_df.to_string())
    
    if JOB_TRACKING_FILE.exists():
        print(f"\n\nTracked jobs from {JOB_TRACKING_FILE}:")
        print("-" * 60)
        with open(JOB_TRACKING_FILE) as f:
            tracked = json.load(f)
        
        for j in tracked[-10:]:
            try:
                job = get_job(j["job_id"], session=session)
                print(f"{j['pool']:12} | {j['task_type']:20} | {job.status:10} | {j['job_id']}")
            except Exception as e:
                print(f"{j['pool']:12} | {j['task_type']:20} | ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Submit ML Benchmark Jobs to Snowflake")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    submit_parser = subparsers.add_parser("submit", help="Submit a single job")
    submit_parser.add_argument("--pool", required=True, choices=COMPUTE_POOLS)
    submit_parser.add_argument("--task-type", required=True, choices=TASK_TYPES)
    submit_parser.add_argument("--runs", type=int, default=2)
    submit_parser.add_argument("--max-combos", type=int)
    
    all_parser = subparsers.add_parser("submit-all", help="Submit jobs for all pools × task types")
    all_parser.add_argument("--runs", type=int, default=2)
    
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("--job-id", help="Specific job ID to check")
    
    args = parser.parse_args()
    
    session = get_session()
    
    if args.command == "submit":
        ensure_stage(session)
        job = submit_single_job(session, args.pool, args.task_type, args.runs, args.max_combos)
        save_job_tracking([{
            "job_id": job.id,
            "pool": args.pool,
            "task_type": args.task_type,
            "submitted_at": datetime.now().isoformat(),
        }])
        print("\nJob submitted. You can close your laptop.")
        
    elif args.command == "submit-all":
        ensure_stage(session)
        submit_all_jobs(session, args.runs)
        
    elif args.command == "status":
        show_status(session, args.job_id)


if __name__ == "__main__":
    main()
