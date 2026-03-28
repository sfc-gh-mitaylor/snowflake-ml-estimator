"""
Inference Benchmark Job - Runs on Snowflake Compute Pool via ML Jobs.

Benchmarks batch prediction time for trained sklearn classifiers.
Measures how long it takes to predict N rows after training on a fixed dataset.
Results are written to ML_ESTIMATOR.PUBLIC.ML_INFERENCE_RESULTS.

Usage (via ML Jobs — see submit_jobs.py):
    from snowflake.ml.jobs import submit_file
    job = submit_file("src/inference_benchmark_job.py", "CPU_X64_S_TEST",
                      stage_name="ML_ESTIMATOR.PUBLIC.ML_JOBS_STAGE",
                      args=["--pool", "CPU_X64_S_TEST", "--runs", "3"])
"""
import argparse
import time
from datetime import datetime
from typing import Tuple
import platform

import numpy as np
import pandas as pd
from snowflake.snowpark import Session


CLASSIFIERS = {
    "GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier", {"n_estimators": 100, "max_depth": 3, "random_state": 42}),
    "AdaBoostClassifier": ("sklearn.ensemble", "AdaBoostClassifier", {"n_estimators": 50, "random_state": 42}),
    "LogisticRegression": ("sklearn.linear_model", "LogisticRegression", {"max_iter": 1000, "random_state": 42, "n_jobs": -1}),
    "SVC": ("sklearn.svm", "SVC", {"random_state": 42}),
    "KNeighborsClassifier": ("sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 5, "n_jobs": -1}),
    "GaussianNB": ("sklearn.naive_bayes", "GaussianNB", {}),
    "RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
    "DecisionTreeClassifier": ("sklearn.tree", "DecisionTreeClassifier", {"max_depth": 15, "random_state": 42}),
    "ExtraTreesClassifier": ("sklearn.ensemble", "ExtraTreesClassifier", {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
    "HistGradientBoostingClassifier": ("sklearn.ensemble", "HistGradientBoostingClassifier", {"max_iter": 100, "random_state": 42}),
    "SGDClassifier": ("sklearn.linear_model", "SGDClassifier", {"loss": "log_loss", "max_iter": 1000, "random_state": 42, "n_jobs": -1}),
}

ROW_LIMITS = {
    "SVC": 100_000,
}

CREDIT_RATES = {
    "CPU_X64_XS_TEST": 0.06,
    "CPU_X64_S_TEST": 0.12,
    "CPU_X64_M_TEST": 0.24,
    "CPU_X64_SL_TEST": 0.48,
}

TRAIN_ROWS = 100_000
TRAIN_COLS = 50
GRID_PREDICT_ROWS = [1_000, 10_000, 50_000, 100_000, 500_000]
GRID_COLS = [25, 50, 100]

RESULTS_TABLE = "ML_ESTIMATOR.PUBLIC.ML_INFERENCE_RESULTS"


def create_model(name: str):
    module_name, class_name, params = CLASSIFIERS[name]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**params)


def generate_data(n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import make_classification
    n_informative = max(2, int(n_features * 0.7))
    n_redundant = max(0, min(int(n_features * 0.1), n_features - n_informative - 1))
    return make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant,
        random_state=42
    )


def run_inference_benchmark(model_name: str, n_cols: int, n_predict_rows: int) -> float:
    train_rows = min(TRAIN_ROWS, ROW_LIMITS.get(model_name, TRAIN_ROWS))
    X_train, y_train = generate_data(n_samples=train_rows, n_features=n_cols)
    model = create_model(model_name)
    model.fit(X_train, y_train)

    X_pred, _ = generate_data(n_samples=n_predict_rows, n_features=n_cols)

    start = time.perf_counter()
    model.predict(X_pred)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Inference Benchmark Job")
    parser.add_argument("--pool", required=True, help="Compute pool name (for recording)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per combination")
    parser.add_argument("--models", nargs="*", default=None, help="Specific models to benchmark")
    args = parser.parse_args()

    session = Session.builder.getOrCreate()
    session.sql("USE DATABASE ML_ESTIMATOR").collect()
    session.sql("USE SCHEMA PUBLIC").collect()

    pool = args.pool
    runs_per_combo = args.runs
    credit_rate = CREDIT_RATES.get(pool, 0.12)

    models = args.models if args.models else list(CLASSIFIERS.keys())
    models = [m for m in models if m in CLASSIFIERS]

    combinations = []
    for model_name in models:
        for n_cols in GRID_COLS:
            for n_predict_rows in GRID_PREDICT_ROWS:
                row_limit = ROW_LIMITS.get(model_name)
                if row_limit and n_predict_rows > row_limit:
                    continue
                combinations.append((model_name, n_cols, n_predict_rows))

    total = len(combinations) * runs_per_combo
    print(f"Pool: {pool} | Credit rate: {credit_rate}/hr")
    print(f"Models: {len(models)} | Combos: {len(combinations)} | Runs: {total}")
    print(f"Host: {platform.node()}")
    print("=" * 60)

    results = []
    completed = 0

    for model_name, n_cols, n_predict_rows in combinations:
        for run_id in range(1, runs_per_combo + 1):
            completed += 1
            print(f"[{completed}/{total}] {model_name} | {n_cols}c | predict {n_predict_rows:,}r | run {run_id}")

            try:
                duration = run_inference_benchmark(model_name, n_cols, n_predict_rows)
                credits = (duration / 3600) * credit_rate
                print(f"  -> {duration:.4f}s | {credits:.6f} credits")
            except Exception as e:
                print(f"  ERROR: {e}")
                duration = -1.0
                credits = 0.0

            results.append({
                "MODEL_CLASS": model_name,
                "TASK_TYPE": "classification",
                "COMPUTE_POOL": pool,
                "RUN_ID": run_id,
                "N_COLS": n_cols,
                "N_PREDICT_ROWS": n_predict_rows,
                "PREDICT_DURATION_SECONDS": round(duration, 6),
                "ESTIMATED_CREDITS": round(credits, 8),
                "START_TIMESTAMP": datetime.now().isoformat(),
            })

            if len(results) >= 10:
                try:
                    df = pd.DataFrame(results)
                    snow_df = session.create_dataframe(df)
                    snow_df.write.mode("append").save_as_table(RESULTS_TABLE)
                    print(f"  [SAVED {len(results)} results]")
                    results = []
                except Exception as e:
                    print(f"  [SAVE FAILED: {e}]")

    if results:
        try:
            df = pd.DataFrame(results)
            snow_df = session.create_dataframe(df)
            snow_df.write.mode("append").save_as_table(RESULTS_TABLE)
            print(f"\nSaved final {len(results)} results")
        except Exception as e:
            print(f"\nFailed to save final batch: {e}")

    print("\n" + "=" * 60)
    print(f"INFERENCE BENCHMARK COMPLETE — {completed} runs on {pool}")
    print("=" * 60)


if __name__ == "__main__":
    main()
