"""
Regression Benchmark Job - Runs on Snowflake Compute Pool via ML Jobs.

Benchmarks sklearn + boosting regressor training time on actual SPCS compute.
Results written to ML_ESTIMATOR.PUBLIC.ML_BENCHMARK_RESULTS (TASK_TYPE='regression').

Usage (via ML Jobs — see submit_jobs.py):
    python -m src.submit_jobs submit --pool CPU_X64_S_TEST --type regression
"""
import argparse
import time
from datetime import datetime
from typing import Tuple
import platform

import numpy as np
import pandas as pd
from snowflake.snowpark import Session


SKLEARN_REGRESSORS = {
    "RandomForestRegressor":         ("sklearn.ensemble",      "RandomForestRegressor",         {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42}),
    "GradientBoostingRegressor":     ("sklearn.ensemble",      "GradientBoostingRegressor",     {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8, "random_state": 42}),
    "ExtraTreesRegressor":           ("sklearn.ensemble",      "ExtraTreesRegressor",           {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42}),
    "HistGradientBoostingRegressor": ("sklearn.ensemble",      "HistGradientBoostingRegressor", {"max_iter": 100, "learning_rate": 0.1, "max_depth": 10, "random_state": 42}),
    "DecisionTreeRegressor":         ("sklearn.tree",          "DecisionTreeRegressor",         {"max_depth": 15, "min_samples_leaf": 5, "random_state": 42}),
    "LinearRegression":              ("sklearn.linear_model",  "LinearRegression",              {"n_jobs": -1}),
    "Ridge":                         ("sklearn.linear_model",  "Ridge",                         {"alpha": 1.0}),
    "Lasso":                         ("sklearn.linear_model",  "Lasso",                         {"alpha": 1.0}),
    "ElasticNet":                    ("sklearn.linear_model",  "ElasticNet",                    {"alpha": 1.0, "l1_ratio": 0.5}),
    "SGDRegressor":                  ("sklearn.linear_model",  "SGDRegressor",                  {"loss": "squared_error", "max_iter": 1000, "random_state": 42}),
    "SVR":                           ("sklearn.svm",           "SVR",                           {"C": 1.0, "kernel": "rbf", "gamma": "scale"}),
}

BOOSTING_REGRESSORS = {
    "XGBRegressor":     ("xgboost",   "XGBRegressor",     {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1, "random_state": 42}),
    "LGBMRegressor":    ("lightgbm",  "LGBMRegressor",    {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 31, "n_jobs": -1, "random_state": 42, "verbose": -1}),
    "CatBoostRegressor":("catboost",  "CatBoostRegressor",{"iterations": 150, "learning_rate": 0.05, "depth": 6, "random_state": 42, "verbose": 0}),
}

ROW_LIMITS = {
    "SVR": 100_000,
}

CREDIT_RATES = {
    "CPU_X64_XS_TEST": 0.06,
    "CPU_X64_S_TEST":  0.12,
    "CPU_X64_M_TEST":  0.24,
    "CPU_X64_SL_TEST": 0.48,
}

GRID_COLS = [25, 50, 100]
GRID_ROWS = [50_000, 200_000, 500_000]

RESULTS_TABLE = "ML_ESTIMATOR.PUBLIC.ML_BENCHMARK_RESULTS"


def load_available_regressors() -> dict:
    available = dict(SKLEARN_REGRESSORS)
    for name, (module_name, class_name, params) in BOOSTING_REGRESSORS.items():
        try:
            __import__(module_name)
            available[name] = (module_name, class_name, params)
            print(f"  [OK] {name} ({module_name})")
        except ImportError:
            print(f"  [SKIP] {name} — {module_name} not available")
    return available


def create_model(name: str, regressors: dict):
    module_name, class_name, params = regressors[name]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**params)


def generate_data(n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import make_regression
    n_informative = max(2, int(n_features * 0.7))
    return make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=0.1,
        random_state=42,
    )


def run_single_benchmark(model_name: str, n_cols: int, n_rows: int, regressors: dict) -> float:
    X, y = generate_data(n_samples=n_rows, n_features=n_cols)
    model = create_model(model_name, regressors)
    start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Regression Benchmark Job")
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

    print(f"Pool: {pool} | Credit rate: {credit_rate}/hr")
    print(f"Host: {platform.node()}")
    print("Loading available regressors...")
    regressors = load_available_regressors()

    models = args.models if args.models else list(regressors.keys())
    models = [m for m in models if m in regressors]

    combinations = []
    for model_name in models:
        row_limit = ROW_LIMITS.get(model_name)
        for n_cols in GRID_COLS:
            for n_rows in GRID_ROWS:
                if row_limit and n_rows > row_limit:
                    continue
                combinations.append((model_name, n_cols, n_rows))

    total = len(combinations) * runs_per_combo
    print(f"Models: {len(models)} | Combos: {len(combinations)} | Runs: {total}")
    print("=" * 60)

    results = []
    completed = 0

    for model_name, n_cols, n_rows in combinations:
        for run_id in range(1, runs_per_combo + 1):
            completed += 1
            print(f"[{completed}/{total}] {model_name} | {n_cols}c x {n_rows:,}r | run {run_id}")

            try:
                duration = run_single_benchmark(model_name, n_cols, n_rows, regressors)
                credits = (duration / 3600) * credit_rate
                print(f"  -> {duration:.2f}s | {credits:.6f} credits")
            except Exception as e:
                print(f"  ERROR: {e}")
                duration = -1.0
                credits = 0.0

            results.append({
                "MODEL_CLASS": model_name,
                "TASK_TYPE": "regression",
                "COMPUTE_POOL": pool,
                "RUN_ID": run_id,
                "N_COLS_SAMPLED": n_cols,
                "N_ROWS_SAMPLED": n_rows,
                "DURATION_SECONDS": round(duration, 4),
                "ESTIMATED_CREDITS": round(credits, 6),
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
    print(f"REGRESSION BENCHMARK COMPLETE — {completed} runs on {pool}")
    print("=" * 60)


if __name__ == "__main__":
    main()
