"""
Benchmark Job - Runs on Snowflake Compute Pool via ML Jobs.

This script is submitted to Snowflake ML Jobs and executes ON the compute pool,
not locally. It benchmarks sklearn model training and writes results directly
to a Snowflake table.

Usage (via ML Jobs):
    from snowflake.ml.jobs import submit_file
    job = submit_file("src/benchmark_job.py", "CPU_X64_S", 
                      args=["--pool", "CPU_X64_S", "--task-type", "classification"])
"""
import argparse
import time
from datetime import datetime
from typing import Tuple
import platform
import os

import numpy as np
import pandas as pd
from snowflake.snowpark import Session


ESTIMATOR_CONFIGS = {
    "GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier", "classification", {"n_estimators": 100, "max_depth": 3, "random_state": 42}),
    "AdaBoostClassifier": ("sklearn.ensemble", "AdaBoostClassifier", "classification", {"n_estimators": 50, "random_state": 42}),
    "LogisticRegression": ("sklearn.linear_model", "LogisticRegression", "classification", {"max_iter": 1000, "random_state": 42}),
    "SVC": ("sklearn.svm", "SVC", "classification", {"random_state": 42}),
    "KNeighborsClassifier": ("sklearn.neighbors", "KNeighborsClassifier", "classification", {"n_neighbors": 5}),
    "GaussianNB": ("sklearn.naive_bayes", "GaussianNB", "classification", {}),
    "RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier", "classification", {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
    "DecisionTreeClassifier": ("sklearn.tree", "DecisionTreeClassifier", "classification", {"random_state": 42}),
    "ExtraTreesClassifier": ("sklearn.ensemble", "ExtraTreesClassifier", "classification", {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
    "HistGradientBoostingClassifier": ("sklearn.ensemble", "HistGradientBoostingClassifier", "classification", {"max_iter": 100, "random_state": 42}),
    "SGDClassifier": ("sklearn.linear_model", "SGDClassifier", "classification", {"random_state": 42}),
    
    "GradientBoostingRegressor": ("sklearn.ensemble", "GradientBoostingRegressor", "regression", {"n_estimators": 100, "max_depth": 3, "random_state": 42}),
    "RandomForestRegressor": ("sklearn.ensemble", "RandomForestRegressor", "regression", {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
    "LinearRegression": ("sklearn.linear_model", "LinearRegression", "regression", {}),
    "Ridge": ("sklearn.linear_model", "Ridge", "regression", {"random_state": 42}),
    "Lasso": ("sklearn.linear_model", "Lasso", "regression", {"random_state": 42}),
    "ElasticNet": ("sklearn.linear_model", "ElasticNet", "regression", {"random_state": 42}),
    "SVR": ("sklearn.svm", "SVR", "regression", {}),
    "KNeighborsRegressor": ("sklearn.neighbors", "KNeighborsRegressor", "regression", {"n_neighbors": 5}),
    "DecisionTreeRegressor": ("sklearn.tree", "DecisionTreeRegressor", "regression", {"random_state": 42}),
    "ExtraTreesRegressor": ("sklearn.ensemble", "ExtraTreesRegressor", "regression", {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
    "HistGradientBoostingRegressor": ("sklearn.ensemble", "HistGradientBoostingRegressor", "regression", {"max_iter": 100, "random_state": 42}),
    "SGDRegressor": ("sklearn.linear_model", "SGDRegressor", "regression", {"random_state": 42}),
    "AdaBoostRegressor": ("sklearn.ensemble", "AdaBoostRegressor", "regression", {"n_estimators": 50, "random_state": 42}),
    
    "KMeans": ("sklearn.cluster", "KMeans", "clustering", {"n_clusters": 8, "random_state": 42, "n_init": 10}),
    "MiniBatchKMeans": ("sklearn.cluster", "MiniBatchKMeans", "clustering", {"n_clusters": 8, "random_state": 42, "n_init": 10}),
    "DBSCAN": ("sklearn.cluster", "DBSCAN", "clustering", {"eps": 0.5, "min_samples": 5}),
    "AgglomerativeClustering": ("sklearn.cluster", "AgglomerativeClustering", "clustering", {"n_clusters": 8}),
    "Birch": ("sklearn.cluster", "Birch", "clustering", {"n_clusters": 8}),
    
    "IsolationForest": ("sklearn.ensemble", "IsolationForest", "anomaly_detection", {"n_estimators": 100, "random_state": 42}),
    "OneClassSVM": ("sklearn.svm", "OneClassSVM", "anomaly_detection", {}),
}

ROW_LIMITS = {
    "SVC": 50_000,
    "SVR": 50_000,
    "DBSCAN": 50_000,
    "AgglomerativeClustering": 30_000,
    "OneClassSVM": 30_000,
}

GRID_COLS = [25, 50, 100]
GRID_ROWS = [50_000, 200_000, 500_000]

RESULTS_TABLE = "ML_ESTIMATOR.PUBLIC.ML_BENCHMARK_RESULTS"


def create_model(name: str):
    """Dynamically import and instantiate a model."""
    module_name, class_name, _, params = ESTIMATOR_CONFIGS[name]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**params)


def generate_data(task_type: str, n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for benchmarking."""
    from sklearn.datasets import make_classification, make_regression, make_blobs
    
    if task_type == "classification":
        n_informative = max(2, int(n_features * 0.7))
        n_redundant = max(0, min(int(n_features * 0.1), n_features - n_informative - 1))
        return make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=n_informative, n_redundant=n_redundant,
            random_state=42
        )
    elif task_type == "regression":
        n_informative = max(2, int(n_features * 0.7))
        return make_regression(
            n_samples=n_samples, n_features=n_features,
            n_informative=n_informative, random_state=42
        )
    elif task_type == "clustering":
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=8, random_state=42)
        return X, y
    elif task_type == "anomaly_detection":
        rng = np.random.RandomState(42)
        n_outliers = int(n_samples * 0.05)
        n_normal = n_samples - n_outliers
        X_normal = rng.randn(n_normal, n_features)
        X_outliers = rng.uniform(-10, 10, (n_outliers, n_features))
        X = np.vstack([X_normal, X_outliers])
        y = np.hstack([np.ones(n_normal), -np.ones(n_outliers)])
        shuffle_idx = rng.permutation(n_samples)
        return X[shuffle_idx], y[shuffle_idx]
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def run_single_benchmark(model_name: str, task_type: str, n_cols: int, n_rows: int) -> float:
    """Run a single model fit and return duration in seconds."""
    X, y = generate_data(task_type, n_samples=n_rows, n_features=n_cols)
    
    if n_cols < X.shape[1]:
        X = X[:, :n_cols]
    
    model = create_model(model_name)
    
    start = time.perf_counter()
    if task_type in ("clustering", "anomaly_detection"):
        model.fit(X)
    else:
        model.fit(X, y)
    duration = time.perf_counter() - start
    
    return duration


def get_models_for_task(task_type: str) -> list:
    """Get all models for a given task type."""
    return [name for name, config in ESTIMATOR_CONFIGS.items() if config[2] == task_type]


def main():
    parser = argparse.ArgumentParser(description="ML Benchmark Job")
    parser.add_argument("--pool", required=True, help="Compute pool name (for recording)")
    parser.add_argument("--task-type", required=True, choices=["classification", "regression", "clustering", "anomaly_detection"])
    parser.add_argument("--runs", type=int, default=2, help="Runs per combination")
    parser.add_argument("--max-combos", type=int, default=None, help="Max combinations to run")
    args = parser.parse_args()
    
    session = Session.builder.getOrCreate()
    session.sql("USE DATABASE ML_ESTIMATOR").collect()
    session.sql("USE SCHEMA PUBLIC").collect()
    
    pool = args.pool
    task_type = args.task_type
    runs_per_combo = args.runs
    
    models = get_models_for_task(task_type)
    print(f"Running benchmarks on pool: {pool}")
    print(f"Task type: {task_type}")
    print(f"Models: {len(models)}")
    print(f"Host: {platform.node()}")
    print("=" * 60)
    
    combinations = []
    for model_name in models:
        row_limit = ROW_LIMITS.get(model_name)
        for n_cols in GRID_COLS:
            for n_rows in GRID_ROWS:
                if row_limit and n_rows > row_limit:
                    continue
                combinations.append((model_name, n_cols, n_rows))
    
    if args.max_combos:
        combinations = combinations[:args.max_combos]
    
    total = len(combinations) * runs_per_combo
    print(f"Total benchmarks to run: {total}")
    print("=" * 60)
    
    results = []
    completed = 0
    
    for model_name, n_cols, n_rows in combinations:
        for run_id in range(1, runs_per_combo + 1):
            completed += 1
            print(f"[{completed}/{total}] {model_name} | {n_cols}c x {n_rows:,}r | run {run_id}")
            
            try:
                duration = run_single_benchmark(model_name, task_type, n_cols, n_rows)
                print(f"  -> {duration:.2f}s")
            except Exception as e:
                print(f"  ERROR: {e}")
                duration = -1.0
            
            result = {
                "MODEL_CLASS": model_name,
                "TASK_TYPE": task_type,
                "COMPUTE_POOL": pool,
                "RUN_ID": run_id,
                "N_COLS_SAMPLED": n_cols,
                "N_ROWS_SAMPLED": n_rows,
                "DURATION_SECONDS": round(duration, 4),
                "START_TIMESTAMP": datetime.now().isoformat(),
            }
            results.append(result)
            
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
    print("BENCHMARK JOB COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
