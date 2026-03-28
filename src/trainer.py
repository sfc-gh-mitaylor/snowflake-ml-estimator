"""
ML Cost Estimator Trainer - Trains and registers predictor models.

Trains two models:
  1. Training duration predictor (from ML_BENCHMARK_RESULTS)
  2. Inference duration predictor (from ML_INFERENCE_RESULTS)

Usage:
    python -m src.trainer train-training --version V4
    python -m src.trainer train-inference --version V1
    python -m src.trainer train-all
"""
import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def get_snowflake_session():
    from snowflake.snowpark import Session
    conn_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
    session = Session.builder.config("connection_name", conn_name).create()
    session.sql("USE DATABASE ML_ESTIMATOR").collect()
    session.sql("USE SCHEMA PUBLIC").collect()
    return session


def train_predictor(df: pd.DataFrame, feature_cols: list, target_col: str) -> tuple:
    df = df[df[target_col] > 0].copy()
    if len(df) < 10:
        raise ValueError(f"Need at least 10 successful runs, got {len(df)}")

    le_model = LabelEncoder()
    le_pool = LabelEncoder()
    le_task = LabelEncoder()

    df["MODEL_ENCODED"] = le_model.fit_transform(df["MODEL_CLASS"])
    df["POOL_ENCODED"] = le_pool.fit_transform(df["COMPUTE_POOL"])
    df["TASK_ENCODED"] = le_task.fit_transform(df["TASK_TYPE"])

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_models": int(df["MODEL_CLASS"].nunique()),
        "n_pools": int(df["COMPUTE_POOL"].nunique()),
    }

    encoders = {"model": le_model, "pool": le_pool, "task": le_task}
    return model, metrics, encoders


def register_model(session, sklearn_model, model_name: str, version: str,
                   sample_input: pd.DataFrame, metrics: dict, comment: str):
    from snowflake.ml.registry import Registry

    registry = Registry(session, database_name="ML_ESTIMATOR", schema_name="PUBLIC")
    mv = registry.log_model(
        model=sklearn_model,
        model_name=model_name,
        version_name=version,
        sample_input_data=sample_input,
        metrics=metrics,
        comment=comment,
    )
    print(f"Registered: {mv.model_name}/{mv.version_name}")
    return mv


def train_training_model(session, version: str = "V4"):
    print("Loading training benchmark data...")
    df = session.table("ML_ESTIMATOR.PUBLIC.ML_BENCHMARK_RESULTS").to_pandas()
    print(f"Loaded {len(df)} rows ({df['MODEL_CLASS'].nunique()} models, {df['COMPUTE_POOL'].nunique()} pools)")

    feature_cols = ["MODEL_ENCODED", "POOL_ENCODED", "TASK_ENCODED", "N_COLS_SAMPLED", "N_ROWS_SAMPLED"]
    model, metrics, encoders = train_predictor(df, feature_cols, "DURATION_SECONDS")

    print(f"\nTraining Predictor Metrics:")
    print(f"  R2:   {metrics['r2']:.4f}")
    print(f"  MAE:  {metrics['mae']:.2f}s")
    print(f"  RMSE: {metrics['rmse']:.2f}s")
    print(f"  Train: {metrics['n_train']} | Test: {metrics['n_test']}")

    sample_df = pd.DataFrame({
        "MODEL_ENCODED": [0], "POOL_ENCODED": [1], "TASK_ENCODED": [0],
        "N_COLS_SAMPLED": [50], "N_ROWS_SAMPLED": [100000],
    })

    register_model(session, model, "ML_COST_ESTIMATOR", version, sample_df, metrics,
                   f"Training duration predictor {version}")
    return model, metrics


def train_inference_model(session, version: str = "V1"):
    print("Loading inference benchmark data...")
    df = session.table("ML_ESTIMATOR.PUBLIC.ML_INFERENCE_RESULTS").to_pandas()
    print(f"Loaded {len(df)} rows ({df['MODEL_CLASS'].nunique()} models, {df['COMPUTE_POOL'].nunique()} pools)")

    feature_cols = ["MODEL_ENCODED", "POOL_ENCODED", "TASK_ENCODED", "N_COLS", "N_PREDICT_ROWS"]
    model, metrics, encoders = train_predictor(df, feature_cols, "PREDICT_DURATION_SECONDS")

    print(f"\nInference Predictor Metrics:")
    print(f"  R2:   {metrics['r2']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}s")
    print(f"  RMSE: {metrics['rmse']:.4f}s")
    print(f"  Train: {metrics['n_train']} | Test: {metrics['n_test']}")

    sample_df = pd.DataFrame({
        "MODEL_ENCODED": [0], "POOL_ENCODED": [1], "TASK_ENCODED": [0],
        "N_COLS": [50], "N_PREDICT_ROWS": [10000],
    })

    register_model(session, model, "ML_INFERENCE_ESTIMATOR", version, sample_df, metrics,
                   f"Inference duration predictor {version}")
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="ML Cost Estimator Trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    t = subparsers.add_parser("train-training", help="Train training duration predictor")
    t.add_argument("--version", default="V4")

    i = subparsers.add_parser("train-inference", help="Train inference duration predictor")
    i.add_argument("--version", default="V1")

    a = subparsers.add_parser("train-all", help="Train both models")
    a.add_argument("--training-version", default="V4")
    a.add_argument("--inference-version", default="V1")

    args = parser.parse_args()
    session = get_snowflake_session()

    if args.command == "train-training":
        train_training_model(session, args.version)
    elif args.command == "train-inference":
        train_inference_model(session, args.version)
    elif args.command == "train-all":
        train_training_model(session, args.training_version)
        train_inference_model(session, args.inference_version)


if __name__ == "__main__":
    main()
