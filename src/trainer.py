"""
ML Cost Estimator Trainer - Trains and registers the cost prediction model.

Usage:
    python -m src.trainer train
    python -m src.trainer evaluate
"""
import argparse
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .config import DEFAULT_CONFIG, CREDIT_RATES


class CostEstimator:
    """ML model that predicts training duration and credit cost."""
    
    def __init__(self):
        self.duration_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        )
        self.le_model = LabelEncoder()
        self.le_pool = LabelEncoder()
        self.le_task = LabelEncoder()
        self.is_fitted = False
        self.metrics = {}
    
    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the estimator on benchmark results."""
        df = df[df["DURATION_SECONDS"] > 0].copy()
        
        if len(df) < 10:
            raise ValueError(f"Need at least 10 successful runs, got {len(df)}")
        
        df["MODEL_ENCODED"] = self.le_model.fit_transform(df["MODEL_CLASS"])
        df["POOL_ENCODED"] = self.le_pool.fit_transform(df["COMPUTE_POOL"])
        df["TASK_ENCODED"] = self.le_task.fit_transform(df["TASK_TYPE"])
        
        feature_cols = ["MODEL_ENCODED", "POOL_ENCODED", "TASK_ENCODED", 
                        "N_COLS_SAMPLED", "N_ROWS_SAMPLED"]
        X = df[feature_cols]
        y = df["DURATION_SECONDS"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.duration_model.fit(X_train, y_train)
        self.is_fitted = True
        
        y_pred = self.duration_model.predict(X_test)
        self.metrics = {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        
        return self.metrics
    
    def predict(
        self,
        model_class: str,
        task_type: str,
        compute_pool: str,
        n_cols: int,
        n_rows: int,
    ) -> Dict[str, Any]:
        """Predict duration and credit cost for a configuration."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        try:
            model_enc = self.le_model.transform([model_class])[0]
            pool_enc = self.le_pool.transform([compute_pool])[0]
            task_enc = self.le_task.transform([task_type])[0]
        except ValueError as e:
            return {
                "error": f"Unknown value: {e}",
                "duration_seconds": None,
                "estimated_credits": None,
            }
        
        features = np.array([[model_enc, pool_enc, task_enc, n_cols, n_rows]])
        duration = self.duration_model.predict(features)[0]
        
        credit_rate = CREDIT_RATES.get(compute_pool, 0.12)
        credits = (duration / 3600) * credit_rate
        
        return {
            "duration_seconds": round(duration, 2),
            "estimated_credits": round(credits, 6),
            "credit_rate_per_hour": credit_rate,
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for a batch of configurations."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        results = []
        for _, row in df.iterrows():
            pred = self.predict(
                model_class=row["MODEL_CLASS"],
                task_type=row["TASK_TYPE"],
                compute_pool=row["COMPUTE_POOL"],
                n_cols=row["N_COLS_SAMPLED"],
                n_rows=row["N_ROWS_SAMPLED"],
            )
            results.append(pred)
        
        return pd.DataFrame(results)
    
    def get_known_values(self) -> Dict[str, list]:
        """Return lists of known model/pool/task values."""
        return {
            "models": list(self.le_model.classes_),
            "pools": list(self.le_pool.classes_),
            "task_types": list(self.le_task.classes_),
        }


def get_snowflake_session():
    from snowflake.snowpark import Session
    conn_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
    return Session.builder.config("connection_name", conn_name).create()


def load_benchmark_data(session) -> pd.DataFrame:
    """Load benchmark results from Snowflake."""
    return session.table(DEFAULT_CONFIG.results_table_name).to_pandas()


def train_and_register(session, model_name: str = "ML_COST_ESTIMATOR", version: str = "v1"):
    """Train model and register to Snowflake Model Registry."""
    from snowflake.ml.registry import Registry
    
    print("Loading benchmark data...")
    df = load_benchmark_data(session)
    print(f"Loaded {len(df)} benchmark results")
    
    print("\nTraining cost estimator...")
    estimator = CostEstimator()
    metrics = estimator.fit(df)
    
    print("\nModel Metrics:")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAE:  {metrics['mae']:.2f}s")
    print(f"  RMSE: {metrics['rmse']:.2f}s")
    print(f"  Train samples: {metrics['n_train']}")
    print(f"  Test samples:  {metrics['n_test']}")
    
    if metrics['r2'] < 0.5:
        print("\nWARNING: R² is low. Model may not be reliable.")
        print("Consider running more benchmarks to improve coverage.")
    
    print(f"\nRegistering model as {model_name}/{version}...")
    
    sample_df = pd.DataFrame({
        "MODEL_ENCODED": [0],
        "POOL_ENCODED": [1],
        "TASK_ENCODED": [0],
        "N_COLS_SAMPLED": [50],
        "N_ROWS_SAMPLED": [100000],
    })
    
    registry = Registry(session, database_name="ML_ESTIMATOR", schema_name="PUBLIC")
    mv = registry.log_model(
        model=estimator.duration_model,
        model_name=model_name,
        version_name=version,
        sample_input_data=sample_df,
        metrics=metrics,
        comment="ML training cost estimator - duration prediction model",
    )
    
    print(f"Model registered: {mv.model_name}/{mv.version_name}")
    return estimator, metrics


def evaluate_model(session):
    """Load and evaluate the registered model."""
    from snowflake.ml.registry import Registry
    
    registry = Registry(session, database_name="ML_ESTIMATOR", schema_name="PUBLIC")
    model = registry.get_model("ML_COST_ESTIMATOR").default
    
    test_df = pd.DataFrame([
        {"MODEL_CLASS": "XGBClassifier", "TASK_TYPE": "classification", "COMPUTE_POOL": "CPU_X64_S_TEST", "N_COLS_SAMPLED": 50, "N_ROWS_SAMPLED": 100000},
        {"MODEL_CLASS": "RandomForestRegressor", "TASK_TYPE": "regression", "COMPUTE_POOL": "CPU_X64_M_TEST", "N_COLS_SAMPLED": 75, "N_ROWS_SAMPLED": 500000},
        {"MODEL_CLASS": "KMeans", "TASK_TYPE": "clustering", "COMPUTE_POOL": "CPU_X64_XS_TEST", "N_COLS_SAMPLED": 25, "N_ROWS_SAMPLED": 200000},
    ])
    
    print("\nTest Predictions:")
    print("-" * 70)
    results = model.run(test_df)
    print(results)


def main():
    parser = argparse.ArgumentParser(description="ML Cost Estimator Trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    train_parser = subparsers.add_parser("train", help="Train and register model")
    train_parser.add_argument("--name", default="ML_COST_ESTIMATOR")
    train_parser.add_argument("--version", default="v1")
    
    subparsers.add_parser("evaluate", help="Evaluate registered model")
    
    args = parser.parse_args()
    session = get_snowflake_session()
    
    session.sql("USE DATABASE ML_ESTIMATOR").collect()
    session.sql("USE SCHEMA PUBLIC").collect()
    
    if args.command == "train":
        train_and_register(session, args.name, args.version)
    elif args.command == "evaluate":
        evaluate_model(session)


if __name__ == "__main__":
    main()
