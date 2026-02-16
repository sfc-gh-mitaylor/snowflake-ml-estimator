"""
Data Generation Module

Generates synthetic classification data for ML benchmarking.
Can generate locally or save directly to Snowflake.

Usage:
    from src.data_generator import generate_classification_data, save_to_snowflake
    
    # Generate locally
    X, y = generate_classification_data(n_samples=100_000, n_features=50)
    
    # Or generate and save to Snowflake in one step
    save_to_snowflake(session, table_name="MY_DATA", n_samples=100_000)
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs
from typing import Tuple, Optional, Literal


def generate_classification_data(
    n_samples: int = 1_000_000,
    n_features: int = 100,
    n_informative: int = 80,
    n_redundant: int = 10,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data for benchmarking.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state,
        shuffle=True,
    )
    return X, y


def generate_regression_data(
    n_samples: int = 1_000_000,
    n_features: int = 100,
    n_informative: int = 80,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data for benchmarking."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
        shuffle=True,
    )
    return X, y


def generate_clustering_data(
    n_samples: int = 1_000_000,
    n_features: int = 100,
    n_clusters: int = 8,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic clustering data for benchmarking."""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
        shuffle=True,
    )
    return X, y


def generate_data(
    task_type: Literal["classification", "regression", "clustering"],
    n_samples: int = 1_000_000,
    n_features: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for the specified task type."""
    if task_type == "classification":
        return generate_classification_data(n_samples, n_features, random_state=random_state)
    elif task_type == "regression":
        return generate_regression_data(n_samples, n_features, random_state=random_state)
    elif task_type == "clustering":
        return generate_clustering_data(n_samples, n_features, random_state=random_state)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def to_dataframe(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Convert feature matrix and target vector to a pandas DataFrame.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        
    Returns:
        DataFrame with columns F0, F1, ..., Fn, TARGET
    """
    feature_cols = [f"F{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["TARGET"] = y
    return df


def get_memory_footprint_gb(X: np.ndarray) -> float:
    """Calculate memory footprint of array in GB."""
    return X.nbytes / (1024 ** 3)


def save_to_snowflake(
    session,
    table_name: str,
    n_samples: int = 1_000_000,
    n_features: int = 100,
    overwrite: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Generate classification data and save directly to Snowflake.
    
    Args:
        session: Active Snowpark session
        table_name: Target table name in Snowflake
        n_samples: Number of rows to generate
        n_features: Number of feature columns
        overwrite: If True, overwrite existing table
        verbose: Print progress messages
        
    Returns:
        Dict with generation stats (shape, memory, table_name)
    """
    if verbose:
        print(f"Generating {n_samples:,} samples with {n_features} features...")
    
    X, y = generate_classification_data(
        n_samples=n_samples,
        n_features=n_features,
    )
    
    memory_gb = get_memory_footprint_gb(X)
    
    if verbose:
        print(f"Data generated. Memory footprint: {memory_gb:.2f} GB")
        print(f"Converting to Snowpark DataFrame...")
    
    df_pandas = to_dataframe(X, y)
    df_snowpark = session.create_dataframe(df_pandas)
    
    write_mode = "overwrite" if overwrite else "append"
    df_snowpark.write.mode(write_mode).save_as_table(table_name)
    
    if verbose:
        print(f"Saved to Snowflake table: {table_name}")
    
    return {
        "table_name": table_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "memory_gb": memory_gb,
        "shape": (n_samples, n_features),
    }
