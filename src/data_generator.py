"""
Synthetic data generation for ML benchmarking.
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from typing import Tuple, Literal


def generate_classification_data(
    n_samples: int = 1_000_000,
    n_features: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    n_informative = max(2, int(n_features * 0.7))
    n_redundant = max(0, min(int(n_features * 0.1), n_features - n_informative - 1))
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        random_state=random_state,
        shuffle=True,
    )


def generate_regression_data(
    n_samples: int = 1_000_000,
    n_features: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    n_informative = max(2, int(n_features * 0.7))
    return make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=0.1,
        random_state=random_state,
        shuffle=True,
    )


def generate_clustering_data(
    n_samples: int = 1_000_000,
    n_features: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    return make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=8,
        random_state=random_state,
        shuffle=True,
    )


def generate_anomaly_data(
    n_samples: int = 1_000_000,
    n_features: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    n_outliers = int(n_samples * 0.05)
    n_normal = n_samples - n_outliers
    X = np.vstack([rng.randn(n_normal, n_features), rng.uniform(-10, 10, (n_outliers, n_features))])
    y = np.hstack([np.ones(n_normal), -np.ones(n_outliers)])
    shuffle_idx = rng.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]


def generate_data(
    task_type: Literal["classification", "regression", "clustering", "anomaly_detection"],
    n_samples: int = 1_000_000,
    n_features: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    generators = {
        "classification": generate_classification_data,
        "regression": generate_regression_data,
        "clustering": generate_clustering_data,
        "anomaly_detection": generate_anomaly_data,
    }
    if task_type not in generators:
        raise ValueError(f"Unknown task type: {task_type}")
    return generators[task_type](n_samples, n_features, random_state=random_state)
