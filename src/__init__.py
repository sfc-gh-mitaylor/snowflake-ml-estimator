# Snowflake ML Estimator - Shared Utilities
"""
Shared Python utilities for the ML Estimator project.

Usage:
    from src import BenchmarkConfig, DEFAULT_CONFIG
    
    # Use defaults
    print(DEFAULT_CONFIG.grid_pools)
    
    # Or customize
    my_config = BenchmarkConfig(max_combinations=100)
"""

__version__ = "0.1.0"

from src.config import BenchmarkConfig, DEFAULT_CONFIG
from src.data_generator import (
    generate_classification_data,
    save_to_snowflake,
    to_dataframe,
    get_memory_footprint_gb,
)
from src.estimators import EstimatorFactory, DEFAULT_FACTORY
from src.benchmark import BenchmarkConfig as GridConfig, BenchmarkGrid, BenchmarkResult

__all__ = [
    "BenchmarkConfig",
    "DEFAULT_CONFIG",
    "generate_classification_data",
    "save_to_snowflake",
    "to_dataframe",
    "get_memory_footprint_gb",
    "EstimatorFactory",
    "DEFAULT_FACTORY",
    "GridConfig",
    "BenchmarkGrid",
    "BenchmarkResult",
    "__version__",
]
