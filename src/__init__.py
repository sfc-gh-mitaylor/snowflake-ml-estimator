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

__all__ = ["BenchmarkConfig", "DEFAULT_CONFIG", "__version__"]
