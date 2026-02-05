"""
ML Estimator Configuration Module

Single source of truth for all benchmark configuration.
Import from here rather than hardcoding values elsewhere.

Usage:
    from src.config import BenchmarkConfig
    config = BenchmarkConfig()
    print(config.grid_pools)

Environment Variable Overrides:
    ML_MAX_COMBINATIONS=100      # Override max combinations
    ML_JOB_TIMEOUT=3000          # Override timeout in seconds
    ML_RUNS_PER_COMBO=3          # Override runs per combination
    ML_DATA_TABLE=MY_TABLE       # Override data table name
    ML_RESULTS_TABLE=MY_RESULTS  # Override results table name
"""
import os
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Callable

T = TypeVar("T")


def env_or_default(key: str, default: T, parser: Callable[[str], T] = str) -> T:
    """
    Get value from environment variable, or fall back to default.
    
    Args:
        key: Environment variable name
        default: Default value if env var not set
        parser: Function to convert string to desired type
        
    Returns:
        Parsed env var value, or default
        
    Example:
        timeout = env_or_default("ML_TIMEOUT", 1500, int)
    """
    value = os.environ.get(key)
    if value is None:
        return default
    if value.lower() in ("none", "null", ""):
        return None
    try:
        return parser(value)
    except (ValueError, TypeError):
        return default


@dataclass
class BenchmarkConfig:
    """
    Centralized configuration for ML benchmark runs.
    
    Using a dataclass gives us:
    - Type hints for IDE support
    - Default values that can be overridden
    - Easy serialization if needed later
    - Immutable-ish structure (we know what's configurable)
    
    Environment variables override defaults when set.
    """
    
    max_combinations: Optional[int] = field(
        default_factory=lambda: env_or_default("ML_MAX_COMBINATIONS", 50, int)
    )
    job_timeout_seconds: int = field(
        default_factory=lambda: env_or_default("ML_JOB_TIMEOUT", 1500, int)
    )
    runs_per_combination: int = field(
        default_factory=lambda: env_or_default("ML_RUNS_PER_COMBO", 5, int)
    )
    
    data_table_name: str = field(
        default_factory=lambda: env_or_default("ML_DATA_TABLE", "BENCHMARK_RAW_DATA")
    )
    results_table_name: str = field(
        default_factory=lambda: env_or_default("ML_RESULTS_TABLE", "ML_BENCHMARK_RESULTS")
    )
    
    num_total_features: int = 100
    
    grid_cols: tuple = (25, 50, 75, 100)
    grid_rows: tuple = (50_000, 250_000, 450_000, 650_000, 850_000)
    grid_pools: tuple = (
        "CPU_X64_XS_TEST",
        "CPU_X64_S_TEST",
        "CPU_X64_M_TEST",
        "CPU_X64_SL_TEST",
    )
    
    model_row_limits: dict = field(default_factory=lambda: {
        "SVC": 100_000,
    })
    
    results_schema: tuple = (
        "MODEL_CLASS",
        "COMPUTE_POOL",
        "RUN_ID",
        "N_COLS_SAMPLED",
        "N_ROWS_SAMPLED",
        "DURATION_SECONDS",
        "START_TIMESTAMP",
    )
    
    def get_model_row_limit(self, model_name: str) -> Optional[int]:
        """Get row limit for a specific model, or None if unlimited."""
        return self.model_row_limits.get(model_name)
    
    def calculate_grid_size(self, n_estimators: int) -> dict:
        """Calculate total grid statistics."""
        full_grid = n_estimators * len(self.grid_cols) * len(self.grid_rows) * len(self.grid_pools)
        capped = min(full_grid, self.max_combinations) if self.max_combinations else full_grid
        return {
            "full_grid_size": full_grid,
            "capped_size": capped,
            "total_executions": capped * self.runs_per_combination,
        }


DEFAULT_CONFIG = BenchmarkConfig()
