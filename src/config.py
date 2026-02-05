"""
ML Estimator Configuration Module

Single source of truth for all benchmark configuration.
Import from here rather than hardcoding values elsewhere.

Usage:
    from src.config import BenchmarkConfig
    config = BenchmarkConfig()
    print(config.grid_pools)
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkConfig:
    """
    Centralized configuration for ML benchmark runs.
    
    Using a dataclass gives us:
    - Type hints for IDE support
    - Default values that can be overridden
    - Easy serialization if needed later
    - Immutable-ish structure (we know what's configurable)
    """
    
    max_combinations: Optional[int] = 50
    job_timeout_seconds: int = 1500
    runs_per_combination: int = 5
    
    data_table_name: str = "BENCHMARK_RAW_DATA"
    results_table_name: str = "ML_BENCHMARK_RESULTS"
    
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
