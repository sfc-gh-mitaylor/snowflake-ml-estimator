"""
ML Estimator Configuration.

Credit rates, grid dimensions, and the BenchmarkConfig used by runner.py for status checks.
"""
from dataclasses import dataclass


CREDIT_RATES = {
    "CPU_X64_XS_TEST": 0.06,
    "CPU_X64_S_TEST": 0.12,
    "CPU_X64_M_TEST": 0.24,
    "CPU_X64_SL_TEST": 0.48,
}


def calculate_credits(pool_name: str, duration_seconds: float) -> float:
    rate = CREDIT_RATES.get(pool_name, 0.12)
    return (duration_seconds / 3600) * rate


@dataclass
class BenchmarkConfig:
    results_table_name: str = "ML_ESTIMATOR.PUBLIC.ML_BENCHMARK_RESULTS"
    grid_cols: tuple = (25, 50, 100)
    grid_rows: tuple = (50_000, 200_000, 500_000)
    grid_pools: tuple = (
        "CPU_X64_XS_TEST",
        "CPU_X64_S_TEST",
        "CPU_X64_M_TEST",
        "CPU_X64_SL_TEST",
    )


DEFAULT_CONFIG = BenchmarkConfig()
