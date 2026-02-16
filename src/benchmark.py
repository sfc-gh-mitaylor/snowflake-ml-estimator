"""
Benchmark Runner Module

Orchestrates ML model benchmarking across compute pools, data dimensions,
and estimators. Manages grid generation, execution tracking, and results.

Usage:
    from src.benchmark import BenchmarkConfig, BenchmarkGrid
    
    config = BenchmarkConfig()
    grid = BenchmarkGrid(config)
    
    # Get all combinations to run
    combos = grid.get_combinations()
    
    # Filter out already-tested combinations
    remaining = grid.filter_tested(combos, tested_set)
"""
import itertools
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import env_or_default


@dataclass
class BenchmarkConfig:
    """
    Centralized benchmark configuration.
    
    All settings in one place - reads from environment variables
    with sensible defaults. No more hunting through notebook cells.
    """
    max_combinations: Optional[int] = field(
        default_factory=lambda: int(env_or_default("ML_MAX_COMBINATIONS", "50"))
        if env_or_default("ML_MAX_COMBINATIONS", None) else 50
    )
    job_timeout_seconds: int = field(
        default_factory=lambda: int(env_or_default("ML_JOB_TIMEOUT", "1500"))
    )
    runs_per_combination: int = field(
        default_factory=lambda: int(env_or_default("ML_RUNS_PER_COMBO", "5"))
    )
    
    data_table: str = field(
        default_factory=lambda: env_or_default("ML_DATA_TABLE", "BENCHMARK_RAW_DATA")
    )
    results_table: str = field(
        default_factory=lambda: env_or_default("ML_RESULTS_TABLE", "ML_BENCHMARK_RESULTS")
    )
    
    num_total_features: int = 100
    
    grid_cols: List[int] = field(
        default_factory=lambda: [25, 50, 75, 100]
    )
    grid_rows: List[int] = field(
        default_factory=lambda: [50_000, 250_000, 450_000, 650_000, 850_000]
    )
    grid_pools: List[str] = field(
        default_factory=lambda: [
            "CPU_X64_XS_TEST",
            "CPU_X64_S_TEST", 
            "CPU_X64_M_TEST",
            "CPU_X64_SL_TEST"
        ]
    )
    
    model_row_limits: Dict[str, int] = field(
        default_factory=lambda: {
            "SVC": 100_000,
        }
    )
    
    results_schema: List[str] = field(
        default_factory=lambda: [
            "MODEL_CLASS",
            "COMPUTE_POOL",
            "RUN_ID",
            "N_COLS_SAMPLED",
            "N_ROWS_SAMPLED",
            "DURATION_SECONDS",
            "START_TIMESTAMP"
        ]
    )
    
    def get_row_limit(self, model_name: str) -> Optional[int]:
        """Get row limit for a model, or None if unlimited."""
        return self.model_row_limits.get(model_name)
    
    def summary(self) -> str:
        """Return human-readable config summary."""
        n_combos = len(self.grid_cols) * len(self.grid_rows) * len(self.grid_pools)
        effective = min(n_combos, self.max_combinations) if self.max_combinations else n_combos
        
        lines = [
            f"Data table: {self.data_table}",
            f"Results table: {self.results_table}",
            f"Grid: {len(self.grid_cols)} cols × {len(self.grid_rows)} rows × {len(self.grid_pools)} pools",
            f"Base combinations: {n_combos:,}",
            f"Max combinations: {self.max_combinations or 'Unlimited'}",
            f"Effective combinations: {effective:,}",
            f"Runs per combo: {self.runs_per_combination}",
            f"Total executions: {effective * self.runs_per_combination:,}",
            f"Job timeout: {self.job_timeout_seconds}s",
        ]
        return "\n".join(lines)


Combination = Tuple[str, str, int, int]


class BenchmarkGrid:
    """
    Generates and manages benchmark parameter combinations.
    
    A combination is: (model_name, compute_pool, n_cols, n_rows)
    """
    
    def __init__(self, config: BenchmarkConfig, model_names: List[str]):
        """
        Initialize grid with config and available models.
        
        Args:
            config: BenchmarkConfig instance
            model_names: List of model names to include in grid
        """
        self.config = config
        self.model_names = model_names
    
    def get_all_combinations(self) -> List[Combination]:
        """
        Generate all possible parameter combinations.
        
        Returns:
            List of (model, pool, cols, rows) tuples
        """
        return list(itertools.product(
            self.model_names,
            self.config.grid_pools,
            self.config.grid_cols,
            self.config.grid_rows
        ))
    
    def apply_row_limits(self, combinations: List[Combination]) -> List[Combination]:
        """
        Filter combinations that exceed model-specific row limits.
        
        Args:
            combinations: List of combinations to filter
            
        Returns:
            Filtered list with row-limited models capped
        """
        filtered = []
        for model, pool, cols, rows in combinations:
            limit = self.config.get_row_limit(model)
            if limit is None or rows <= limit:
                filtered.append((model, pool, cols, rows))
        return filtered
    
    def filter_tested(
        self, 
        combinations: List[Combination], 
        tested: Set[Combination]
    ) -> List[Combination]:
        """
        Remove already-tested combinations.
        
        Args:
            combinations: Full list of combinations
            tested: Set of already-tested (model, pool, cols, rows) tuples
            
        Returns:
            List of untested combinations
        """
        return [c for c in combinations if c not in tested]
    
    def cap_combinations(self, combinations: List[Combination]) -> List[Combination]:
        """
        Apply max_combinations limit from config.
        
        Args:
            combinations: List to cap
            
        Returns:
            Capped list (or original if no limit)
        """
        if self.config.max_combinations is None:
            return combinations
        return combinations[:self.config.max_combinations]
    
    def get_runnable(self, tested: Optional[Set[Combination]] = None) -> List[Combination]:
        """
        Get final list of combinations to run.
        
        Applies all filters: row limits, tested exclusion, cap.
        
        Args:
            tested: Optional set of already-tested combinations
            
        Returns:
            Final list of combinations ready to execute
        """
        combos = self.get_all_combinations()
        combos = self.apply_row_limits(combos)
        
        if tested:
            combos = self.filter_tested(combos, tested)
        
        combos = self.cap_combinations(combos)
        return combos
    
    def stats(self, tested: Optional[Set[Combination]] = None) -> Dict[str, Any]:
        """
        Get grid statistics.
        
        Args:
            tested: Optional set of already-tested combinations
            
        Returns:
            Dict with counts at each filtering stage
        """
        all_combos = self.get_all_combinations()
        after_limits = self.apply_row_limits(all_combos)
        
        after_tested = after_limits
        if tested:
            after_tested = self.filter_tested(after_limits, tested)
        
        after_cap = self.cap_combinations(after_tested)
        
        return {
            "total_possible": len(all_combos),
            "after_row_limits": len(after_limits),
            "already_tested": len(tested) if tested else 0,
            "after_exclusions": len(after_tested),
            "after_cap": len(after_cap),
            "final_runnable": len(after_cap),
            "total_executions": len(after_cap) * self.config.runs_per_combination,
        }


@dataclass 
class BenchmarkResult:
    """
    Single benchmark execution result.
    
    Matches the RESULTS_SCHEMA for easy table insertion.
    """
    model_class: str
    compute_pool: str
    run_id: int
    n_cols_sampled: int
    n_rows_sampled: int
    duration_seconds: float
    start_timestamp: float
    
    def to_tuple(self) -> tuple:
        """Convert to tuple matching RESULTS_SCHEMA order."""
        return (
            self.model_class,
            self.compute_pool,
            self.run_id,
            self.n_cols_sampled,
            self.n_rows_sampled,
            self.duration_seconds,
            self.start_timestamp
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for DataFrame creation."""
        return {
            "MODEL_CLASS": self.model_class,
            "COMPUTE_POOL": self.compute_pool,
            "RUN_ID": self.run_id,
            "N_COLS_SAMPLED": self.n_cols_sampled,
            "N_ROWS_SAMPLED": self.n_rows_sampled,
            "DURATION_SECONDS": self.duration_seconds,
            "START_TIMESTAMP": self.start_timestamp,
        }
