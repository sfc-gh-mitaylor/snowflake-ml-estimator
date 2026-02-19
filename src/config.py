"""
ML Estimator Configuration Module

Single source of truth for all benchmark configuration.
Import from here rather than hardcoding values elsewhere.

Usage:
    from src.config import BenchmarkConfig, COMPUTE_POOLS, get_valid_pools_for_model
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
from typing import Optional, TypeVar, Callable, List, Dict, Set

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
class ComputePoolSpec:
    """Specification for a compute pool instance family."""
    name: str
    vcpu: int
    memory_gb: int
    credit_rate: float
    enabled: bool = True
    gpu: Optional[str] = None
    notes: str = ""


COMPUTE_POOLS: Dict[str, ComputePoolSpec] = {
    "CPU_X64_XS": ComputePoolSpec(
        name="CPU_X64_XS", vcpu=1, memory_gb=6, credit_rate=0.06, enabled=True,
        notes="Smallest. Fast models, small data."
    ),
    "CPU_X64_S": ComputePoolSpec(
        name="CPU_X64_S", vcpu=3, memory_gb=13, credit_rate=0.12, enabled=True,
        notes="Fast models, medium data."
    ),
    "CPU_X64_M": ComputePoolSpec(
        name="CPU_X64_M", vcpu=6, memory_gb=28, credit_rate=0.24, enabled=True,
        notes="Medium models, standard workloads."
    ),
    "CPU_X64_SL": ComputePoolSpec(
        name="CPU_X64_SL", vcpu=14, memory_gb=54, credit_rate=0.48, enabled=True,
        notes="Larger workloads, ensemble models."
    ),
    "CPU_X64_L": ComputePoolSpec(
        name="CPU_X64_L", vcpu=28, memory_gb=116, credit_rate=0.96, enabled=False,
        notes="Heavy CPU workloads, large ensembles."
    ),
    "HIGHMEM_X64_S": ComputePoolSpec(
        name="HIGHMEM_X64_S", vcpu=6, memory_gb=58, credit_rate=0.36, enabled=False,
        notes="Memory-hungry models (SVC, HDBSCAN)."
    ),
    "HIGHMEM_X64_M": ComputePoolSpec(
        name="HIGHMEM_X64_M", vcpu=28, memory_gb=240, credit_rate=1.44, enabled=False,
        notes="Very large datasets, distance matrices."
    ),
    "GPU_NV_S": ComputePoolSpec(
        name="GPU_NV_S", vcpu=6, memory_gb=27, credit_rate=1.50, enabled=False,
        gpu="1x NVIDIA A10G (24GB)", notes="Entry GPU for deep learning."
    ),
    "GPU_NV_M": ComputePoolSpec(
        name="GPU_NV_M", vcpu=44, memory_gb=178, credit_rate=6.00, enabled=False,
        gpu="4x NVIDIA A10G (24GB)", notes="Serious GPU workloads, LLMs."
    ),
}

MODEL_POOL_AFFINITY: Dict[str, str] = {
    "LinearRegression": "CPU_X64_XS",
    "Ridge": "CPU_X64_XS",
    "Lasso": "CPU_X64_XS",
    "ElasticNet": "CPU_X64_XS",
    "SGDClassifier": "CPU_X64_XS",
    "SGDRegressor": "CPU_X64_XS",
    "GaussianNB": "CPU_X64_XS",
    "DecisionTreeClassifier": "CPU_X64_XS",
    "DecisionTreeRegressor": "CPU_X64_XS",
    "LogisticRegression": "CPU_X64_S",
    "KNeighborsClassifier": "CPU_X64_S",
    "HistGradientBoostingClassifier": "CPU_X64_S",
    "HistGradientBoostingRegressor": "CPU_X64_S",
    "AdaBoostClassifier": "CPU_X64_S",
    "Birch": "CPU_X64_S",
    "KMeans": "CPU_X64_S",
    "MiniBatchKMeans": "CPU_X64_XS",
    "IsolationForest": "CPU_X64_S",
    "RandomForestClassifier": "CPU_X64_M",
    "RandomForestRegressor": "CPU_X64_M",
    "GradientBoostingClassifier": "CPU_X64_M",
    "GradientBoostingRegressor": "CPU_X64_M",
    "XGBClassifier": "CPU_X64_M",
    "XGBRegressor": "CPU_X64_M",
    "LGBMClassifier": "CPU_X64_M",
    "LGBMRegressor": "CPU_X64_M",
    "CatBoostClassifier": "CPU_X64_M",
    "CatBoostRegressor": "CPU_X64_M",
    "ExtraTreesClassifier": "CPU_X64_SL",
    "ExtraTreesRegressor": "CPU_X64_SL",
    "SVC": "HIGHMEM_X64_S",
    "SVR": "HIGHMEM_X64_S",
    "OneClassSVM": "HIGHMEM_X64_S",
    "HDBSCAN": "HIGHMEM_X64_S",
    "SpectralClustering": "HIGHMEM_X64_S",
    "DBSCAN": "CPU_X64_M",
    "AgglomerativeClustering": "CPU_X64_M",
}

POOL_ROW_LIMITS: Dict[str, Dict[str, int]] = {
    "CPU_X64_XS": {
        "svm": 20_000,
        "clustering_distance": 30_000,
        "default": 500_000,
    },
    "CPU_X64_S": {
        "svm": 50_000,
        "clustering_distance": 75_000,
        "default": 1_000_000,
    },
    "CPU_X64_M": {
        "svm": 100_000,
        "clustering_distance": 150_000,
        "default": 2_000_000,
    },
    "CPU_X64_SL": {
        "svm": 150_000,
        "clustering_distance": 250_000,
        "default": 5_000_000,
    },
    "CPU_X64_L": {
        "svm": 200_000,
        "clustering_distance": 400_000,
        "default": 10_000_000,
    },
    "HIGHMEM_X64_S": {
        "svm": 200_000,
        "clustering_distance": 300_000,
        "default": 5_000_000,
    },
    "HIGHMEM_X64_M": {
        "svm": 500_000,
        "clustering_distance": 750_000,
        "default": 20_000_000,
    },
    "GPU_NV_S": {
        "svm": 100_000,
        "clustering_distance": 150_000,
        "default": 2_000_000,
    },
    "GPU_NV_M": {
        "svm": 200_000,
        "clustering_distance": 300_000,
        "default": 10_000_000,
    },
}

SVM_MODELS: Set[str] = {"SVC", "SVR", "OneClassSVM"}
DISTANCE_CLUSTERING_MODELS: Set[str] = {"HDBSCAN", "SpectralClustering", "DBSCAN", "AgglomerativeClustering"}


def get_model_category(model_name: str) -> str:
    """Get the category for row limit purposes."""
    if model_name in SVM_MODELS:
        return "svm"
    if model_name in DISTANCE_CLUSTERING_MODELS:
        return "clustering_distance"
    return "default"


def get_row_limit_for_pool(pool_name: str, model_name: str) -> int:
    """Get max rows a model can handle on a given pool."""
    base_pool = pool_name.replace("_TEST", "")
    limits = POOL_ROW_LIMITS.get(base_pool, POOL_ROW_LIMITS["CPU_X64_M"])
    category = get_model_category(model_name)
    return limits.get(category, limits["default"])


def get_min_pool_for_model(model_name: str) -> str:
    """Get minimum recommended pool for a model."""
    return MODEL_POOL_AFFINITY.get(model_name, "CPU_X64_M")


def get_enabled_pools() -> List[str]:
    """Get list of enabled compute pools."""
    return [name for name, spec in COMPUTE_POOLS.items() if spec.enabled]


def get_valid_pools_for_model(model_name: str, n_rows: int) -> List[str]:
    """
    Get pools that can run this model with this data size.
    
    Returns pools that are:
    1. Enabled
    2. At or above the model's minimum pool size
    3. Have sufficient row capacity for the model type
    """
    min_pool = get_min_pool_for_model(model_name)
    min_pool_base = min_pool.replace("_TEST", "")
    
    pool_order = [
        "CPU_X64_XS", "CPU_X64_S", "CPU_X64_M", "CPU_X64_SL", "CPU_X64_L",
        "HIGHMEM_X64_S", "HIGHMEM_X64_M", "GPU_NV_S", "GPU_NV_M"
    ]
    
    try:
        min_idx = pool_order.index(min_pool_base)
    except ValueError:
        min_idx = 0
    
    valid = []
    for pool_name, spec in COMPUTE_POOLS.items():
        if not spec.enabled:
            continue
        
        base_name = pool_name.replace("_TEST", "")
        try:
            pool_idx = pool_order.index(base_name)
        except ValueError:
            continue
            
        if pool_idx < min_idx:
            continue
        
        row_limit = get_row_limit_for_pool(pool_name, model_name)
        if n_rows > row_limit:
            continue
            
        valid.append(pool_name)
    
    return valid


def is_valid_combination(model_name: str, pool_name: str, n_rows: int) -> bool:
    """Check if a model/pool/rows combination is valid."""
    row_limit = get_row_limit_for_pool(pool_name, model_name)
    return n_rows <= row_limit


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
        default_factory=lambda: env_or_default("ML_RESULTS_TABLE", "ML_ESTIMATOR.PUBLIC.ML_BENCHMARK_RESULTS")
    )
    
    num_total_features: int = 100
    
    grid_cols: tuple = (25, 50, 100)
    grid_rows: tuple = (50_000, 200_000, 500_000)
    
    @property
    def grid_pools(self) -> tuple:
        """Get enabled pools, with _TEST suffix for actual pool names."""
        return tuple(f"{name}_TEST" for name in get_enabled_pools())
    
    model_row_limits: dict = field(default_factory=lambda: {
        "SVC": 100_000,
        "SVR": 100_000,
        "OneClassSVM": 50_000,
        "HDBSCAN": 100_000,
        "SpectralClustering": 10_000,
        "AgglomerativeClustering": 50_000,
        "DBSCAN": 100_000,
    })
    
    results_schema: tuple = (
        "MODEL_CLASS",
        "TASK_TYPE",
        "COMPUTE_POOL",
        "RUN_ID",
        "N_COLS_SAMPLED",
        "N_ROWS_SAMPLED",
        "DURATION_SECONDS",
        "ESTIMATED_CREDITS",
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


CREDIT_RATES: Dict[str, float] = {
    f"{name}_TEST": spec.credit_rate 
    for name, spec in COMPUTE_POOLS.items()
}
CREDIT_RATES.update({
    name: spec.credit_rate 
    for name, spec in COMPUTE_POOLS.items()
})


def calculate_credits(pool_name: str, duration_seconds: float) -> float:
    """Calculate credit cost from duration and pool type."""
    rate = CREDIT_RATES.get(pool_name, 0.12)
    return (duration_seconds / 3600) * rate


def print_pool_summary():
    """Print a summary of all compute pools."""
    print("\n" + "=" * 80)
    print("COMPUTE POOL CONFIGURATION")
    print("=" * 80)
    print(f"{'Pool':<20} {'vCPU':>5} {'RAM':>8} {'Credit/hr':>10} {'Status':<10} {'GPU':<20}")
    print("-" * 80)
    for name, spec in COMPUTE_POOLS.items():
        status = "✅ ON" if spec.enabled else "🔴 OFF"
        gpu = spec.gpu or "-"
        print(f"{name:<20} {spec.vcpu:>5} {spec.memory_gb:>6}GB {spec.credit_rate:>10.2f} {status:<10} {gpu:<20}")
    print("=" * 80)
    print(f"Enabled pools: {', '.join(get_enabled_pools())}")
    print()


if __name__ == "__main__":
    print_pool_summary()
