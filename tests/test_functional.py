"""
Functional tests - verify model predictions match expected behavior.

These tests verify:
1. A freshly trained model produces consistent predictions
2. Predictions are within reasonable bounds
3. Model can be serialized and deserialized correctly

Run with: pytest tests/test_functional.py -v
"""
import pytest
import pandas as pd
import numpy as np
import pickle
import tempfile
from pathlib import Path

from src.trainer import CostEstimator
from src.config import CREDIT_RATES


@pytest.fixture
def realistic_benchmark_data():
    """
    Generate benchmark data that mimics real training patterns:
    - Larger data = longer training time
    - More features = longer training time  
    - Faster pools = shorter training time
    - Different models have different base speeds
    """
    np.random.seed(42)
    
    records = []
    
    model_base_times = {
        "RandomForestClassifier": 50,
        "GradientBoostingClassifier": 200,
        "LogisticRegression": 5,
        "XGBClassifier": 30,
    }
    
    pool_multipliers = {
        "CPU_X64_XS_TEST": 2.0,
        "CPU_X64_S_TEST": 1.0,
        "CPU_X64_M_TEST": 0.5,
        "CPU_X64_SL_TEST": 0.25,
    }
    
    for model, base_time in model_base_times.items():
        for pool, pool_mult in pool_multipliers.items():
            for n_cols in [25, 50, 75, 100]:
                for n_rows in [50000, 250000, 500000, 750000]:
                    col_factor = n_cols / 50
                    row_factor = n_rows / 250000
                    
                    duration = base_time * pool_mult * col_factor * row_factor
                    duration *= np.random.uniform(0.8, 1.2)
                    
                    for run_id in range(3):
                        records.append({
                            "MODEL_CLASS": model,
                            "COMPUTE_POOL": pool,
                            "TASK_TYPE": "classification",
                            "N_COLS_SAMPLED": n_cols,
                            "N_ROWS_SAMPLED": n_rows,
                            "DURATION_SECONDS": duration * np.random.uniform(0.95, 1.05),
                        })
    
    return pd.DataFrame(records)


class TestPredictionConsistency:
    """Test that model predictions are consistent and reproducible."""
    
    def test_same_input_same_output(self, realistic_benchmark_data):
        """Same inputs should produce same predictions."""
        estimator = CostEstimator()
        estimator.fit(realistic_benchmark_data)
        
        pred1 = estimator.predict(
            "RandomForestClassifier", "classification", 
            "CPU_X64_S_TEST", 50, 250000
        )
        pred2 = estimator.predict(
            "RandomForestClassifier", "classification",
            "CPU_X64_S_TEST", 50, 250000
        )
        
        assert pred1["duration_seconds"] == pred2["duration_seconds"]
        assert pred1["estimated_credits"] == pred2["estimated_credits"]
    
    def test_two_models_same_data_same_predictions(self, realistic_benchmark_data):
        """Two models trained on same data should predict similarly."""
        est1 = CostEstimator()
        est1.fit(realistic_benchmark_data)
        
        est2 = CostEstimator()
        est2.fit(realistic_benchmark_data)
        
        pred1 = est1.predict(
            "XGBClassifier", "classification",
            "CPU_X64_M_TEST", 75, 500000
        )
        pred2 = est2.predict(
            "XGBClassifier", "classification", 
            "CPU_X64_M_TEST", 75, 500000
        )
        
        assert pred1["duration_seconds"] == pred2["duration_seconds"]


class TestPredictionReasonableness:
    """Test that predictions fall within reasonable bounds."""
    
    def test_larger_data_longer_time(self, realistic_benchmark_data):
        """More rows should predict longer training time."""
        estimator = CostEstimator()
        estimator.fit(realistic_benchmark_data)
        
        small = estimator.predict(
            "RandomForestClassifier", "classification",
            "CPU_X64_S_TEST", 50, 50000
        )
        large = estimator.predict(
            "RandomForestClassifier", "classification",
            "CPU_X64_S_TEST", 50, 750000
        )
        
        assert large["duration_seconds"] > small["duration_seconds"]
    
    def test_faster_pool_shorter_time(self, realistic_benchmark_data):
        """Larger compute pool should predict shorter training time."""
        estimator = CostEstimator()
        estimator.fit(realistic_benchmark_data)
        
        slow_pool = estimator.predict(
            "GradientBoostingClassifier", "classification",
            "CPU_X64_XS_TEST", 50, 250000
        )
        fast_pool = estimator.predict(
            "GradientBoostingClassifier", "classification",
            "CPU_X64_SL_TEST", 50, 250000
        )
        
        assert fast_pool["duration_seconds"] < slow_pool["duration_seconds"]
    
    def test_predictions_positive(self, realistic_benchmark_data):
        """All predictions should be positive."""
        estimator = CostEstimator()
        estimator.fit(realistic_benchmark_data)
        
        for model in ["RandomForestClassifier", "XGBClassifier", "LogisticRegression"]:
            for pool in ["CPU_X64_XS_TEST", "CPU_X64_S_TEST"]:
                result = estimator.predict(
                    model, "classification", pool, 50, 100000
                )
                assert result["duration_seconds"] > 0
                assert result["estimated_credits"] > 0


class TestModelSerialization:
    """Test that model can be pickled and unpickled correctly."""
    
    def test_pickle_roundtrip(self, realistic_benchmark_data):
        """Model should survive pickle serialization."""
        estimator = CostEstimator()
        estimator.fit(realistic_benchmark_data)
        
        pred_before = estimator.predict(
            "XGBClassifier", "classification",
            "CPU_X64_S_TEST", 50, 250000
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(estimator, f)
            temp_path = f.name
        
        with open(temp_path, "rb") as f:
            loaded_estimator = pickle.load(f)
        
        pred_after = loaded_estimator.predict(
            "XGBClassifier", "classification",
            "CPU_X64_S_TEST", 50, 250000
        )
        
        assert pred_before["duration_seconds"] == pred_after["duration_seconds"]
        
        Path(temp_path).unlink()
    
    def test_loaded_model_has_correct_state(self, realistic_benchmark_data):
        """Loaded model should have all attributes intact."""
        estimator = CostEstimator()
        metrics = estimator.fit(realistic_benchmark_data)
        known_before = estimator.get_known_values()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(estimator, f)
            temp_path = f.name
        
        with open(temp_path, "rb") as f:
            loaded = pickle.load(f)
        
        assert loaded.is_fitted is True
        assert loaded.metrics == metrics
        assert loaded.get_known_values() == known_before
        
        Path(temp_path).unlink()


class TestMetricsQuality:
    """Test that model achieves reasonable accuracy on synthetic data."""
    
    def test_r2_above_threshold(self, realistic_benchmark_data):
        """Model should achieve reasonable R² on well-structured data."""
        estimator = CostEstimator()
        metrics = estimator.fit(realistic_benchmark_data)
        
        assert metrics["r2"] > 0.5, f"R² too low: {metrics['r2']}"
    
    def test_mae_reasonable(self, realistic_benchmark_data):
        """MAE should be reasonable relative to mean duration."""
        estimator = CostEstimator()
        metrics = estimator.fit(realistic_benchmark_data)
        
        mean_duration = realistic_benchmark_data["DURATION_SECONDS"].mean()
        relative_mae = metrics["mae"] / mean_duration
        
        assert relative_mae < 0.5, f"Relative MAE too high: {relative_mae}"
