"""
Unit tests for the CostEstimator class.

Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from src.trainer import CostEstimator
from src.config import CREDIT_RATES


@pytest.fixture
def sample_benchmark_data():
    """Generate synthetic benchmark data for testing."""
    np.random.seed(42)
    n_samples = 200
    
    models = ["RandomForestClassifier", "XGBClassifier", "LogisticRegression"]
    pools = ["CPU_X64_XS_TEST", "CPU_X64_S_TEST", "CPU_X64_M_TEST"]
    tasks = ["classification"]
    
    data = {
        "MODEL_CLASS": np.random.choice(models, n_samples),
        "COMPUTE_POOL": np.random.choice(pools, n_samples),
        "TASK_TYPE": np.random.choice(tasks, n_samples),
        "N_COLS_SAMPLED": np.random.choice([25, 50, 75, 100], n_samples),
        "N_ROWS_SAMPLED": np.random.choice([50000, 250000, 500000], n_samples),
        "DURATION_SECONDS": np.random.exponential(100, n_samples) + 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def trained_estimator(sample_benchmark_data):
    """Return a fitted CostEstimator."""
    estimator = CostEstimator()
    estimator.fit(sample_benchmark_data)
    return estimator


class TestCostEstimatorInit:
    def test_initial_state(self):
        estimator = CostEstimator()
        assert estimator.is_fitted is False
        assert estimator.metrics == {}
    
    def test_model_params(self):
        estimator = CostEstimator()
        assert estimator.duration_model.n_estimators == 100
        assert estimator.duration_model.random_state == 42


class TestCostEstimatorFit:
    def test_fit_returns_metrics(self, sample_benchmark_data):
        estimator = CostEstimator()
        metrics = estimator.fit(sample_benchmark_data)
        
        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "n_train" in metrics
        assert "n_test" in metrics
    
    def test_fit_sets_fitted_flag(self, sample_benchmark_data):
        estimator = CostEstimator()
        estimator.fit(sample_benchmark_data)
        assert estimator.is_fitted is True
    
    def test_fit_filters_failed_runs(self, sample_benchmark_data):
        df = sample_benchmark_data.copy()
        df.loc[0:10, "DURATION_SECONDS"] = -1
        
        estimator = CostEstimator()
        metrics = estimator.fit(df)
        
        total_valid = len(df[df["DURATION_SECONDS"] > 0])
        assert metrics["n_train"] + metrics["n_test"] == total_valid
    
    def test_fit_raises_on_insufficient_data(self):
        df = pd.DataFrame({
            "MODEL_CLASS": ["A", "B"],
            "COMPUTE_POOL": ["P1", "P2"],
            "TASK_TYPE": ["classification", "classification"],
            "N_COLS_SAMPLED": [50, 50],
            "N_ROWS_SAMPLED": [1000, 1000],
            "DURATION_SECONDS": [10, 20],
        })
        
        estimator = CostEstimator()
        with pytest.raises(ValueError, match="Need at least 10"):
            estimator.fit(df)


class TestCostEstimatorPredict:
    def test_predict_returns_duration_and_credits(self, trained_estimator):
        result = trained_estimator.predict(
            model_class="RandomForestClassifier",
            task_type="classification",
            compute_pool="CPU_X64_S_TEST",
            n_cols=50,
            n_rows=100000,
        )
        
        assert "duration_seconds" in result
        assert "estimated_credits" in result
        assert "credit_rate_per_hour" in result
        assert result["duration_seconds"] > 0
        assert result["estimated_credits"] > 0
    
    def test_predict_unknown_model_returns_error(self, trained_estimator):
        result = trained_estimator.predict(
            model_class="UnknownModel",
            task_type="classification",
            compute_pool="CPU_X64_S_TEST",
            n_cols=50,
            n_rows=100000,
        )
        
        assert "error" in result
        assert result["duration_seconds"] is None
    
    def test_predict_before_fit_raises(self):
        estimator = CostEstimator()
        with pytest.raises(RuntimeError, match="not fitted"):
            estimator.predict("Model", "task", "pool", 50, 1000)
    
    def test_credit_calculation_uses_correct_rate(self, trained_estimator):
        result = trained_estimator.predict(
            model_class="RandomForestClassifier",
            task_type="classification",
            compute_pool="CPU_X64_M_TEST",
            n_cols=50,
            n_rows=100000,
        )
        
        expected_rate = CREDIT_RATES["CPU_X64_M_TEST"]
        assert result["credit_rate_per_hour"] == expected_rate
        
        expected_credits = (result["duration_seconds"] / 3600) * expected_rate
        assert abs(result["estimated_credits"] - expected_credits) < 0.0001


class TestCostEstimatorKnownValues:
    def test_get_known_values_after_fit(self, trained_estimator):
        known = trained_estimator.get_known_values()
        
        assert "models" in known
        assert "pools" in known
        assert "task_types" in known
        assert len(known["models"]) > 0
        assert len(known["pools"]) > 0


class TestCreditRates:
    def test_all_test_pools_have_rates(self):
        test_pools = [
            "CPU_X64_XS_TEST",
            "CPU_X64_S_TEST", 
            "CPU_X64_M_TEST",
            "CPU_X64_SL_TEST",
        ]
        for pool in test_pools:
            assert pool in CREDIT_RATES, f"Missing rate for {pool}"
            assert CREDIT_RATES[pool] > 0
