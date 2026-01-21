# Snowflake ML Runtime Estimator

A benchmarking system that measures ML model training times across different Snowflake compute pools and data sizes, then trains a meta-model to predict runtimes for new configurations.

## 🎯 Purpose

When running ML workloads on Snowflake, it's hard to know:
- How long will my model take to train?
- Which compute pool should I use?
- How does training time scale with data size?

This project collects empirical benchmark data and builds a predictor to answer these questions.

## 📁 Project Structure

```
snowflake-ml-estimator/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
│
├── notebooks/               # Jupyter/Snowflake notebooks
│   └── benchmark_runner.ipynb   # Main benchmarking notebook
│
├── streamlit/               # Streamlit dashboard app
│   ├── app.py              # Main Streamlit application
│   └── config.py           # App configuration
│
└── src/                    # Shared Python utilities
    └── __init__.py
```

## 🚀 Quick Start

### Prerequisites

- Snowflake account with:
  - Access to create compute pools
  - Snowpark ML enabled
  - Snowflake Notebooks capability
- Python 3.9+

### Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd snowflake-ml-estimator
   ```

2. **Install dependencies** (for local development)
   ```bash
   pip install -r requirements.txt
   ```

3. **Upload notebook to Snowflake**
   - Upload `notebooks/benchmark_runner.ipynb` to your Snowflake account
   - Run cells sequentially to set up tables and compute pools

### Running Benchmarks

1. Open `benchmark_runner.ipynb` in Snowflake Notebooks
2. Configure parameters in the **CENTRALIZED CONFIGURATION** cell:
   ```python
   MAX_COMBINATIONS_TO_RUN = 50    # Limit runs for testing
   JOB_TIMEOUT_SECONDS = 1500      # 25 minute timeout
   RUNS_PER_COMBINATION = 5        # Statistical replicates
   ```
3. Run the test harness to validate setup
4. Run the gap-fill cell to benchmark untested combinations

### Viewing Results

Launch the Streamlit dashboard:
```bash
cd streamlit
streamlit run app.py
```

## 📊 What Gets Benchmarked

### Models
- Logistic Regression
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors
- Naive Bayes
- LightGBM
- XGBoost
- SVC (capped at 100k rows due to O(n²) complexity)

### Compute Pools
- CPU_X64_XS (Extra Small)
- CPU_X64_S (Small)
- CPU_X64_M (Medium)
- CPU_X64_SL (Standard Large)

### Data Dimensions
- Rows: 50k, 250k, 450k, 650k, 850k
- Columns: 25, 50, 75, 100

## 🔧 Configuration

Key parameters in the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_COMBINATIONS_TO_RUN` | 50 | Cap on total combinations (None = unlimited) |
| `JOB_TIMEOUT_SECONDS` | 1500 | Timeout per remote job |
| `RUNS_PER_COMBINATION` | 5 | Repeated runs for statistics |
| `MODEL_ROW_LIMITS` | `{'SVC': 100000}` | Row caps for slow models |

## 📈 Output

### Snowflake Tables
- `ML_BENCHMARK_RESULTS` - Raw timing data
- `BENCHMARK_RAW_DATA` - Synthetic test data (1M rows × 100 features)

### Predictor Model
The notebook trains a two-tier model:
1. **Tier 1**: Success/failure classifier (for configurations that might OOM)
2. **Tier 2**: Duration regressor (RandomForest-based)

Use `predict_ml_execution()` to get predictions:
```python
result = predict_ml_execution('XGBClassifier', 'CPU_X64_M_TEST', 50, 250000)
# {'will_succeed': True, 'predicted_duration': 12.5, 'confidence': 'High'}
```

## ⚠️ Cost Considerations

- Compute pools cost credits while running
- Use `suspend_all_pools()` when done
- Pools auto-resume when jobs are submitted
- Start with small `MAX_COMBINATIONS_TO_RUN` to estimate costs

## 🛠️ Development

### Adding New Models
1. Define the estimator in the **Estimator Setup** cell
2. Add to `base_estimators_lst`
3. (Optional) Add row limits to `MODEL_ROW_LIMITS` if O(n²) or worse

### Adding New Compute Pools
1. Create the pool in the SQL cell
2. Add to `GRID_POOLS` in config
3. Add to pool management functions

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]
