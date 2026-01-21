# Streamlit App Configuration
"""
Configuration for the ML Estimator Streamlit dashboard.
"""

# Snowflake connection settings (override via secrets.toml or environment)
SNOWFLAKE_CONFIG = {
    "account": "",      # Set in .streamlit/secrets.toml
    "user": "",
    "password": "",
    "warehouse": "",
    "database": "",
    "schema": "",
}

# Table names (must match notebook configuration)
RESULTS_TABLE = "ML_BENCHMARK_RESULTS"
DATA_TABLE = "BENCHMARK_RAW_DATA"

# Display settings
MODELS_DISPLAY_ORDER = [
    "LogisticRegression",
    "GaussianNB", 
    "AdaBoostClassifier",
    "GradientBoostingClassifier",
    "LGBMClassifier",
    "XGBClassifier",
    "KNeighborsClassifier",
    "SVC",
    "RandomForestClassifier",
]

POOL_DISPLAY_NAMES = {
    "CPU_X64_XS_TEST": "XS (Extra Small)",
    "CPU_X64_S_TEST": "S (Small)",
    "CPU_X64_M_TEST": "M (Medium)",
    "CPU_X64_SL_TEST": "SL (Standard Large)",
}

# Chart colors
CHART_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e", 
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ffbb00",
}
