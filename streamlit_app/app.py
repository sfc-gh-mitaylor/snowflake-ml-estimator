"""
ML Cost Estimator - Streamlit in Snowflake App

Predicts training time and credit cost for ML workloads.
"""
import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="ML Cost Estimator", page_icon="💰", layout="wide")

CREDIT_RATES = {
    "CPU_X64_XS_TEST": 0.06,
    "CPU_X64_S_TEST": 0.12,
    "CPU_X64_M_TEST": 0.24,
    "CPU_X64_SL_TEST": 0.48,
    "GPU_NV_S": 0.60,
    "GPU_NV_M": 1.20,
}

POOL_DISPLAY = {
    "CPU_X64_XS_TEST": "CPU XS (0.06 cr/hr)",
    "CPU_X64_S_TEST": "CPU S (0.12 cr/hr)",
    "CPU_X64_M_TEST": "CPU M (0.24 cr/hr)",
    "CPU_X64_SL_TEST": "CPU SL (0.48 cr/hr)",
    "GPU_NV_S": "GPU S (0.60 cr/hr)",
    "GPU_NV_M": "GPU M (1.20 cr/hr)",
}


@st.cache_resource
def get_session():
    return get_active_session()


@st.cache_data(ttl=300)
def load_benchmark_data():
    session = get_session()
    return session.table("ML_ESTIMATOR.PUBLIC.ML_BENCHMARK_RESULTS").to_pandas()


@st.cache_resource
def load_model():
    from snowflake.ml.registry import Registry
    session = get_session()
    registry = Registry(session, database_name="ML_ESTIMATOR", schema_name="PUBLIC")
    return registry.get_model("ML_COST_ESTIMATOR").default


def get_available_options(df):
    return {
        "models": sorted(df["MODEL_CLASS"].unique().tolist()),
        "task_types": sorted(df["TASK_TYPE"].unique().tolist()),
        "pools": sorted(df["COMPUTE_POOL"].unique().tolist()),
    }


def predict_cost(model, model_class, task_type, pool, n_cols, n_rows):
    try:
        result = model.predict(model_class, task_type, pool, n_cols, n_rows)
        return result
    except Exception as e:
        return {"error": str(e), "duration_seconds": None, "estimated_credits": None}


def retrain_model():
    from snowflake.ml.registry import Registry
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    import numpy as np
    
    session = get_session()
    df = session.table("ML_ESTIMATOR.PUBLIC.ML_BENCHMARK_RESULTS").to_pandas()
    df = df[df["DURATION_SECONDS"] > 0].copy()
    
    le_model = LabelEncoder()
    le_pool = LabelEncoder()
    le_task = LabelEncoder()
    
    df["MODEL_ENCODED"] = le_model.fit_transform(df["MODEL_CLASS"])
    df["POOL_ENCODED"] = le_pool.fit_transform(df["COMPUTE_POOL"])
    df["TASK_ENCODED"] = le_task.fit_transform(df["TASK_TYPE"])
    
    feature_cols = ["MODEL_ENCODED", "POOL_ENCODED", "TASK_ENCODED", 
                    "N_COLS_SAMPLED", "N_ROWS_SAMPLED"]
    X = df[feature_cols]
    y = df["DURATION_SECONDS"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    
    return metrics


st.title("💰 ML Cost Estimator")
st.markdown("Predict training time and credit cost for Snowflake ML workloads")

try:
    df = load_benchmark_data()
    options = get_available_options(df)
except Exception as e:
    st.error(f"Failed to load benchmark data: {e}")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Configuration")
    
    task_type = st.selectbox("Task Type", options["task_types"], index=0)
    
    task_models = df[df["TASK_TYPE"] == task_type]["MODEL_CLASS"].unique().tolist()
    model_class = st.selectbox("Model", sorted(task_models))
    
    pool_options = list(POOL_DISPLAY.keys())
    pool_labels = [POOL_DISPLAY[p] for p in pool_options]
    pool_idx = st.selectbox("Compute Pool", range(len(pool_options)), 
                            format_func=lambda i: pool_labels[i])
    compute_pool = pool_options[pool_idx]
    
    c1, c2 = st.columns(2)
    with c1:
        n_rows = st.number_input("Rows", min_value=1000, max_value=10_000_000, 
                                  value=100_000, step=10_000, format="%d")
    with c2:
        n_cols = st.number_input("Columns", min_value=5, max_value=500, 
                                  value=50, step=5)

with col2:
    st.subheader("Prediction")
    
    if st.button("Estimate Cost", type="primary", use_container_width=True):
        try:
            model = load_model()
            result = predict_cost(model, model_class, task_type, compute_pool, n_cols, n_rows)
            
            if result.get("error"):
                st.error(f"Prediction failed: {result['error']}")
            else:
                duration = result["duration_seconds"]
                credits = result["estimated_credits"]
                
                st.metric("Duration", f"{duration:.1f}s" if duration < 60 else f"{duration/60:.1f}m")
                st.metric("Credits", f"{credits:.4f}")
                
                rate = CREDIT_RATES.get(compute_pool, 0.12)
                st.caption(f"Rate: {rate} credits/hr")
        except Exception as e:
            st.error(f"Model not available: {e}")
            st.info("Using fallback estimation from historical data...")
            
            subset = df[
                (df["MODEL_CLASS"] == model_class) & 
                (df["COMPUTE_POOL"] == compute_pool) &
                (df["DURATION_SECONDS"] > 0)
            ]
            
            if len(subset) > 0:
                avg_dur_per_row = subset["DURATION_SECONDS"].mean() / subset["N_ROWS_SAMPLED"].mean()
                est_duration = avg_dur_per_row * n_rows
                rate = CREDIT_RATES.get(compute_pool, 0.12)
                est_credits = (est_duration / 3600) * rate
                
                st.metric("Est. Duration", f"{est_duration:.1f}s")
                st.metric("Est. Credits", f"{est_credits:.4f}")
                st.caption("⚠️ Rough estimate from historical average")
            else:
                st.warning("No historical data for this configuration")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Analysis", "⚙️ Admin"])

with tab1:
    st.subheader("Benchmark Coverage")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Runs", f"{len(df):,}")
    c2.metric("Unique Configs", f"{df.groupby(['MODEL_CLASS','COMPUTE_POOL','N_ROWS_SAMPLED']).ngroups:,}")
    c3.metric("Success Rate", f"{(df['DURATION_SECONDS'] > 0).mean():.1%}")
    c4.metric("Models Tested", f"{df['MODEL_CLASS'].nunique()}")
    
    st.dataframe(
        df.groupby(["TASK_TYPE", "MODEL_CLASS"])
        .agg({"DURATION_SECONDS": ["count", "mean", "std"]})
        .round(2),
        use_container_width=True
    )

with tab2:
    st.subheader("Duration by Configuration")
    
    import plotly.express as px
    
    success_df = df[df["DURATION_SECONDS"] > 0]
    
    fig = px.box(success_df, x="MODEL_CLASS", y="DURATION_SECONDS", 
                 color="TASK_TYPE", log_y=True)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = px.scatter(success_df, x="N_ROWS_SAMPLED", y="DURATION_SECONDS",
                      color="MODEL_CLASS", facet_col="COMPUTE_POOL", facet_col_wrap=3)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Model Management")
    
    if st.button("🔄 Retrain Model", help="Retrain the cost estimator on latest data"):
        with st.spinner("Retraining model..."):
            try:
                metrics = retrain_model()
                st.success(f"Model retrained! R²={metrics['r2']:.3f}, MAE={metrics['mae']:.1f}s")
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Retrain failed: {e}")
    
    st.subheader("Credit Rates")
    rates_df = pd.DataFrame([
        {"Pool": k, "Credits/Hour": v} for k, v in CREDIT_RATES.items()
    ])
    st.dataframe(rates_df, use_container_width=True, hide_index=True)
