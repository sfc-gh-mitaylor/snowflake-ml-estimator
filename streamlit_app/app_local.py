"""
ML Cost Estimator - Local Streamlit App

Predicts training time and credit cost for ML workloads.
Run with: SNOWFLAKE_CONNECTION_NAME=eudemo streamlit run streamlit_app/app_local.py
"""
import os
import streamlit as st
import pandas as pd
from snowflake.snowpark import Session

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
    conn_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "eudemo")
    session = Session.builder.config("connection_name", conn_name).create()
    session.sql("USE DATABASE ML_ESTIMATOR").collect()
    session.sql("USE SCHEMA PUBLIC").collect()
    return session


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
    
    st.subheader("Hyperparameter Tuning")
    hp_enabled = st.checkbox("Include HP tuning estimate", value=False)
    if hp_enabled:
        hp_trials = st.slider("Number of HP trials", min_value=5, max_value=200, value=20, step=5,
                              help="Number of hyperparameter combinations to search")
        st.caption("Linear scaling assumed (worst case). Early stopping may reduce actual time.")

with col2:
    st.subheader("Prediction")
    
    if st.button("Estimate Cost", type="primary", use_container_width=True):
        with st.spinner("Loading model and predicting..."):
            try:
                from sklearn.preprocessing import LabelEncoder
                
                le_model = LabelEncoder().fit(df["MODEL_CLASS"].unique())
                le_pool = LabelEncoder().fit(df["COMPUTE_POOL"].unique())
                le_task = LabelEncoder().fit(df["TASK_TYPE"].unique())
                
                model_enc = le_model.transform([model_class])[0]
                pool_enc = le_pool.transform([compute_pool])[0]
                task_enc = le_task.transform([task_type])[0]
                
                model = load_model()
                input_df = pd.DataFrame({
                    "MODEL_ENCODED": [model_enc],
                    "POOL_ENCODED": [pool_enc],
                    "TASK_ENCODED": [task_enc],
                    "N_COLS_SAMPLED": [n_cols],
                    "N_ROWS_SAMPLED": [n_rows],
                })
                
                result = model.run(input_df, function_name="predict")
                base_duration = float(result["output_feature_0"].iloc[0])
                
                rate = CREDIT_RATES.get(compute_pool, 0.12)
                base_credits = (base_duration / 3600) * rate
                
                st.markdown("**Single Training Run**")
                st.metric("Duration", f"{base_duration:.1f}s" if base_duration < 60 else f"{base_duration/60:.1f}m")
                st.metric("Credits", f"{base_credits:.4f}")
                
                if hp_enabled:
                    st.divider()
                    st.markdown("**HP Tuning Estimate**")
                    hp_duration = base_duration * hp_trials
                    hp_credits = base_credits * hp_trials
                    
                    if hp_duration < 60:
                        dur_str = f"{hp_duration:.1f}s"
                    elif hp_duration < 3600:
                        dur_str = f"{hp_duration/60:.1f}m"
                    else:
                        dur_str = f"{hp_duration/3600:.1f}h"
                    
                    st.metric("Total Duration", dur_str, help=f"{hp_trials} trials")
                    st.metric("Total Credits", f"{hp_credits:.4f}")
                    st.caption(f"⚠️ Worst-case linear scaling ({hp_trials}x). "
                              "Early stopping, parallel trials, or Bayesian optimization may reduce actual cost.")
                
                st.caption(f"Rate: {rate} credits/hr")
                
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                st.info("Using fallback estimation from historical data...")
                
                subset = df[
                    (df["MODEL_CLASS"] == model_class) & 
                    (df["COMPUTE_POOL"] == compute_pool) &
                    (df["DURATION_SECONDS"] > 0)
                ]
                
                if len(subset) > 0:
                    avg_dur_per_row = subset["DURATION_SECONDS"].mean() / subset["N_ROWS_SAMPLED"].mean()
                    base_duration = avg_dur_per_row * n_rows
                    rate = CREDIT_RATES.get(compute_pool, 0.12)
                    base_credits = (base_duration / 3600) * rate
                    
                    st.metric("Est. Duration", f"{base_duration:.1f}s")
                    st.metric("Est. Credits", f"{base_credits:.4f}")
                    
                    if hp_enabled:
                        st.divider()
                        hp_duration = base_duration * hp_trials
                        hp_credits = base_credits * hp_trials
                        st.metric("HP Duration", f"{hp_duration/60:.1f}m" if hp_duration < 3600 else f"{hp_duration/3600:.1f}h")
                        st.metric("HP Credits", f"{hp_credits:.4f}")
                    
                    st.caption("⚠️ Rough estimate from historical average")
                else:
                    st.warning("No historical data for this configuration")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Analysis", "⚙️ Model Info"])

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
    st.subheader("Registered Model")
    st.code("ML_ESTIMATOR.PUBLIC.ML_COST_ESTIMATOR/V2", language="sql")
    
    st.markdown("""
    **Model Metrics (V2):**
    - R²: 0.9077
    - MAE: 9.31s
    - RMSE: 97.62s
    - Training samples: 4,931 (16 models)
    
    **Features (encoded):**
    - MODEL_ENCODED (categorical → int)
    - POOL_ENCODED (categorical → int)  
    - TASK_ENCODED (categorical → int)
    - N_COLS_SAMPLED (int)
    - N_ROWS_SAMPLED (int)
    
    **HP Tuning Scaling:**
    - Linear scaling is worst-case assumption
    - Actual time depends on search strategy (grid/random/Bayesian)
    - Early stopping can reduce iterations by 30-50%
    - Cross-validation multiplies by k folds
    """)
    
    st.subheader("Credit Rates")
    rates_df = pd.DataFrame([
        {"Pool": k, "Credits/Hour": v} for k, v in CREDIT_RATES.items()
    ])
    st.dataframe(rates_df, use_container_width=True, hide_index=True)
