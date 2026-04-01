"""
ML Cost Estimator — Streamlit App

Works both in Snowflake (SiS) and locally.
Local: SNOWFLAKE_CONNECTION_NAME=eudemo streamlit run streamlit_app/app.py
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ML Cost Estimator", page_icon="$", layout="wide")

CREDIT_RATES = {
    "CPU_X64_XS_TEST": 0.06,
    "CPU_X64_S_TEST": 0.12,
    "CPU_X64_M_TEST": 0.24,
    "CPU_X64_SL_TEST": 0.48,
}

POOL_DISPLAY = {
    "CPU_X64_XS_TEST": "XS  |  1 vCPU,  6 GB  |  0.06 cr/hr",
    "CPU_X64_S_TEST":  "S   |  3 vCPU, 13 GB  |  0.12 cr/hr",
    "CPU_X64_M_TEST":  "M   |  6 vCPU, 28 GB  |  0.24 cr/hr",
    "CPU_X64_SL_TEST": "SL  | 14 vCPU, 54 GB  |  0.48 cr/hr",
}

MONTHLY_TRAIN_RUNS = {
    "One-time": 0,
    "Monthly": 1,
    "Weekly": 4.33,
    "Daily": 30,
}


@st.cache_resource
def get_session():
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        from snowflake.snowpark import Session
        conn_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "eudemo")
        session = Session.builder.config("connection_name", conn_name).create()
        session.sql("USE DATABASE ML_ESTIMATOR").collect()
        session.sql("USE SCHEMA PUBLIC").collect()
        return session


@st.cache_data(ttl=300)
def load_training_data():
    return get_session().table("ML_ESTIMATOR.PUBLIC.ML_BENCHMARK_RESULTS").to_pandas()


@st.cache_data(ttl=300)
def load_inference_data():
    try:
        return get_session().table("ML_ESTIMATOR.PUBLIC.ML_INFERENCE_RESULTS").to_pandas()
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def build_training_estimator():
    from snowflake.ml.registry import Registry
    from sklearn.preprocessing import LabelEncoder
    mv = Registry(get_session(), database_name="ML_ESTIMATOR", schema_name="PUBLIC").get_model("ML_COST_ESTIMATOR").default
    model = mv.load(force=True)
    df = load_training_data()
    df = df[df["DURATION_SECONDS"] > 0]
    return model, {
        "model": LabelEncoder().fit(df["MODEL_CLASS"]),
        "pool": LabelEncoder().fit(df["COMPUTE_POOL"]),
        "task": LabelEncoder().fit(df["TASK_TYPE"]),
    }


@st.cache_resource
def build_inference_estimator():
    from snowflake.ml.registry import Registry
    from sklearn.preprocessing import LabelEncoder
    mv = Registry(get_session(), database_name="ML_ESTIMATOR", schema_name="PUBLIC").get_model("ML_INFERENCE_ESTIMATOR").default
    model = mv.load(force=True)
    df = load_inference_data()
    if len(df) == 0:
        return None
    df = df[df["PREDICT_DURATION_SECONDS"] > 0]
    return model, {
        "model": LabelEncoder().fit(df["MODEL_CLASS"]),
        "pool": LabelEncoder().fit(df["COMPUTE_POOL"]),
        "task": LabelEncoder().fit(df["TASK_TYPE"]),
    }


def predict_local(estimator_result, model_class, pool, task_type, feature_name, feature_val, row_name, row_val):
    if estimator_result is None:
        return None
    try:
        model, enc = estimator_result
        X = pd.DataFrame({
            "MODEL_ENCODED": [enc["model"].transform([model_class])[0]],
            "POOL_ENCODED": [enc["pool"].transform([pool])[0]],
            "TASK_ENCODED": [enc["task"].transform([task_type])[0]],
            feature_name: [feature_val],
            row_name: [row_val],
        })
        return float(model.predict(X)[0])
    except Exception:
        return None


def fallback_estimate(df, model_class, pool, row_val, duration_col, row_col):
    subset = df[
        (df["MODEL_CLASS"] == model_class) &
        (df["COMPUTE_POOL"] == pool) &
        (df[duration_col] > 0)
    ]
    if len(subset) == 0:
        return None
    return (subset[duration_col].mean() / subset[row_col].mean()) * row_val


# --- Sidebar ---

with st.sidebar:
    st.header("Settings")
    credit_rate_dollars = st.number_input(
        "Credit Rate ($/credit)", min_value=0.50, max_value=20.0, value=3.00, step=0.25,
        help="Your Snowflake credit price in dollars. Default $3/credit."
    )
    st.divider()
    st.caption("Beta: Classification models only")
    for tt in ["Regression", "Clustering", "Anomaly Detection"]:
        st.caption(f"Coming soon: {tt}")

# --- Main ---

st.title("ML Cost Estimator")
st.markdown("Estimate training and inference costs for Snowflake ML workloads")

try:
    df_train = load_training_data()
    df_infer = load_inference_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

with st.spinner("Loading models..."):
    train_est = build_training_estimator()
    infer_est = build_inference_estimator()

available_models = sorted(df_train[df_train["DURATION_SECONDS"] > 0]["MODEL_CLASS"].unique().tolist()) if len(df_train) > 0 else []
pool_options = list(POOL_DISPLAY.keys())

with st.container():
    st.subheader("Workload Configuration")

    c1, c2 = st.columns(2)
    with c1:
        model_class = st.selectbox("Model", available_models if available_models else ["No models available"])
    with c2:
        pool_idx = st.selectbox(
            "Compute Pool",
            range(len(pool_options)),
            format_func=lambda i: POOL_DISPLAY[pool_options[i]],
        )
        compute_pool = pool_options[pool_idx]

    include_training = st.checkbox("Include Training Cost", value=True)
    train_rows = 200_000
    train_cols = 50
    retrain_freq = "One-time"
    hp_enabled = False
    hp_trials = 20
    if include_training:
        tc1, tc2 = st.columns(2)
        with tc1:
            train_rows = st.number_input("Training Rows", min_value=1_000, max_value=10_000_000, value=200_000, step=10_000, format="%d")
        with tc2:
            train_cols = st.number_input("Training Columns", min_value=5, max_value=500, value=50, step=5)
        retrain_freq = st.selectbox("Retraining Frequency", list(MONTHLY_TRAIN_RUNS.keys()), index=0)
        hp_enabled = st.checkbox("Include HP tuning estimate", value=False)
        if hp_enabled:
            hp_trials = st.number_input("HP Trials", min_value=5, max_value=200, value=20, step=5, format="%d")

    include_inference = st.checkbox("Include Inference Cost", value=True)
    batch_size = 10_000
    infer_cols = train_cols
    batches_per_day = 10
    if include_inference:
        ic1, ic2 = st.columns(2)
        with ic1:
            batch_size = st.number_input("Batch Size (rows)", min_value=100, max_value=1_000_000, value=10_000, step=1_000, format="%d")
        with ic2:
            infer_cols = st.number_input("Inference Columns", min_value=5, max_value=500, value=train_cols, step=5, key="inf_cols")
        batches_per_day = st.number_input("Batches per Day", min_value=0, max_value=1000, value=10, step=1)
        if infer_cols != train_cols:
            st.markdown(
                f":red[**Warning:** Inference columns ({infer_cols}) ≠ Training columns ({train_cols}). "
                f"Models typically expect the same number of features at inference as during training.]"
            )

st.divider()
if st.button("Estimate Cost", type="primary", use_container_width=True):
    st.subheader("Cost Estimate")
    train_duration = None
    infer_duration = None

    if include_training:
        train_duration = predict_local(
            train_est, model_class, compute_pool, "classification",
            "N_COLS_SAMPLED", train_cols, "N_ROWS_SAMPLED", train_rows
        )
        if train_duration is None:
            train_duration = fallback_estimate(df_train, model_class, compute_pool, train_rows, "DURATION_SECONDS", "N_ROWS_SAMPLED")

    if include_inference:
        infer_duration = predict_local(
            infer_est, model_class, compute_pool, "classification",
            "N_COLS", infer_cols, "N_PREDICT_ROWS", batch_size
        )
        if infer_duration is None:
            infer_duration = fallback_estimate(df_infer, model_class, compute_pool, batch_size, "PREDICT_DURATION_SECONDS", "N_PREDICT_ROWS")

    pool_cr_rate = CREDIT_RATES.get(compute_pool, 0.12)

    if include_training:
        st.markdown("---")
        st.markdown("**Training (single run)**")
        if train_duration is not None and train_duration > 0:
            train_credits = (train_duration / 3600) * pool_cr_rate
            train_cost = train_credits * credit_rate_dollars

            if train_duration < 60:
                dur_str = f"{train_duration:.1f}s"
            elif train_duration < 3600:
                dur_str = f"{train_duration/60:.1f}m"
            else:
                dur_str = f"{train_duration/3600:.1f}h"

            m1, m2, m3 = st.columns(3)
            m1.metric("Duration", dur_str)
            m2.metric("Credits", f"{train_credits:.4f}")
            m3.metric("Cost", f"${train_cost:.2f}")

            if hp_enabled:
                hp_credits = train_credits * hp_trials
                hp_cost = hp_credits * credit_rate_dollars
                st.caption(f"With HP tuning ({hp_trials} trials): {train_duration * hp_trials / 60:.1f}m | {hp_credits:.4f} cr | ${hp_cost:.2f}")
        else:
            st.warning("No training estimate available for this configuration")

    if include_inference:
        st.markdown("---")
        st.markdown("**Inference (single batch)**")
        if infer_duration is not None and infer_duration > 0:
            infer_credits = (infer_duration / 3600) * pool_cr_rate
            infer_cost = infer_credits * credit_rate_dollars

            if infer_duration < 1:
                infer_dur_str = f"{infer_duration*1000:.0f}ms"
            elif infer_duration < 60:
                infer_dur_str = f"{infer_duration:.2f}s"
            else:
                infer_dur_str = f"{infer_duration/60:.1f}m"

            m1, m2, m3 = st.columns(3)
            m1.metric("Duration", infer_dur_str)
            m2.metric("Credits", f"{infer_credits:.6f}")
            m3.metric("Cost", f"${infer_cost:.4f}")
        else:
            st.info("Inference benchmarks loading... check back soon.")

    st.markdown("---")
    st.markdown("**Monthly Projection**")

    monthly_train_runs = MONTHLY_TRAIN_RUNS.get(retrain_freq, 0)
    monthly_batches = batches_per_day * 30

    monthly_train_credits = 0
    monthly_infer_credits = 0

    if train_duration and train_duration > 0:
        single_train_cr = (train_duration / 3600) * pool_cr_rate
        if hp_enabled:
            single_train_cr *= hp_trials
        monthly_train_credits = single_train_cr * monthly_train_runs

    if infer_duration and infer_duration > 0:
        single_infer_cr = (infer_duration / 3600) * pool_cr_rate
        monthly_infer_credits = single_infer_cr * monthly_batches

    total_monthly_credits = monthly_train_credits + monthly_infer_credits
    total_monthly_cost = total_monthly_credits * credit_rate_dollars

    m1, m2 = st.columns(2)
    m1.metric("Monthly Credits", f"{total_monthly_credits:.2f}")
    m2.metric("Monthly Cost", f"${total_monthly_cost:.2f}")

    breakdown = pd.DataFrame({
        "Component": ["Training", "Inference", "Total"],
        "Monthly Credits": [
            round(monthly_train_credits, 4),
            round(monthly_infer_credits, 4),
            round(total_monthly_credits, 4),
        ],
        "Monthly Cost ($)": [
            round(monthly_train_credits * credit_rate_dollars, 2),
            round(monthly_infer_credits * credit_rate_dollars, 2),
            round(total_monthly_cost, 2),
        ],
    })
    st.dataframe(breakdown, use_container_width=True)

    st.caption(
        f"Training: {monthly_train_runs:.1f} runs/month | "
        f"Inference: {monthly_batches:,} batches/month ({batches_per_day}/day x 30d) | "
        f"Rate: ${credit_rate_dollars}/credit"
    )

st.divider()

tab1, tab2, tab3 = st.tabs(["Data Coverage", "Analysis", "Compute Pool Info"])

with tab1:
    st.subheader("Training Benchmark Coverage")
    if len(df_train) > 0:
        success_train = df_train[df_train["DURATION_SECONDS"] > 0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs", f"{len(success_train):,}")
        c2.metric("Models", f"{success_train['MODEL_CLASS'].nunique()}")
        c3.metric("Pools", f"{success_train['COMPUTE_POOL'].nunique()}")
        c4.metric("Task Types", f"{success_train['TASK_TYPE'].nunique()}")

        st.dataframe(
            success_train.groupby(["MODEL_CLASS", "COMPUTE_POOL", "N_ROWS_SAMPLED", "N_COLS_SAMPLED"])
            .agg(runs=("DURATION_SECONDS", "count"), avg_sec=("DURATION_SECONDS", "mean"), std_sec=("DURATION_SECONDS", "std"))
            .round(2).reset_index(),
            use_container_width=True
        )
    else:
        st.info("Training benchmarks are running... data will appear here soon.")

    st.subheader("Inference Benchmark Coverage")
    if len(df_infer) > 0:
        success_infer = df_infer[df_infer["PREDICT_DURATION_SECONDS"] > 0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Runs", f"{len(success_infer):,}")
        c2.metric("Models", f"{success_infer['MODEL_CLASS'].nunique()}")
        c3.metric("Pools", f"{success_infer['COMPUTE_POOL'].nunique()}")
    else:
        st.info("Inference benchmarks are running... data will appear here soon.")

with tab2:
    if len(df_train) > 0:
        import plotly.express as px
        success_df = df_train[df_train["DURATION_SECONDS"] > 0]

        row_options = sorted(success_df["N_ROWS_SAMPLED"].unique().tolist())
        col_options = sorted(success_df["N_COLS_SAMPLED"].unique().tolist())
        size_options = sorted(success_df[["N_ROWS_SAMPLED", "N_COLS_SAMPLED"]].drop_duplicates().apply(
            lambda r: f"{int(r['N_ROWS_SAMPLED']):,} rows × {int(r['N_COLS_SAMPLED'])} cols", axis=1
        ).tolist())
        selected_size = st.selectbox("Dataset size", size_options, index=len(size_options) // 2)
        sel_rows = int(selected_size.split(" rows")[0].replace(",", ""))
        sel_cols = int(selected_size.split("× ")[1].split(" cols")[0])
        filtered_df = success_df[
            (success_df["N_ROWS_SAMPLED"] == sel_rows) &
            (success_df["N_COLS_SAMPLED"] == sel_cols)
        ]

        st.subheader("Training Duration by Model")
        fig = px.box(filtered_df, x="MODEL_CLASS", y="DURATION_SECONDS", color="COMPUTE_POOL", log_y=True)
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Compute Pool Rates")
    rates_df = pd.DataFrame([
        {"Pool": k, "Credits/Hour": v, "$/Hour": round(v * credit_rate_dollars, 2)}
        for k, v in CREDIT_RATES.items()
    ])
    st.dataframe(rates_df, use_container_width=True)

    st.subheader("Methodology")
    st.markdown("""
    - Models are trained on synthetic data and timed on actual Snowflake SPCS compute pools
    - Each model x pool x data-size combination is benchmarked multiple times
    - A RandomForest meta-model predicts duration for unseen configurations
    - Credits = (duration_seconds / 3600) x pool_credit_rate
    - Cost = credits x your credit rate (configurable in sidebar)
    """)
