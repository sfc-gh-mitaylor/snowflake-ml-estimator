"""
Snowflake ML Runtime Estimator - Streamlit Dashboard

This app provides an interactive dashboard for:
- Viewing benchmark results
- Exploring duration patterns across models, pools, and data sizes
- Making runtime predictions for new configurations
- Monitoring grid coverage
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local imports
from config import (
    RESULTS_TABLE, 
    MODELS_DISPLAY_ORDER, 
    POOL_DISPLAY_NAMES,
    CHART_COLORS
)

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="ML Runtime Estimator",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Snowflake Connection
# =============================================================================
@st.cache_resource
def get_snowflake_connection():
    """Establish Snowflake connection using Streamlit secrets."""
    # TODO: Implement Snowflake connection
    # Option 1: Use st.connection (Streamlit native)
    # Option 2: Use snowflake-connector-python
    # Option 3: Use snowflake-snowpark-python
    
    # For now, return None - implement based on your setup
    st.warning("⚠️ Snowflake connection not configured. Using sample data.")
    return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_benchmark_data(_conn):
    """Load benchmark results from Snowflake."""
    if _conn is None:
        # Return sample data for development
        return _generate_sample_data()
    
    # TODO: Query actual data
    # query = f"SELECT * FROM {RESULTS_TABLE} WHERE DURATION_SECONDS > 0"
    # return pd.read_sql(query, _conn)
    return _generate_sample_data()


def _generate_sample_data():
    """Generate sample data for development/demo."""
    np.random.seed(42)
    
    models = ['LogisticRegression', 'GradientBoostingClassifier', 'XGBClassifier', 
              'LGBMClassifier', 'AdaBoostClassifier', 'GaussianNB', 'KNeighborsClassifier']
    pools = ['CPU_X64_XS_TEST', 'CPU_X64_S_TEST', 'CPU_X64_M_TEST', 'CPU_X64_SL_TEST']
    cols = [25, 50, 75, 100]
    rows = [50000, 250000, 450000, 650000, 850000]
    
    data = []
    for model in models:
        base_time = np.random.uniform(0.5, 10)  # Base time varies by model
        for pool in pools:
            pool_factor = {'CPU_X64_XS_TEST': 1.5, 'CPU_X64_S_TEST': 1.2, 
                          'CPU_X64_M_TEST': 1.0, 'CPU_X64_SL_TEST': 0.8}[pool]
            for n_cols in cols:
                for n_rows in rows:
                    for run_id in range(1, 6):
                        duration = base_time * pool_factor * (n_rows / 50000) * (n_cols / 25)
                        duration *= np.random.uniform(0.9, 1.1)  # Add noise
                        data.append({
                            'MODEL_CLASS': model,
                            'COMPUTE_POOL': pool,
                            'N_COLS_SAMPLED': n_cols,
                            'N_ROWS_SAMPLED': n_rows,
                            'RUN_ID': run_id,
                            'DURATION_SECONDS': duration,
                            'START_TIMESTAMP': pd.Timestamp.now().timestamp()
                        })
    
    return pd.DataFrame(data)


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar(df):
    """Render sidebar with filters."""
    st.sidebar.header("🎛️ Filters")
    
    # Model filter
    models = st.sidebar.multiselect(
        "Models",
        options=sorted(df['MODEL_CLASS'].unique()),
        default=sorted(df['MODEL_CLASS'].unique())
    )
    
    # Pool filter
    pools = st.sidebar.multiselect(
        "Compute Pools",
        options=sorted(df['COMPUTE_POOL'].unique()),
        default=sorted(df['COMPUTE_POOL'].unique())
    )
    
    # Row count filter
    min_rows, max_rows = st.sidebar.slider(
        "Row Count Range",
        min_value=int(df['N_ROWS_SAMPLED'].min()),
        max_value=int(df['N_ROWS_SAMPLED'].max()),
        value=(int(df['N_ROWS_SAMPLED'].min()), int(df['N_ROWS_SAMPLED'].max()))
    )
    
    return models, pools, min_rows, max_rows


# =============================================================================
# Main Dashboard
# =============================================================================
def main():
    st.title("⏱️ ML Runtime Estimator Dashboard")
    st.markdown("Benchmark analysis and runtime predictions for Snowflake ML workloads")
    
    # Load data
    conn = get_snowflake_connection()
    df = load_benchmark_data(conn)
    
    if df is None or len(df) == 0:
        st.error("No benchmark data available. Run the benchmark notebook first.")
        return
    
    # Sidebar filters
    models, pools, min_rows, max_rows = render_sidebar(df)
    
    # Apply filters
    mask = (
        (df['MODEL_CLASS'].isin(models)) &
        (df['COMPUTE_POOL'].isin(pools)) &
        (df['N_ROWS_SAMPLED'] >= min_rows) &
        (df['N_ROWS_SAMPLED'] <= max_rows)
    )
    filtered_df = df[mask]
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", f"{len(filtered_df):,}")
    with col2:
        st.metric("Unique Combinations", f"{filtered_df.groupby(['MODEL_CLASS', 'COMPUTE_POOL', 'N_COLS_SAMPLED', 'N_ROWS_SAMPLED']).ngroups:,}")
    with col3:
        st.metric("Median Duration", f"{filtered_df['DURATION_SECONDS'].median():.2f}s")
    with col4:
        st.metric("Max Duration", f"{filtered_df['DURATION_SECONDS'].max():.2f}s")
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Model Analysis", "📈 Scaling", "🎯 Predictor"])
    
    with tab1:
        render_overview_tab(filtered_df)
    
    with tab2:
        render_model_analysis_tab(filtered_df)
    
    with tab3:
        render_scaling_tab(filtered_df)
    
    with tab4:
        render_predictor_tab(df)  # Use full df for predictor


def render_overview_tab(df):
    """Render overview tab with summary charts."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Duration by Model")
        fig = px.box(df, x='MODEL_CLASS', y='DURATION_SECONDS', 
                     color='MODEL_CLASS', log_y=True)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Duration by Compute Pool")
        fig = px.box(df, x='COMPUTE_POOL', y='DURATION_SECONDS',
                     color='COMPUTE_POOL')
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("Mean Duration Heatmap (Model × Pool)")
    pivot = df.pivot_table(values='DURATION_SECONDS', index='MODEL_CLASS', 
                           columns='COMPUTE_POOL', aggfunc='mean')
    fig = px.imshow(pivot, text_auto='.1f', aspect='auto', color_continuous_scale='YlOrRd')
    st.plotly_chart(fig, use_container_width=True)


def render_model_analysis_tab(df):
    """Render model-specific analysis."""
    selected_model = st.selectbox("Select Model", sorted(df['MODEL_CLASS'].unique()))
    model_df = df[df['MODEL_CLASS'] == selected_model]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{selected_model} - Duration vs Rows")
        fig = px.scatter(model_df, x='N_ROWS_SAMPLED', y='DURATION_SECONDS',
                        color='COMPUTE_POOL', trendline='lowess')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"{selected_model} - Duration vs Columns")
        fig = px.scatter(model_df, x='N_COLS_SAMPLED', y='DURATION_SECONDS',
                        color='COMPUTE_POOL', trendline='lowess')
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    st.subheader(f"{selected_model} - Statistics by Pool")
    stats = model_df.groupby('COMPUTE_POOL')['DURATION_SECONDS'].agg(['count', 'mean', 'std', 'min', 'max'])
    stats.columns = ['Runs', 'Mean (s)', 'Std (s)', 'Min (s)', 'Max (s)']
    st.dataframe(stats.round(2), use_container_width=True)


def render_scaling_tab(df):
    """Render scaling analysis."""
    st.subheader("How Duration Scales with Data Size")
    
    # Group by rows and show scaling
    scaling_df = df.groupby(['MODEL_CLASS', 'N_ROWS_SAMPLED'])['DURATION_SECONDS'].mean().reset_index()
    
    fig = px.line(scaling_df, x='N_ROWS_SAMPLED', y='DURATION_SECONDS',
                  color='MODEL_CLASS', markers=True, log_y=True)
    fig.update_layout(xaxis_title='Number of Rows', yaxis_title='Mean Duration (s, log scale)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Scaling by columns
    col_scaling_df = df.groupby(['MODEL_CLASS', 'N_COLS_SAMPLED'])['DURATION_SECONDS'].mean().reset_index()
    
    fig2 = px.line(col_scaling_df, x='N_COLS_SAMPLED', y='DURATION_SECONDS',
                   color='MODEL_CLASS', markers=True)
    fig2.update_layout(xaxis_title='Number of Columns', yaxis_title='Mean Duration (s)')
    st.plotly_chart(fig2, use_container_width=True)


def render_predictor_tab(df):
    """Render prediction interface."""
    st.subheader("🎯 Runtime Predictor")
    st.markdown("Estimate training time for a new configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model = st.selectbox("Model", sorted(df['MODEL_CLASS'].unique()), key='pred_model')
        pool = st.selectbox("Compute Pool", sorted(df['COMPUTE_POOL'].unique()), key='pred_pool')
    
    with col2:
        n_cols = st.slider("Number of Columns", 10, 200, 50)
        n_rows = st.slider("Number of Rows", 10000, 1000000, 100000, step=10000)
    
    if st.button("Predict Runtime", type="primary"):
        # Simple prediction using historical data
        similar = df[
            (df['MODEL_CLASS'] == model) & 
            (df['COMPUTE_POOL'] == pool)
        ]
        
        if len(similar) > 0:
            # Basic interpolation based on data size
            avg_per_row = similar['DURATION_SECONDS'].mean() / similar['N_ROWS_SAMPLED'].mean()
            avg_per_col = similar['DURATION_SECONDS'].mean() / similar['N_COLS_SAMPLED'].mean()
            
            predicted = (avg_per_row * n_rows + avg_per_col * n_cols) / 2
            
            st.success(f"**Predicted Runtime: {predicted:.2f} seconds**")
            st.caption("Note: This is a simple estimate. Train the full model in the notebook for better predictions.")
        else:
            st.warning("No historical data for this combination. Run benchmarks first.")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    main()
