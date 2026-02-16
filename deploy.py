"""
Deployment script for ML Cost Estimator to Snowflake.

Usage:
    python deploy.py setup      # Create database objects
    python deploy.py migrate    # Migrate existing data to new schema  
    python deploy.py app        # Deploy Streamlit app
    python deploy.py all        # Run all steps
"""
import argparse
import os

DATABASE = "ML_ESTIMATOR"
SCHEMA = "PUBLIC"
RESULTS_TABLE = "ML_BENCHMARK_RESULTS"
STAGE = "APP_STAGE"
APP_NAME = "COST_ESTIMATOR_APP"


def get_session():
    from snowflake.snowpark import Session
    conn_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
    return Session.builder.config("connection_name", conn_name).create()


def setup_database(session):
    """Create database, schema, and tables."""
    print(f"Creating database {DATABASE}...")
    session.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}").collect()
    session.sql(f"USE DATABASE {DATABASE}").collect()
    session.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}").collect()
    session.sql(f"USE SCHEMA {SCHEMA}").collect()
    
    print(f"Creating results table {RESULTS_TABLE}...")
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
            MODEL_CLASS VARCHAR(100),
            TASK_TYPE VARCHAR(50),
            COMPUTE_POOL VARCHAR(50),
            RUN_ID INTEGER,
            N_COLS_SAMPLED INTEGER,
            N_ROWS_SAMPLED INTEGER,
            DURATION_SECONDS FLOAT,
            ESTIMATED_CREDITS FLOAT,
            START_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """).collect()
    
    print(f"Creating stage {STAGE}...")
    session.sql(f"CREATE STAGE IF NOT EXISTS {STAGE}").collect()
    
    print("Database setup complete!")


def migrate_existing_data(session):
    """Migrate data from old schema to new schema if needed."""
    print("Checking for existing data to migrate...")
    
    try:
        old_df = session.sql("""
            SELECT * FROM ML_BENCHMARK_RESULTS 
            WHERE TASK_TYPE IS NULL
            LIMIT 1
        """).collect()
        
        if old_df:
            print("Migrating old records (adding TASK_TYPE='classification')...")
            session.sql("""
                UPDATE ML_BENCHMARK_RESULTS 
                SET TASK_TYPE = 'classification',
                    ESTIMATED_CREDITS = COALESCE(ESTIMATED_CREDITS, 
                        (DURATION_SECONDS / 3600) * 
                        CASE COMPUTE_POOL
                            WHEN 'CPU_X64_XS_TEST' THEN 0.06
                            WHEN 'CPU_X64_S_TEST' THEN 0.12
                            WHEN 'CPU_X64_M_TEST' THEN 0.24
                            WHEN 'CPU_X64_SL_TEST' THEN 0.48
                            ELSE 0.12
                        END
                    )
                WHERE TASK_TYPE IS NULL
            """).collect()
            print("Migration complete!")
        else:
            print("No migration needed.")
    except Exception as e:
        print(f"Migration check failed (table may not exist yet): {e}")


def deploy_streamlit_app(session):
    """Deploy Streamlit app to Snowflake."""
    import subprocess
    
    print(f"Deploying Streamlit app {APP_NAME}...")
    
    app_dir = os.path.join(os.path.dirname(__file__), "streamlit")
    
    result = subprocess.run(
        [
            "snow", "streamlit", "deploy",
            "--database", DATABASE,
            "--schema", SCHEMA,
            "--replace",
        ],
        cwd=app_dir,
        capture_output=True,
        text=True,
    )
    
    if result.returncode == 0:
        print(f"App deployed successfully!")
        print(f"Access at: https://app.snowflake.com/{DATABASE}.{SCHEMA}.{APP_NAME}")
    else:
        print(f"Deployment failed: {result.stderr}")
        print("Trying manual SQL deployment...")
        
        session.sql(f"USE DATABASE {DATABASE}").collect()
        session.sql(f"USE SCHEMA {SCHEMA}").collect()
        
        with open(os.path.join(app_dir, "app.py"), "r") as f:
            app_code = f.read()
        
        session.sql(f"PUT file://{app_dir}/app.py @{STAGE} AUTO_COMPRESS=FALSE OVERWRITE=TRUE").collect()
        
        session.sql(f"""
            CREATE OR REPLACE STREAMLIT {APP_NAME}
            ROOT_LOCATION = '@{DATABASE}.{SCHEMA}.{STAGE}'
            MAIN_FILE = 'app.py'
            QUERY_WAREHOUSE = 'COMPUTE_WH'
        """).collect()
        
        print(f"App deployed via SQL!")


def show_status(session):
    """Show deployment status."""
    session.sql(f"USE DATABASE {DATABASE}").collect()
    session.sql(f"USE SCHEMA {SCHEMA}").collect()
    
    print("\n" + "=" * 50)
    print("DEPLOYMENT STATUS")
    print("=" * 50)
    
    try:
        count = session.sql(f"SELECT COUNT(*) as cnt FROM {RESULTS_TABLE}").collect()[0]["CNT"]
        print(f"Benchmark results: {count:,} rows")
    except:
        print("Benchmark results: TABLE NOT FOUND")
    
    try:
        models = session.sql("SHOW MODELS").collect()
        print(f"Registered models: {len(models)}")
    except:
        print("Registered models: NONE")
    
    try:
        apps = session.sql(f"SHOW STREAMLITS LIKE '{APP_NAME}'").collect()
        print(f"Streamlit app: {'DEPLOYED' if apps else 'NOT DEPLOYED'}")
    except:
        print("Streamlit app: UNKNOWN")


def main():
    parser = argparse.ArgumentParser(description="Deploy ML Cost Estimator")
    parser.add_argument("command", choices=["setup", "migrate", "app", "status", "all"])
    args = parser.parse_args()
    
    session = get_session()
    
    if args.command == "setup" or args.command == "all":
        setup_database(session)
    
    if args.command == "migrate" or args.command == "all":
        migrate_existing_data(session)
    
    if args.command == "app" or args.command == "all":
        deploy_streamlit_app(session)
    
    if args.command == "status" or args.command == "all":
        show_status(session)


if __name__ == "__main__":
    main()
