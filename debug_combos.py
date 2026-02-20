from src.runner import get_snowflake_session, get_tested_combinations, generate_all_combinations
from src.config import DEFAULT_CONFIG
from src.estimators import DEFAULT_FACTORY
import os
os.environ["SNOWFLAKE_CONNECTION_NAME"] = "eudemo"

session = get_snowflake_session()
config = DEFAULT_CONFIG

all_combos = generate_all_combinations(DEFAULT_FACTORY, config)
tested = get_tested_combinations(session, config.results_table_name)

for task in ['classification', 'regression', 'clustering', 'anomaly_detection']:
    task_all = [c for c in all_combos if c[1] == task]
    task_remaining = [c for c in task_all if c not in tested]
    print(f'{task}: {len(task_all) - len(task_remaining)}/{len(task_all)} tested, {len(task_remaining)} remaining')
