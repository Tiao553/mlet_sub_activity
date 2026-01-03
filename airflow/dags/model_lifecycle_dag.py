from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os
import mlflow
from mlflow.tracking import MlflowClient

# Ensure we can import from models module (mounted at /opt/airflow/models)
sys.path.append("/opt/airflow")
from models.utils.get_mlflow_uri import get_mlflow_uri

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def promote_logic():
    mlflow_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    print(f"Connecting to MLflow at {mlflow_uri}...")
    print("Starting process to identify Best Models for Homologation (HMG)...")
    
    experiments = client.search_experiments()
    
    for exp in experiments:
        # Filter: Experiment_{SYMBOL}_{PERIOD}_{INTERVAL}
        if not exp.name.startswith("Experiment_"):
            continue
            
        print(f"\nProcessing Experiment: {exp.name}")
        
        # 1. Search for best runs based on RMSE and MAE
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="metrics.rmse < 1000",
            order_by=["metrics.rmse ASC", "metrics.mae ASC"],
            max_results=1
        )
        
        if not runs:
            print(f"  No valid runs found for {exp.name}")
            continue
            
        best_run = runs[0]
        rmse = best_run.data.metrics.get('rmse')
        mae = best_run.data.metrics.get('mae')
        run_id = best_run.info.run_id
        
        if rmse is None:
            print("  Skipping run with no RMSE metric")
            continue

        print(f"  Best Run Identified: {run_id}")
        print(f"  Metrics - RMSE: {rmse}, MAE: {mae}")
        
        # 2. Identify/Construct Model Name
        reg_model_name = exp.name.replace("Experiment_", "model_")
        
        # 3. Register Model Version & Tag
        try:
            # Check if this run is already registered
            versions = client.search_model_versions(f"name='{reg_model_name}' and run_id='{run_id}'")
            if versions:
                target_version = versions[0]
                print(f"  Run already registered as version {target_version.version}")
            else:
                print("  Registering new model version...")
                result = mlflow.register_model(
                    f"runs:/{run_id}/model",
                    reg_model_name
                )
                target_version = client.get_model_version(reg_model_name, result.version)
            
            # 4. Assign @HMG alias
            print(f"  Assigning alias @HMG to version {target_version.version}...")
            client.set_registered_model_alias(
                name=reg_model_name,
                alias="HMG",
                version=target_version.version
            )
            print("  Success: Model tagged as HMG.")
            
        except Exception as e:
            print(f"  Error processing {reg_model_name}: {e}")

with DAG(
    'model_promotion_dag',
    default_args=default_args,
    description='Evaluate Models and Promote to HMG (Python Native)',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Promote to HMG (Python Operator)
    promote_hmg_task = PythonOperator(
        task_id='promote_to_hmg_logic',
        python_callable=promote_logic,
    )

    # Task 2: Notify Human
    notify_human = BashOperator(
        task_id='notify_ready_for_prod',
        bash_command='echo "Models available in HMG! Run approve_model.py to promote to Champion/Production."',
    )

    promote_hmg_task >> notify_human
