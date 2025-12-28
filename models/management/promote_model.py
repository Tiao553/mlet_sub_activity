import mlflow
import os
import sys
from mlflow.tracking import MlflowClient

# Add models directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.utils.get_mlflow_uri import get_mlflow_uri

def promote_models():
    mlflow_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    print(f"Connecting to MLflow at {mlflow_uri}...")
    
    # 1. List all Experiments
    experiments = client.search_experiments()
    
    for exp in experiments:
        if not exp.name.startswith("Experiment_"):
            continue
            
        print(f"Processing Experiment: {exp.name}")
        
        # 2. Find Best Run in Experiment
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.rmse ASC"],
            max_results=1
        )
        
        if not runs:
            print(f"No runs found for {exp.name}")
            continue
            
        best_run = runs[0]
        rmse = best_run.data.metrics.get('rmse')
        run_id = best_run.info.run_id
        print(f"Best Run: {run_id} with RMSE: {rmse}")
        
        # 3. Get Registered Model Name from Run Parameters/Tags
        # We need to reconstruct the name or find the model version linked to this run
        # Strategy: Look for the model version created by this run
        
        # We named registered models as "model_{symbol}_{period}_{interval}"
        # The experiment name is "Experiment_{symbol}_{period}_{interval}"
        # So replace prefix
        reg_model_name = exp.name.replace("Experiment_", "model_")
        
        try:
            # Find the version associated with this run
            versions = client.search_model_versions(f"name='{reg_model_name}'")
            target_version = None
            for v in versions:
                if v.run_id == run_id:
                    target_version = v
                    break
            
            if target_version:
                print(f"Promoting version {target_version.version} of {reg_model_name} to Production")
                client.transition_model_version_stage(
                    name=reg_model_name,
                    version=target_version.version,
                    stage="Production",
                    archive_existing_versions=True
                )
            else:
                print(f"Could not find registered model version for run {run_id}")

        except Exception as e:
            print(f"Error promoting model {reg_model_name}: {e}")

if __name__ == "__main__":
    promote_models()
