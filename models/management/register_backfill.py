import os
import sys

from mlflow.tracking import MlflowClient

import mlflow

# Add models directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from models.utils.get_mlflow_uri import get_mlflow_uri


def backfill_registration():
    mlflow_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    print(f"Connecting to MLflow at {mlflow_uri}...")

    experiments = client.search_experiments()

    for exp in experiments:
        if not exp.name.startswith("Experiment_"):
            continue

        print(f"Checking Experiment: {exp.name}")

        # 1. Register Model Name (if not exists)
        reg_model_name = exp.name.replace("Experiment_", "model_")
        try:
            client.create_registered_model(reg_model_name)
            print(f"  Created Registered Model: {reg_model_name}")
        except:
            pass  # Already exists

        # 2. Find Best Run
        try:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.rmse ASC"],
                max_results=1,
                filter_string="status = 'FINISHED'",
            )
        except Exception as e:
            print(f"  Error searching runs: {e}")
            continue

        if not runs:
            print(f"  No runs found for {exp.name}")
            continue

        best_run = runs[0]
        run_id = best_run.info.run_id
        rmse = best_run.data.metrics.get("rmse")
        print(f"  Best Run: {run_id} (RMSE: {rmse})")

        # 3. Check if this run is registered
        versions = client.search_model_versions(f"name='{reg_model_name}'")
        is_registered = any(v.run_id == run_id for v in versions)

        if is_registered:
            print(f"  Run {run_id} is already registered. Skipping.")
        else:
            print(f"  Registering Run {run_id} as new version...")
            try:
                # We assume the artifact path was 'model' based on training script
                result = mlflow.register_model(
                    model_uri=f"runs:/{run_id}/model", name=reg_model_name
                )
                print(f"  Success! Version {result.version} created.")
            except Exception as e:
                print(f"  Failed to register: {e}")


if __name__ == "__main__":
    backfill_registration()
