import mlflow
import sys
import os
import argparse
from mlflow.tracking import MlflowClient

# Add models directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.utils.get_mlflow_uri import get_mlflow_uri

def rollback_model(model_name, target_version):
    mlflow_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    print(f"Connecting to MLflow at {mlflow_uri}...")
    print(f"Rolling back {model_name} to Version: {target_version}")
    
    try:
        # Verify version exists
        client.get_model_version(model_name, target_version)
        
        # Assign @champion alias
        print(f"Assigning alias @champion to version {target_version}...")
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=target_version
        )
        print("Success! Rollback complete.")
        
    except Exception as e:
        print(f"Error rolling back model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollback Champion to a specific version.")
    parser.add_argument("--model", type=str, required=True, help="Registered Model Name")
    parser.add_argument("--target_version", type=str, required=True, help="Version to rollback TO")
    
    args = parser.parse_args()
    rollback_model(args.model, args.target_version)
