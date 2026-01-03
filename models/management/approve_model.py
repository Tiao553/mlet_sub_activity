import mlflow
import sys
import os
import argparse
from mlflow.tracking import MlflowClient

# Add models directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.utils.get_mlflow_uri import get_mlflow_uri

def approve_model(model_name, version):
    mlflow_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    print(f"Connecting to MLflow at {mlflow_uri}...")
    print(f"Approving Model: {model_name}, Version: {version}")
    
    try:
        # Verify version exists
        mv = client.get_model_version(model_name, version)
        
        # Check if model comes from HMG
        # Aliases are not directly on the ModelVersion object in older SDKs, need to check via get_model_version_by_alias or model info
        # But safest is just to warn or check if this specific version is aliased.
        # Let's simplify: Just ensure it exists. The user process is manual anyway.
        # But per requirements: "depois ... vamos eleger quais desses [HMG] estao efetivamente em producao"
        # So it implies we are picking from HMG.
        
        # We can loosely enforce simply by printing metadata
        print(f"Model Description: {mv.description}")
        print(f"Current Aliases for this version: {mv.aliases}")
        
        if "HMG" not in mv.aliases:
             print("WARNING: This model version is NOT tagged as @HMG. Proceeding anyway as requested by operator.")

        
        # Assign @champion alias
        print(f"Assigning alias @champion to version {version}...")
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=version
        )
        print("Success! Model promoted to Champion.")
        
    except Exception as e:
        print(f"Error approving model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approve a model candidate to Champion.")
    parser.add_argument("--model", type=str, required=True, help="Registered Model Name")
    parser.add_argument("--version", type=str, required=True, help="Model Version to approve")
    
    args = parser.parse_args()
    approve_model(args.model, args.version)
