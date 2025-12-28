import requests
import socket
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mlflow_uri(port=5000):
    # 1. Check Environment Variable
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        logger.info(f"Using MLflow URI from environment: {env_uri}")
        return env_uri

    # 2. Check Terraform Output (Robust method for this project)
    try:
        # Navigate to terraform dir relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tf_dir = os.path.join(script_dir, "../../infrastructure/aws")
        
        if os.path.exists(tf_dir):
            cmd = ["terraform", "output", "-raw", "instance_public_ip"]
            result = subprocess.run(cmd, cwd=tf_dir, capture_output=True, text=True)
            if result.returncode == 0:
                public_ip = result.stdout.strip()
                if public_ip:
                     uri = f"http://{public_ip}:{port}"
                     logger.info(f"Detected Terraform Managed IP. MLflow URI: {uri}")
                     return uri
    except Exception as e:
        logger.debug(f"Failed to read terraform output: {e}")

    # 3. AWS EC2 Metadata (Fallback)
    try:
        token_url = "http://169.254.169.254/latest/api/token"
        headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
        token_response = requests.put(token_url, headers=headers, timeout=2)
        
        if token_response.status_code == 200:
            token = token_response.text
            meta_url = "http://169.254.169.254/latest/meta-data/public-ipv4"
            headers = {"X-aws-ec2-metadata-token": token}
            ip_response = requests.get(meta_url, headers=headers, timeout=2)
            
            if ip_response.status_code == 200:
                public_ip = ip_response.text
                uri = f"http://{public_ip}:{port}"
                logger.info(f"Detected EC2 Public IP. MLflow URI: {uri}")
                return uri
    except Exception as e:
        logger.debug(f"Could not fetch EC2 metadata: {e}")

    # 4. Fallback to localhost
    uri = f"http://localhost:{port}"
    logger.info(f"Using default MLflow URI: {uri}")
    return uri

if __name__ == "__main__":
    print(get_mlflow_uri())
