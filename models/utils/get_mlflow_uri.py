import requests
import socket
import logging
import os


import subprocess
import boto3
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_FILE = os.path.expanduser("~/.mlflow_uri_cache")
CACHE_TTL = 3600  # 1 hour

def _read_cache():
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
            
            # Check TTL
            if time.time() - data.get('timestamp', 0) < CACHE_TTL:
                logger.debug(f"Cache hit. URI: {data.get('uri')}")
                return data.get('uri')
            else:
                logger.debug("Cache expired.")
    except Exception as e:
        logger.debug(f"Error reading cache: {e}")
    return None

def _write_cache(uri):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump({'uri': uri, 'timestamp': time.time()}, f)
        logger.debug(f"Cache written for URI: {uri}")
    except Exception as e:
        logger.debug(f"Error writing cache: {e}")

def get_mlflow_uri(port=5000):
    # 1. Check Environment Variable
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        logger.info(f"Using MLflow URI from environment: {env_uri}")
        return env_uri

    # 2. Check Cache
    cached_uri = _read_cache()
    if cached_uri:
        return cached_uri

    # 2. Check Docker Service Name (Internal Docker Network)
    try:
        socket.gethostbyname("mlflow_server")
        uri = f"http://mlflow_server:{port}"
        logger.info(f"Detected Docker Environment. MLflow URI: {uri}")
        _write_cache(uri)
        return uri
    except socket.error:
        pass

    # 3. Check AWS Dynamic IP (Robust against restarts)
    try:
        ec2 = boto3.client('ec2', region_name='us-east-1')
        filters = [
            {'Name': 'tag:Name', 'Values': ['sub-challanger-prd-mlflow-airflow-server']},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ]
        response = ec2.describe_instances(Filters=filters)
        reservations = response.get('Reservations', [])
        if reservations:
            instances = reservations[0].get('Instances', [])
            if instances:
                public_ip = instances[0].get('PublicIpAddress')
                if public_ip:
                    uri = f"http://{public_ip}:{port}"
                    logger.info(f"Detected AWS Instance IP (via Boto3). MLflow URI: {uri}")
                    _write_cache(uri)
                    return uri
    except Exception as e:
        logger.debug(f"Failed to fetch IP via Boto3: {e}")

    # 4. Check Terraform Output (Robust method for this project)
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
                     _write_cache(uri)
                     return uri
    except Exception as e:
        logger.debug(f"Failed to read terraform output: {e}")

    # 5. AWS EC2 Metadata (Fallback)
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
                _write_cache(uri)
                return uri
    except Exception as e:
        logger.debug(f"Could not fetch EC2 metadata: {e}")

    # 6. Fallback to localhost
    uri = f"http://localhost:{port}"
    logger.info(f"Using default MLflow URI: {uri}")
    return uri

if __name__ == "__main__":
    print(get_mlflow_uri())
