# 🚀 Deployment Guide

This project is configured to deploy MLflow and Airflow on AWS using Terraform and Docker.

## Prerequisites

- **Terraform** >= 1.0
- **AWS CLI** configured with credentials (`aws configure`)
- **SSH Client**

## 🛠️ Deploying to AWS

We provide a helper script to automate the deployment process.

1. **Run the Deployment Script**

   Go to the project root and run:

   ```bash
   ./scripts/deploy_mlflow_stack.sh -var="key_name=sub-mlet-mlairflow" -auto-approve
   ```

   **What this does:**
   - Initializes Terraform.
   - Creates AWS resources (VPC, Security Groups, EC2, IAM Roles, S3).
   - Generates an SSH Key Pair (`sub-mlet-mlairflow.pem`) in the project root.
   - Deploys the stack (Airflow + MLflow) on the EC2 instance via `user_data`.

2. **Accessing the Instance**

   The script outputs the **Public IP** of the instance. You can also find it in the Terraform output.

   ```bash
   # Connect via SSH
   ssh -i sub-mlet-mlairflow.pem ubuntu@<PUBLIC-IP>
   ```

   *Note: The key file permissions are automatically set to `400`.*

## 🔍 Verification

Once deployed, wait a few minutes for the services to initialize. You can verify them via SSH:

```bash
ssh -i sub-mlet-mlairflow.pem ubuntu@<PUBLIC-IP> "docker ps"
```

### Access Services

- **MLflow UI**: `http://<PUBLIC-IP>:5000`
- **Airflow UI**: `http://<PUBLIC-IP>:8080`

## ⚠️ Troubleshooting

**Airflow Permissions**
If Airflow fails to start with "Permission denied" errors for logs, the deployment script automatically applies the necessary permissions (`chmod 777`) to the `airflow/` directory on the instance. If you need to fix this manually on an existing instance:

```bash
docker compose restart airflow-init
```

**DAG Synchronization**
Airflow is configured with a **Git-Sync** sidecar.

- DAGs are automatically pulled from the `main` branch of the repository.
- Sync interval: 30 seconds.
- You do **not** need to manually updating the server to deploy new DAGs; just push to GitHub.
