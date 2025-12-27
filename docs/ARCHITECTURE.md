# 🏗️ Architecture

This project utilizes a cloud-native architecture on AWS to serve ML models and orchestrate pipelines.

## ☁️ Cloud Infrastructure (AWS)

The infrastructure is provisioned as code (IaC) using **HashiCorp Terraform**.

### Components

- **Compute (EC2)**: A `t3.medium` instance hosting the Docker stack.
- **Network (VPC)**: Custom VPC with a public subnet, Internet Gateway, and Route Table.
- **Security (SG)**:
  - `22` (SSH): Open for management.
  - `5000` (MLflow): Open for ML tracking UI.
  - `8080` (Airflow): Open for Workflow UI.
- **Storage (S3)**:
  - `raw-zone`: Storage for raw data.
  - `delivery-zone`: Storage for processed data.
  - `mlflow-artifacts`: Dedicated bucket for ML model artifacts.
- **IAM**: EC2 Instance Profile with read/write access to the specific S3 buckets.

## 🐳 Application Stack (Docker)

The application runs as a containerized stack managed by **Docker Compose**.

### Services

1. **MLflow Server**:
   - Tracks experiments and models.
   - Backend Store: Postgres.
   - Artifact Store: **AWS S3** (`s3://mlflow-artifacts`).
2. **Airflow**:
   - Orchestrates data pipelines and model training.
   - Components: Webserver, Scheduler, Init.
   - Backend Store: Postgres.
3. **PostgreSQL**:
   - Two distinct containers for MLflow and Airflow metadata.

## 🔄 Workflow

1. **Deployment**: `deploy_mlflow_stack.sh` provisions the EC2 instance.
2. **Initialization**: `user_data` script installs Docker, clones the repo, fixes permissions, and starts `docker compose up`.
3. **Pipelines**: Airflow DAGs run ML tasks, logging metrics and models to MLflow.
4. **Artifacts**: Models are physically stored in S3, accessible via the MLflow UI.
