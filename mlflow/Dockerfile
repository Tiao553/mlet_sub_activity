# Using Python base to ensure we can install specific pip version
FROM python:3.10-slim

# Install system dependencies for psycopg2 (if needed, binary usually covers it but good to be safe)
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Install MLflow 3.7.0 and dependencies
RUN pip install mlflow==3.7.0 psycopg2-binary boto3

# Expose port (standard MLflow port)
EXPOSE 5000
