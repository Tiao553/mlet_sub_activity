#!/bin/bash

# Configuration
PROJECT_DIR="/home/ubuntu/tech-challenge/mlet_sub_activity"

echo "Syncing repository..."
cd "$PROJECT_DIR" || { echo "Directory $PROJECT_DIR not found!"; exit 1; }

# Pull latest changes
git pull origin main

# Fix permissions again if new folders were created
sudo chmod -R 777 airflow

# Restart Airflow to pick up DAG changes (optional but recommended)
echo "Restarting Airflow Scheduler and Webserver..."
sudo docker compose restart airflow-scheduler airflow-webserver

echo "Sync complete!"
