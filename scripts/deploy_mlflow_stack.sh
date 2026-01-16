#!/bin/bash
set -e

# Initialize Terraform if needed
cd infrastructure/aws
terraform init

# Apply only the resources related to MLflow/Airflow
# Targeting Network, EC2, IAM, and Security Groups.
# Also targeting the buckets variable, but be aware this checks all buckets in the list.
# If you strictly want ONLY the new resources and avoid checking existing buckets, remove the buckets target
# (assuming the bucket already exists or is managed separately).
# However, since we added 'mlflow-artifacts' to the list, we likely need to apply it.
# Terraform is declarative; targeting the resource block handles the count.

TARGETS=""
TARGETS="$TARGETS -target=aws_vpc.main"
TARGETS="$TARGETS -target=aws_subnet.public"
TARGETS="$TARGETS -target=aws_internet_gateway.main"
TARGETS="$TARGETS -target=aws_route_table.public"
TARGETS="$TARGETS -target=aws_route_table_association.public"
TARGETS="$TARGETS -target=aws_security_group.mlflow_airflow_sg"
TARGETS="$TARGETS -target=aws_iam_role.ec2_s3_access_role"
TARGETS="$TARGETS -target=aws_iam_role_policy.s3_access_policy"
TARGETS="$TARGETS -target=aws_iam_instance_profile.ec2_profile"
TARGETS="$TARGETS -target=aws_instance.app_server"
TARGETS="$TARGETS -target=local_file.private_key_pem"
# Targeting all buckets to ensure mlflow-artifacts is created.
# Terraform will not destroy existing ones unless config changed.
TARGETS="$TARGETS -target=aws_s3_bucket.buckets"

echo "Deploying MLflow/Airflow Infrastructure..."
echo "Targets: $TARGETS"

# We use -var 'key_name=...' if not set in tfvars. User can pass args to this script.
# Defaulting to interactive input if not provided.

terraform apply $TARGETS "$@"
