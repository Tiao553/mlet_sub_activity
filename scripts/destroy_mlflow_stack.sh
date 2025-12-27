#!/bin/bash
set -e

# Initialize Terraform (just in case)
cd infrastructure/aws

# Define Targets for Destroy
TARGETS=""
TARGETS="$TARGETS -target=aws_vpc.main"
TARGETS="$TARGETS -target=aws_subnet.public"
TARGETS="$TARGETS -target=aws_internet_gateway.gw"
TARGETS="$TARGETS -target=aws_route_table.public_rt"
TARGETS="$TARGETS -target=aws_route_table_association.public_assoc"
TARGETS="$TARGETS -target=aws_security_group.mlflow_airflow_sg"
TARGETS="$TARGETS -target=aws_iam_role.ec2_s3_access_role"
TARGETS="$TARGETS -target=aws_iam_role_policy.s3_access_policy"
TARGETS="$TARGETS -target=aws_iam_instance_profile.ec2_profile"
TARGETS="$TARGETS -target=aws_instance.app_server"
TARGETS="$TARGETS -target=local_file.private_key_pem"
TARGETS="$TARGETS -target=aws_key_pair.generated_key_pair"
TARGETS="$TARGETS -target=tls_private_key.generated_key"

# Attempt to destroy the MLflow artifact bucket specifically (Index 2 in the list)
TARGETS="$TARGETS -target=aws_s3_bucket.buckets[2]"

echo "Destroying MLflow/Airflow Infrastructure..."
echo "Targets: $TARGETS"

# Using -auto-approve to proceed without interactive prompt since user requested it
terraform destroy -auto-approve $TARGETS -var "key_name=sub-tech-mlet03" "$@"
