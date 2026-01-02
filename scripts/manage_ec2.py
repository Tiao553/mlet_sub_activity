import boto3
import sys
import argparse
import time

# Configuration based on Terraform variables
REGION = "us-east-1"
PROJECT_TAG = "sub-challanger"
TAG_NAME_SUFFIX = "mlflow-airflow-server"

def get_instance(ec2):
    """
    Finds the EC2 instance associated with the project.
    """
    print(f"Searching for instance with Tag:Project={PROJECT_TAG} and Name=*{TAG_NAME_SUFFIX} in {REGION}...")
    
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Project', 'Values': [PROJECT_TAG]},
            {'Name': 'tag:Name', 'Values': [f'*{TAG_NAME_SUFFIX}']},
            {'Name': 'instance-state-name', 'Values': ['pending', 'running', 'stopping', 'stopped']}
        ]
    )
    
    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instances.append(instance)
    
    if not instances:
        print("No matching instance found.")
        return None
    
    # Check for multiple instances
    if len(instances) > 1:
        print(f"Warning: Found {len(instances)} instances. Using the first one.")
        
    return instances[0]

def start_instance(ec2, instance_id):
    print(f"Starting instance {instance_id}...")
    ec2.start_instances(InstanceIds=[instance_id])
    print("Waiting for instance to reach 'running' state...")
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    
    # Get public IP
    desc = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = desc['Reservations'][0]['Instances'][0].get('PublicIpAddress', 'N/A')
    print(f"Instance is running. Public IP: {public_ip}")

def stop_instance(ec2, instance_id):
    print(f"Stopping (pausing) instance {instance_id}...")
    ec2.stop_instances(InstanceIds=[instance_id])
    print("Waiting for instance to reach 'stopped' state...")
    waiter = ec2.get_waiter('instance_stopped')
    waiter.wait(InstanceIds=[instance_id])
    print("Instance is stopped.")

def main():
    parser = argparse.ArgumentParser(description='Manage the MLflow/Airflow EC2 instance.')
    parser.add_argument('action', choices=['start', 'stop', 'status'], help='Action to perform: start, stop (pause), or check status.')
    
    args = parser.parse_args()
    
    try:
        ec2 = boto3.client('ec2', region_name=REGION)
    except Exception as e:
        print(f"Error initializing boto3 client: {e}")
        print("Ensure you have AWS credentials configured.")
        sys.exit(1)
    
    instance = get_instance(ec2)
    if not instance:
        sys.exit(1)
        
    instance_id = instance['InstanceId']
    state = instance['State']['Name']
    name_tag = next((tag['Value'] for tag in instance.get('Tags', []) if tag['Key'] == 'Name'), 'Unknown')
    
    print(f"Found Instance: {instance_id} ({name_tag})")
    print(f"Current State: {state}")
    
    if args.action == 'status':
        # Status already printed
        pass
    
    elif args.action == 'start':
        if state == 'running':
            print("Instance is already running.")
        elif state == 'pending':
            print("Instance is pending. Please wait.")
        else:
            start_instance(ec2, instance_id)
            
    elif args.action == 'stop':
        if state == 'stopped':
            print("Instance is already stopped.")
        elif state == 'stopping':
            print("Instance is already stopping.")
        elif state == 'terminated':
             print("Instance is terminated. Cannot stop.")
        else:
            stop_instance(ec2, instance_id)

if __name__ == '__main__':
    main()
