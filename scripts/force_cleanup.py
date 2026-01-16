import boto3
import sys

# Constants for OLD resources (based on error logs)
OLD_BUCKETS = [
    "tech-challanger-4-prd-raw-zone-593793061865",
    "tech-challanger-4-prd-delivery-zone-593793061865",
    "tech-challanger-4-prd-mlflow-artifacts-593793061865"
]
OLD_ECR_REPO = "tech-challanger-4-prd-lambda-repo-tech-challenger-4-prd"

def empty_bucket(bucket_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    try:
        print(f"Emptying bucket: {bucket_name}...")
        bucket.object_versions.delete()
        print(f"Successfully emptied {bucket_name}")
    except Exception as e:
        print(f"Error emptying {bucket_name}: {e}")

def empty_ecr_repo(repo_name):
    ecr = boto3.client('ecr')
    try:
        print(f"Emptying ECR repo: {repo_name}...")
        response = ecr.list_images(repositoryName=repo_name)
        image_ids = response.get('imageIds', [])
        
        if image_ids:
            ecr.batch_delete_image(repositoryName=repo_name, imageIds=image_ids)
            print(f"Successfully deleted {len(image_ids)} images from {repo_name}")
        else:
            print(f"Repo {repo_name} is already empty.")
            
    except Exception as e:
        print(f"Error emptying ECR {repo_name}: {e}")

if __name__ == "__main__":
    print("Starting Force Cleanup of Old Resources...")
    
    # 1. Clean Buckets
    for b in OLD_BUCKETS:
        empty_bucket(b)
        
    # 2. Clean ECR
    empty_ecr_repo(OLD_ECR_REPO)
    
    print("Cleanup Complete. You can now run 'terraform apply'.")
