output "crawler_api_endpoint" {
  description = "Public endpoint for FastAPI container exposed via API Gateway"
  value       = "${aws_apigatewayv2_api.crawler_api.api_endpoint}/${aws_apigatewayv2_stage.prod.name}/"
}

output "instance_elastic_ip" {
  description = "Static Elastic IP address of the EC2 instance"
  value       = aws_eip.mlflow_airflow_eip.public_ip
}

output "lambda_repo_url" {
  description = "ECR Repository URL for the Lambda function"
  value       = aws_ecr_repository.lambda_repo.repository_url
}
