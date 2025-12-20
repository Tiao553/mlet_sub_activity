# output "crawler_api_endpoint" {
#   description = "Public endpoint for FastAPI container exposed via API Gateway"
#   value       = "${aws_apigatewayv2_api.crawler_api.api_endpoint}/${aws_apigatewayv2_stage.prod.name}/"
# }

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.app_server.public_ip
}
