# Criar Lambda Function
resource "aws_lambda_function" "crawler_lambda" {
  function_name = "${local.prefix}-api-stock-prediction"
  description   = "Função Lambda para crawler de ações do Yahoo Finance"
  role          = aws_iam_role.lambda_decompress.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.lambda_repo.repository_url}:latest"
  timeout       = 900
  memory_size   = 1024
  architectures = ["x86_64"]

  tags = merge(
    local.common_tags,
  )

  environment {
    variables = {
      API_GATEWAY_ROOT_PATH              = "/prod"
      STAGE                              = "prod"
      MLFLOW_TRACKING_URI                = "http://${aws_eip.mlflow_airflow_eip.public_ip}:5000"
      MLFLOW_BUCKET_NAME                 = "sub-challanger-prd-mlflow-artifacts-593793061865"
      MLFLOW_PYFUNC_DISABLE_ENV_CREATION = "true"
    }
  }

  depends_on = [aws_ecr_repository.lambda_repo]
}
