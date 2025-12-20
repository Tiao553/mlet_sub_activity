resource "aws_iam_role" "ec2_s3_access_role" {
  name = "${local.prefix}_ec2_s3_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "s3_access_policy" {
  name = "${local.prefix}_ec2_s3_policy"
  role = aws_iam_role.ec2_s3_access_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:aws:s3:::*mlflow-artifacts*",
          "arn:aws:s3:::*mlflow-artifacts*/*"
          # Allow access to other buckets if needed
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${local.prefix}_ec2_profile"
  role = aws_iam_role.ec2_s3_access_role.name
}
