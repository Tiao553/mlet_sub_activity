resource "aws_security_group" "mlflow_airflow_sg" {
  name        = "mlflow_airflow_sg"
  description = "Allow inbound traffic for SSH, MLflow, and Airflow"
  vpc_id      = aws_vpc.main.id # Ensure vpc_id is correct relative to network

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "MLflow"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Airflow"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # REMOVED API PORT 8000

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "app_server" {
  ami           = "ami-0c7217cdde317cfec" # Ubuntu 22.04 LTS us-east-1
  instance_type = "t3.medium"
  # Use the generated key pair
  key_name               = aws_key_pair.generated_key_pair.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.mlflow_airflow_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  # User Data to setup environment and launch stack for MLflow & Airflow
  user_data = <<-EOF
              #!/bin/bash
              # Update and Install Dependencies
              apt-get update
              apt-get install -y docker.io git curl
              
              # Install Docker Compose (V2)
              mkdir -p /usr/local/lib/docker/cli-plugins
              curl -SL https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose
              chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
              
              # Enable Docker
              systemctl start docker
              systemctl enable docker
              usermod -aG docker ubuntu
              
              # Setup Project
              mkdir -p /home/ubuntu/tech-challenge
              cd /home/ubuntu/tech-challenge
              
              # Clone the repo (creates mlet_sub_activity directory)
              git clone https://github.com/Tiao553/mlet_sub_activity.git
              
              # Enter the project directory
              cd mlet_sub_activity
              
              # Fix Airflow permissions
              mkdir -p airflow/{dags,logs,plugins}
              chmod -R 777 airflow
              
              # Initialize and start containers
              docker compose up -d
              EOF

  tags = merge(
    local.common_tags,
    {
      Name = "${local.prefix}-mlflow-airflow-server"
    }
  )
}
