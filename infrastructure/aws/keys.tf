# Generate a secure private key
resource "tls_private_key" "generated_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# Create the Key Pair in AWS using the generated public key
resource "aws_key_pair" "generated_key_pair" {
  key_name   = var.key_name
  public_key = tls_private_key.generated_key.public_key_openssh
}

# Save the private key locally for SSH access
resource "local_file" "private_key_pem" {
  content         = tls_private_key.generated_key.private_key_pem
  filename        = "${path.module}/../../${var.key_name}.pem"
  file_permission = "0400"
}
