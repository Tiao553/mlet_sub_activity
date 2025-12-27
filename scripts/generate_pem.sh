#!/bin/bash

# Configuration
KEY_NAME="${1:-my-key-pair}"
OUTPUT_FILE="../${KEY_NAME}.pem"

# Check if key already exists to avoid overwriting
if [ -f "$OUTPUT_FILE" ]; then
    echo "Error: Key file '$OUTPUT_FILE' already exists."
    exit 1
fi

echo "Generating SSH private key: $OUTPUT_FILE"

# Generate 4096-bit RSA private key
openssl genrsa -out "$OUTPUT_FILE" 4096

# Set permissions to read-only for owner (required for SSH)
chmod 400 "$OUTPUT_FILE"

echo "Success! Key generated at: $(realpath $OUTPUT_FILE)"
