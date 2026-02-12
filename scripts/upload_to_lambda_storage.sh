#!/bin/bash
set -e

# Upload prepared dataset to Lambda Cloud Storage
# Usage: ./scripts/upload_to_lambda_storage.sh

echo "=================================================="
echo "Uploading dataset to Lambda Cloud Storage"
echo "=================================================="

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "ERROR: .env file not found"
    exit 1
fi

# Check if dataset exists
if [ ! -d "qwen3_asr_data" ]; then
    echo "ERROR: qwen3_asr_data/ directory not found"
    echo "Run 'uv run python prepare_qwen_data.py' first"
    exit 1
fi

echo ""
echo "Step 1: Compressing dataset..."
echo "Dataset size before compression:"
du -sh qwen3_asr_data/

# Compress with progress (pigz for parallel compression if available)
if command -v pigz &> /dev/null; then
    echo "Using pigz (parallel gzip) for faster compression..."
    tar --use-compress-program=pigz -cf qwen3_asr_data.tar.gz qwen3_asr_data/
else
    echo "Using standard gzip compression..."
    tar -czf qwen3_asr_data.tar.gz qwen3_asr_data/ --checkpoint=1000 --checkpoint-action=dot
fi

COMPRESSED_SIZE=$(du -h qwen3_asr_data.tar.gz | cut -f1)
echo ""
echo "Compressed size: $COMPRESSED_SIZE"

echo ""
echo "Step 2: Uploading to Lambda Cloud Storage..."
echo "Bucket: ozlabs-qwen3-asr"
echo "Key: datasets/qwen3_asr_data.tar.gz"
echo "Endpoint: $S3_ENDPOINT_URL"

# Create bucket if it doesn't exist (ignore error if exists)
aws s3 mb s3://ozlabs-qwen3-asr \
    --endpoint-url "$S3_ENDPOINT_URL" \
    --region "$AWS_REGION" \
    2>/dev/null || echo "(Bucket already exists)"

# Upload with progress
echo "Uploading... (this may take 10-30 minutes depending on internet speed)"
aws s3 cp qwen3_asr_data.tar.gz s3://ozlabs-qwen3-asr/datasets/qwen3_asr_data.tar.gz \
    --endpoint-url "$S3_ENDPOINT_URL" \
    --region "$AWS_REGION"

echo ""
echo "✅ Upload complete!"
echo ""
echo "Dataset location: s3://ozlabs-qwen3-asr/datasets/qwen3_asr_data.tar.gz"
echo "Compressed size: $COMPRESSED_SIZE"
echo ""
echo "Next steps:"
echo "1. Launch Lambda GPU instance (8x H100 or 8x A100)"
echo "2. Clone repo: git clone https://github.com/OzLabs/caspi.git && cd caspi"
echo "3. Copy .env file to GPU instance (contains Lambda storage creds)"
echo "4. Run: ./scripts/download_from_lambda_storage.sh"
echo "5. Run: uv run python train_hebrew_asr_enhanced.py"
echo ""
