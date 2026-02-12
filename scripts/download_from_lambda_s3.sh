#!/bin/bash
set -e

# Download dataset from Lambda Cloud Storage (S3 API)
# Usage: ./scripts/download_from_lambda_s3.sh
#
# This script should be run ON THE GPU INSTANCE

echo "=================================================="
echo "Downloading dataset from Lambda Cloud Storage (S3)"
echo "=================================================="

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "ERROR: .env file not found"
    echo "Copy .env from your local machine or GCP VM"
    exit 1
fi

# Lambda storage bucket UUID
# Friendly name: HEBREW-ASR-TRAIN
BUCKET_NAME="900a8c67-830b-40aa-9bc4-079f4c797735"
S3_KEY="datasets/qwen3_asr_data/"

echo ""
echo "Source: s3://${BUCKET_NAME}/${S3_KEY}"
echo "Target: ./qwen3_asr_data/"
echo "Endpoint: $S3_ENDPOINT_URL"
echo ""

# Download with aws s3 sync
echo "Downloading... (this may take 10-30 minutes)"
time aws s3 sync \
    "s3://${BUCKET_NAME}/${S3_KEY}" \
    ./qwen3_asr_data/ \
    --endpoint-url "$S3_ENDPOINT_URL" \
    --region "$AWS_REGION"

echo ""
echo "Verifying dataset..."
if [ -f "qwen3_asr_data/train.jsonl" ] && [ -f "qwen3_asr_data/eval.jsonl" ] && [ -d "qwen3_asr_data/audio" ]; then
    echo "✅ Dataset structure looks good!"
    echo ""
    echo "Contents:"
    echo "  - train.jsonl: $(wc -l < qwen3_asr_data/train.jsonl) samples"
    echo "  - eval.jsonl: $(wc -l < qwen3_asr_data/eval.jsonl) samples"
    echo "  - audio/: $(find qwen3_asr_data/audio -name '*.wav' | wc -l) WAV files"
    echo ""
    echo "Dataset size:"
    du -sh qwen3_asr_data/
else
    echo "❌ ERROR: Dataset structure incomplete"
    exit 1
fi

echo ""
echo "✅ Dataset ready for training!"
echo ""
echo "Next steps:"
echo "1. Set up W&B: wandb login && export WANDB_PROJECT='qwen3-asr-hebrew'"
echo "2. Start training: uv run python train_hebrew_asr_enhanced.py"
echo ""
