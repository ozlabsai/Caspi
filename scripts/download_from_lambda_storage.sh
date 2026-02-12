#!/bin/bash
set -e

# Download prepared dataset from Lambda Cloud Storage
# Usage: ./scripts/download_from_lambda_storage.sh
#
# This script should be run ON THE GPU INSTANCE after:
# 1. Cloning the repo
# 2. Copying .env file with Lambda storage credentials

echo "=================================================="
echo "Downloading dataset from Lambda Cloud Storage"
echo "=================================================="

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "ERROR: .env file not found"
    echo "Copy .env from your local machine to this GPU instance"
    exit 1
fi

echo ""
echo "Step 1: Downloading compressed dataset..."
echo "Source: s3://ozlabs-qwen3-asr/datasets/qwen3_asr_data.tar.gz"
echo "Endpoint: $S3_ENDPOINT_URL"
echo ""
echo "This should be FAST (internal Lambda network transfer)"
echo ""

# Download from Lambda storage
time aws s3 cp s3://ozlabs-qwen3-asr/datasets/qwen3_asr_data.tar.gz . \
    --endpoint-url "$S3_ENDPOINT_URL" \
    --region "$AWS_REGION"

COMPRESSED_SIZE=$(du -h qwen3_asr_data.tar.gz | cut -f1)
echo ""
echo "Downloaded size: $COMPRESSED_SIZE"

echo ""
echo "Step 2: Extracting dataset..."
echo "This will take 5-15 minutes..."

# Extract with progress (pigz for parallel decompression if available)
if command -v pigz &> /dev/null; then
    echo "Using pigz (parallel gzip) for faster extraction..."
    tar --use-compress-program=pigz -xf qwen3_asr_data.tar.gz
else
    echo "Using standard gzip extraction..."
    tar -xzf qwen3_asr_data.tar.gz --checkpoint=1000 --checkpoint-action=dot
fi

echo ""
echo "Extracted dataset size:"
du -sh qwen3_asr_data/

echo ""
echo "Step 3: Verifying dataset structure..."
if [ -f "qwen3_asr_data/train.jsonl" ] && [ -f "qwen3_asr_data/eval.jsonl" ] && [ -d "qwen3_asr_data/audio" ]; then
    echo "✅ Dataset structure looks good!"
    echo ""
    echo "Contents:"
    echo "  - train.jsonl: $(wc -l < qwen3_asr_data/train.jsonl) samples"
    echo "  - eval.jsonl: $(wc -l < qwen3_asr_data/eval.jsonl) samples"
    echo "  - audio/: $(find qwen3_asr_data/audio -name '*.wav' | wc -l) WAV files"
else
    echo "❌ ERROR: Dataset structure incomplete"
    echo "Expected: qwen3_asr_data/{train.jsonl, eval.jsonl, audio/*.wav}"
    exit 1
fi

echo ""
echo "Step 4: Cleaning up compressed file..."
rm qwen3_asr_data.tar.gz
echo "Removed qwen3_asr_data.tar.gz"

echo ""
echo "✅ Dataset ready for training!"
echo ""
echo "Next steps:"
echo "1. Set up W&B: wandb login && export WANDB_PROJECT='qwen3-asr-hebrew' && export WANDB_RUN_NAME='round2.5-full-finetuning'"
echo "2. Start training: uv run python train_hebrew_asr_enhanced.py"
echo "3. Monitor progress: https://wandb.ai/your-username/qwen3-asr-hebrew"
echo ""
