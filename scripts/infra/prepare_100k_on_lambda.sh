#!/bin/bash
# Run this on Lambda server to prepare 100k balanced dataset

set -e

echo "======================================"
echo "Preparing 100k Dataset on Lambda"
echo "======================================"

cd ~/caspi

# Step 1: Upload local 77k crowd-transcribe data
echo ""
echo "[1/3] Uploading local 77k crowd-transcribe data..."
# (Run from your Mac: scp -r qwen3_asr_data/ ubuntu@lambda-server:~/caspi/)

# Step 2: Download 23k Knesset samples (streaming)
echo ""
echo "[2/3] Downloading 23k Knesset samples..."
uv run python prepare_100k_dataset.py

# Step 3: Verify final dataset
echo ""
echo "[3/3] Verifying merged dataset..."
if [ -f qwen3_asr_data/train_100k.jsonl ]; then
    LINES=$(wc -l < qwen3_asr_data/train_100k.jsonl)
    echo "✓ Dataset ready: $LINES examples"

    echo ""
    echo "Sample entry:"
    head -n 1 qwen3_asr_data/train_100k.jsonl | python3 -m json.tool
else
    echo "✗ ERROR: train_100k.jsonl not found!"
    exit 1
fi

echo ""
echo "======================================"
echo "✓ 100k Dataset Ready for Training"
echo "======================================"
