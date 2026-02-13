#!/bin/bash
set -e

echo "========================================="
echo "Round 2.5 Training Launch"
echo "========================================="
echo ""

# Verify dataset
echo "Checking dataset..."
if [ ! -f "qwen3_asr_data/train.jsonl" ]; then
    echo "ERROR: qwen3_asr_data/train.jsonl not found!"
    echo "Run data prep first: uv run python prepare_qwen_data.py --workers 100"
    exit 1
fi

TRAIN_COUNT=$(wc -l < qwen3_asr_data/train.jsonl)
EVAL_COUNT=$(wc -l < qwen3_asr_data/eval.jsonl)

echo "✓ Dataset ready:"
echo "  Train: $TRAIN_COUNT examples"
echo "  Eval: $EVAL_COUNT examples"
echo ""

# Check GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✓ GPUs detected: $GPU_COUNT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo ""

# Check disk space
echo "Disk space:"
df -h . | tail -1
echo ""

# Verify W&B setup
if [ -z "$WANDB_PROJECT" ]; then
    echo "WARNING: WANDB_PROJECT not set"
    echo "Setting to default: qwen3-asr-hebrew"
    export WANDB_PROJECT="qwen3-asr-hebrew"
fi

if [ -z "$WANDB_RUN_NAME" ]; then
    echo "Setting WANDB_RUN_NAME to: round2.5-$(date +%Y%m%d-%H%M)"
    export WANDB_RUN_NAME="round2.5-$(date +%Y%m%d-%H%M)"
fi

echo "W&B Configuration:"
echo "  Project: $WANDB_PROJECT"
echo "  Run: $WANDB_RUN_NAME"
echo ""

# Training estimates
echo "========================================="
echo "Training Estimates (8x A100 40GB)"
echo "========================================="
echo "Dataset: ~$TRAIN_COUNT examples"
echo "Effective batch: 64 (2 per GPU × 4 grad acc × 8 GPUs)"
echo "Steps per epoch: ~11,000"
echo "Total epochs: 5"
echo "Estimated time: ~30-40 hours (~6-8 hrs/epoch)"
echo "Estimated cost: ~\$2,240 (@\$64/hr)"
echo ""

# Confirm launch
echo "========================================="
read -p "Ready to launch training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "========================================="
echo "Launching training..."
echo "========================================="
echo ""

# Launch training
uv run python train_hebrew_asr_enhanced.py

echo ""
echo "========================================="
echo "Training complete!"
echo "========================================="
echo ""
echo "Model saved to: ./qwen3-asr-hebrew/"
echo "W&B dashboard: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo ""
