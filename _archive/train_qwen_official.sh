#!/usr/bin/env bash
set -e

echo "=========================================="
echo "Qwen3-ASR Hebrew Training - Official Setup"
echo "=========================================="

# Install required packages
echo ""
echo "Installing qwen-asr, datasets, soundfile, librosa..."
pip install -q -U qwen-asr datasets soundfile librosa pyarrow

# Install Flash Attention 2 for faster training
echo ""
echo "Installing Flash Attention 2..."
MAX_JOBS=4 pip install -q -U flash-attn --no-build-isolation || echo "Warning: Flash Attention 2 installation failed (optional)"

# Prepare data
echo ""
echo "=========================================="
echo "Step 1: Preparing Training Data"
echo "=========================================="
python3 prepare_qwen_data.py

# Run official training
echo ""
echo "=========================================="
echo "Step 2: Starting Qwen3-ASR Training"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./qwen3_asr_data/train.jsonl \
  --eval_file ./qwen3_asr_data/val.jsonl \
  --output_dir ./qwen3-asr-hebrew \
  --batch_size 4 \
  --grad_acc 8 \
  --lr 2e-4 \
  --epochs 3 \
  --log_steps 50 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --num_workers 4 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --prefetch_factor 2

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Model saved to: ./qwen3-asr-hebrew"
