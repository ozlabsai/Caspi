#!/bin/bash
# Lambda Labs Training Setup Script for Qwen3-ASR Hebrew Fine-tuning
#
# This script sets up and runs the Hebrew ASR fine-tuning on Lambda Labs GPU instance

set -e  # Exit on error

echo "=========================================="
echo "Qwen3-ASR Hebrew Training - Lambda Labs"
echo "=========================================="

# Install system dependencies (if needed)
echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1 || true

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q torch transformers datasets[audio] accelerate peft librosa soundfile evaluate jiwer scipy

# Authenticate with Hugging Face
echo "Authenticating with Hugging Face..."
if [ -z "$HF_TOKEN" ]; then
    echo "Please enter your Hugging Face token:"
    read -s HF_TOKEN
    export HF_TOKEN
fi

huggingface-cli login --token $HF_TOKEN

# Run the training script
echo ""
echo "Starting training..."
python3 train_qwen3_asr_official.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
