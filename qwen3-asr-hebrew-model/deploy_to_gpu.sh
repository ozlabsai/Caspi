#!/bin/bash
###############################################################################
# Deploy Qwen3-ASR Hebrew to GPU Instance (Lambda Labs, RunPod, etc.)
#
# Usage:
#   1. Upload model to GPU instance:
#      rsync -avz --progress qwen3-asr-hebrew-model/ user@gpu-host:~/qwen3-asr-hebrew-model/
#
#   2. SSH to GPU instance and run this script:
#      ssh user@gpu-host
#      cd ~/qwen3-asr-hebrew-model
#      chmod +x deploy_to_gpu.sh
#      ./deploy_to_gpu.sh
###############################################################################

set -e  # Exit on error

echo "=========================================="
echo "Qwen3-ASR Hebrew Deployment"
echo "=========================================="

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. GPU may not be available."
    echo "   Continuing with CPU deployment..."
else
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Install UV if not present
if ! command -v uv &> /dev/null; then
    echo ""
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    echo "✓ UV installed"
else
    echo "✓ UV already installed"
fi

# Create virtual environment
echo ""
echo "Setting up Python environment..."
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
uv pip install -e .

echo ""
echo "✓ Dependencies installed"

# Check model files
echo ""
echo "Verifying model files..."
REQUIRED_FILES=(
    "model.safetensors"
    "config.json"
    "tokenizer.json"
    "preprocessor_config.json"
    "chat_template.json"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Missing required files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please ensure all model files are uploaded to this directory."
    exit 1
fi

echo ""
echo "✓ All required files present"

# Test model loading
echo ""
echo "Testing model loading..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Start server
echo ""
echo "=========================================="
echo "Starting API Server"
echo "=========================================="
echo ""
echo "Server will be available at:"
echo "  - Local: http://localhost:8000"
echo "  - Health: http://localhost:8000/health"
echo "  - Docs: http://localhost:8000/docs"
echo ""
echo "To run in background:"
echo "  nohup uv run fastapi serve_asr.py --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
echo ""
echo "To test from another machine:"
echo "  curl http://$(hostname -I | awk '{print $1}'):8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start server (foreground)
exec uv run fastapi serve_asr.py --host 0.0.0.0 --port 8000
