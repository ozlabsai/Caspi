#!/bin/bash
###############################################################################
# Deploy Qwen3-ASR Hebrew to Lambda Labs H100
#
# This script downloads the model directly from HuggingFace Hub on the Lambda
# instance for much faster deployment (no local upload needed).
#
# Usage: ./scripts/deploy.sh [LAMBDA_IP]
###############################################################################

set -e

# Configuration
LAMBDA_IP="${1:-209.20.159.167}"
LAMBDA_USER="ubuntu"
MODEL_HF="OzLabs/Qwen3-ASR-Hebrew-1.7B"

echo "=========================================="
echo "Deploying Qwen3-ASR Hebrew to Lambda H100"
echo "Instance: $LAMBDA_IP"
echo "Model: $MODEL_HF"
echo "=========================================="

# Step 1: Wait for SSH
echo ""
echo "Step 1: Waiting for SSH connection..."
for i in {1..30}; do
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ${LAMBDA_USER}@${LAMBDA_IP} "echo 'SSH ready'" 2>/dev/null; then
        echo "✓ SSH connection established"
        break
    fi
    echo "  Attempt $i/30... waiting 5s"
    sleep 5
done

# Step 2: Setup on Lambda (download model from HF Hub)
echo ""
echo "Step 2: Setting up environment and downloading model from HF Hub..."

ssh "${LAMBDA_USER}@${LAMBDA_IP}" << 'ENDSSH'
set -e

echo ""
echo "Installing UV package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo ""
echo "Creating project directory..."
mkdir -p ~/qwen3-asr-hebrew
cd ~/qwen3-asr-hebrew

echo ""
echo "Creating Python environment..."
uv venv --python 3.10

echo ""
echo "Installing vLLM with audio support..."
source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

uv pip install "vllm[audio]"

echo ""
echo "Installing HuggingFace CLI and benchmark dependencies..."
uv pip install huggingface-hub jiwer pandas tqdm datasets soundfile openai torchcodec

echo ""
echo "Downloading model from HuggingFace Hub (OzLabs/Qwen3-ASR-Hebrew-1.7B)..."
echo "This will download ~3.8GB directly on Lambda (fast!)..."
python3 -c "
from huggingface_hub import snapshot_download
model_path = snapshot_download('OzLabs/Qwen3-ASR-Hebrew-1.7B', local_dir='./model', local_dir_use_symlinks=False)
print(f'✓ Model downloaded to: {model_path}')
"

echo ""
echo "✓ Setup complete"

echo ""
echo "Starting vLLM server..."
nohup .venv/bin/vllm serve ./model --host 0.0.0.0 --port 8000 --trust-remote-code > vllm.log 2>&1 &

echo "Waiting for server to start..."
sleep 15

if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ vLLM server is running"
else
    echo "⚠ Server may still be starting, check logs"
fi
ENDSSH

echo ""
echo "=========================================="
echo "✓ Deployment Complete!"
echo "=========================================="
echo ""
echo "Server: http://$LAMBDA_IP:8000"
echo ""
echo "Commands:"
echo "  Test health: ssh ubuntu@$LAMBDA_IP 'curl http://localhost:8000/health'"
echo "  View logs:   ssh ubuntu@$LAMBDA_IP 'tail -f ~/qwen3-asr-hebrew/vllm.log'"
echo "  SSH tunnel:  ssh -L 8000:localhost:8000 ubuntu@$LAMBDA_IP"
echo ""
echo "=========================================="
