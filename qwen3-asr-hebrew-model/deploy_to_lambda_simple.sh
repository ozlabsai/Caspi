#!/bin/bash
set -e

LAMBDA_IP="209.20.159.167"
LAMBDA_USER="ubuntu"
LOCAL_MODEL_DIR="/Users/guynachshon/Documents/ozlabs/labs/caspi/qwen3-asr-hebrew-model"

echo "=========================================="
echo "Deploying Qwen3-ASR Hebrew to Lambda H100"
echo "Instance: $LAMBDA_IP"
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

# Step 2: Upload model
echo ""
echo "Step 2: Uploading model files (3.8GB)..."
rsync -avz --progress \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='test_output' \
    --exclude='*.log' \
    --exclude='.venv' \
    "$LOCAL_MODEL_DIR/" \
    "${LAMBDA_USER}@${LAMBDA_IP}:~/qwen3-asr-hebrew-model/"

echo "✓ Model uploaded"

# Step 3: Setup on Lambda
echo ""
echo "Step 3: Setting up environment on Lambda..."

ssh ${LAMBDA_USER}@${LAMBDA_IP} bash << 'REMOTE_SCRIPT'
set -e

echo "Installing UV..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd ~/qwen3-asr-hebrew-model

echo "Creating Python environment..."
uv venv --python 3.10

echo "Installing vLLM with audio support..."
source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

uv pip install "vllm[audio]"

echo "Installing benchmark dependencies..."
uv pip install jiwer pandas tqdm datasets soundfile openai

echo "✓ Setup complete"

echo ""
echo "Starting vLLM server..."
nohup .venv/bin/vllm serve . --host 0.0.0.0 --port 8000 --trust-remote-code > vllm.log 2>&1 &

echo "Waiting for server to start..."
sleep 10

if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ vLLM server is running"
else
    echo "⚠ Server may still be starting, check logs with:"
    echo "  ssh ubuntu@209.20.159.167 'tail -f ~/qwen3-asr-hebrew-model/vllm.log'"
fi
REMOTE_SCRIPT

echo ""
echo "=========================================="
echo "✓ Deployment Complete!"
echo "=========================================="
echo ""
echo "Server: http://$LAMBDA_IP:8000"
echo ""
echo "Test connection:"
echo "  ssh ubuntu@$LAMBDA_IP 'curl http://localhost:8000/health'"
echo ""
echo "View logs:"
echo "  ssh ubuntu@$LAMBDA_IP 'tail -f ~/qwen3-asr-hebrew-model/vllm.log'"
echo ""
echo "Run benchmark:"
echo "  ssh ubuntu@$LAMBDA_IP 'cd ~/qwen3-asr-hebrew-model && .venv/bin/python3 benchmark_on_h100.py'"
echo ""
echo "=========================================="
