#!/bin/bash
# Semi-automated Lambda Labs deployment script
# User launches instance via web UI, then runs this script
#
# Usage: ./deploy_to_lambda_simple.sh <INSTANCE_IP>

set -e

if [ -z "$1" ]; then
    echo "Error: Please provide the instance IP address"
    echo "Usage: ./deploy_to_lambda_simple.sh <INSTANCE_IP>"
    echo ""
    echo "Steps:"
    echo "1. Launch instance at https://cloud.lambdalabs.com/instances"
    echo "2. Copy the instance IP address"
    echo "3. Run: ./deploy_to_lambda_simple.sh <IP>"
    exit 1
fi

INSTANCE_IP="$1"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "============================================================"
echo "Qwen3-ASR Hebrew Training - Lambda Labs Deployment"
echo "============================================================"
echo "Instance IP: $INSTANCE_IP"
echo ""

# Check files exist
if [ ! -f "train_qwen3_asr_official.py" ] || [ ! -f "train_on_lambda.sh" ]; then
    echo "Error: Training files not found in current directory"
    exit 1
fi

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please run: export HF_TOKEN='your_token_here'"
    exit 1
fi

echo "Step 1: Waiting for SSH to be ready..."
sleep 30

echo ""
echo "Step 2: Uploading training files..."
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i "$SSH_KEY" \
    train_qwen3_asr_official.py train_on_lambda.sh \
    ubuntu@$INSTANCE_IP:~/

echo "✓ Files uploaded"

echo ""
echo "Step 3: Making script executable..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i "$SSH_KEY" \
    ubuntu@$INSTANCE_IP \
    "chmod +x train_on_lambda.sh"

echo "✓ Script executable"

echo ""
echo "Step 4: Starting training in tmux..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -i "$SSH_KEY" \
    ubuntu@$INSTANCE_IP \
    "tmux new-session -d -s training && \
     tmux send-keys -t training 'export HF_TOKEN=$HF_TOKEN' C-m && \
     tmux send-keys -t training './train_on_lambda.sh 2>&1 | tee training.log' C-m"

echo "✓ Training started"

echo ""
echo "============================================================"
echo "Deployment Complete!"
echo "============================================================"
echo ""
echo "Training is running in tmux session 'training'"
echo ""
echo "To monitor progress:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
echo "  tmux attach -t training"
echo ""
echo "To detach from tmux: Ctrl+B, then D"
echo ""
echo "To check logs:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
echo "  tail -f training.log"
echo ""
echo "Estimated training time: 6-8 hours"
echo "Estimated cost: \$6-7 total"
echo ""
echo "⚠️  IMPORTANT: Terminate instance when done via Lambda Labs web UI!"
echo "============================================================"
