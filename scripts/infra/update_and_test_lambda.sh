#!/bin/bash
# Update code on Lambda GPU and test training script

set -e

echo "========================================="
echo "Updating Lambda GPU with latest fixes"
echo "========================================="
echo ""

# SSH connection
LAMBDA_IP="136.116.39.241"
LAMBDA_USER="ubuntu"

echo "Pulling latest code..."
ssh ${LAMBDA_USER}@${LAMBDA_IP} "cd caspi && git pull"

echo ""
echo "Testing training script import..."
ssh ${LAMBDA_USER}@${LAMBDA_IP} "cd caspi && uv run python -c 'from training.train_hebrew_asr_enhanced import setup_round2_freezing_strategy_b; print(\"✓ Import successful\")'"

echo ""
echo "========================================="
echo "✓ Update complete"
echo "========================================="
echo ""
echo "Ready to launch training with:"
echo "  ssh ${LAMBDA_USER}@${LAMBDA_IP}"
echo "  cd caspi"
echo "  bash scripts/launch_training.sh"
echo ""
