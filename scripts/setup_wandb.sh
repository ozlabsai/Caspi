#!/bin/bash
# Weights & Biases Configuration Helper
# Source this file to set up wandb environment variables

set -e

echo "=================================================="
echo "Weights & Biases Configuration"
echo "=================================================="

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "⚠️  wandb not installed. Installing..."
    uv pip install wandb
fi

# Check if logged in
if ! wandb whoami &>/dev/null; then
    echo ""
    echo "You need to login to Weights & Biases."
    echo "Get your API key from: https://wandb.ai/authorize"
    echo ""
    read -p "Login now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wandb login
    else
        echo "You can login later with: wandb login"
        echo "Or set: export WANDB_API_KEY='your-key'"
    fi
fi

# Set project name
read -p "Project name [qwen3-asr-hebrew]: " project_name
export WANDB_PROJECT="${project_name:-qwen3-asr-hebrew}"

# Set run name
read -p "Run name [round2-gradual-unfreezing]: " run_name
export WANDB_RUN_NAME="${run_name:-round2-gradual-unfreezing}"

# Ask about Phase 0 logging
read -p "Enable Phase 0 audit logging? (y/n) [n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    export WANDB_PHASE0_LOGGING="true"
else
    export WANDB_PHASE0_LOGGING="false"
fi

# Ask about offline mode
read -p "Use offline mode? (sync later) (y/n) [n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    export WANDB_MODE="offline"
    echo "✓ Offline mode enabled (sync later with: wandb sync)"
else
    export WANDB_MODE="online"
fi

echo ""
echo "=================================================="
echo "Configuration Set:"
echo "=================================================="
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_RUN_NAME: $WANDB_RUN_NAME"
echo "WANDB_PHASE0_LOGGING: $WANDB_PHASE0_LOGGING"
echo "WANDB_MODE: $WANDB_MODE"
echo "=================================================="
echo ""
echo "✓ Environment configured!"
echo ""
echo "To save these settings for future sessions, add to your ~/.bashrc or ~/.zshrc:"
echo ""
echo "export WANDB_PROJECT=\"$WANDB_PROJECT\""
echo "export WANDB_RUN_NAME=\"$WANDB_RUN_NAME\""
echo "export WANDB_PHASE0_LOGGING=\"$WANDB_PHASE0_LOGGING\""
echo "export WANDB_MODE=\"$WANDB_MODE\""
echo ""
echo "Now you can run:"
echo "  uv run python scripts/phase0_align_audit.py"
echo "  uv run python train_round2_gradual.py"
echo ""
