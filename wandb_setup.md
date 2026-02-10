# Weights & Biases Experiment Tracking Setup

This project uses [Weights & Biases (wandb)](https://wandb.ai) for comprehensive experiment tracking during Round 2 training.

## Quick Start

### 1. Install Dependencies

```bash
uv sync  # wandb>=0.16.0 already included
```

### 2. Create W&B Account (Free Tier)

1. Go to https://wandb.ai/signup
2. Create a free account (no credit card required)
3. Get your API key from https://wandb.ai/authorize

### 3. Login

```bash
wandb login
# Paste your API key when prompted
```

Or set environment variable:

```bash
export WANDB_API_KEY="your-api-key-here"
```

## Configuration

### Basic Configuration

```bash
# Set project name (default: qwen3-asr-hebrew)
export WANDB_PROJECT="qwen3-asr-hebrew"

# Set run name (default: round2-gradual-unfreezing)
export WANDB_RUN_NAME="round2-2xA100-experiment1"
```

### Advanced Configuration

```bash
# Enable offline mode (sync later with: wandb sync)
export WANDB_MODE="offline"

# Disable wandb completely
export WANDB_DISABLED="true"

# Enable Phase 0 audit logging (optional)
export WANDB_PHASE0_LOGGING="true"

# Set entity (for team workspaces)
export WANDB_ENTITY="your-team-name"
```

## What Gets Tracked

### Phase 0 Data Quality Audit

When `WANDB_PHASE0_LOGGING=true`:
- Total samples audited
- Low quality percentage
- Decision gate result (PROCEED/CAUTION/STOP)
- Coverage distribution (p10, p25, median, p75, p90)
- Per-domain statistics (WhatsApp/noisy, KAN/clean, long-tail)
- Alignment report as artifact

### Training (Automatic)

**System Metrics:**
- GPU memory usage
- GPU utilization
- Training throughput (samples/sec)

**Training Metrics:**
- Training loss (per step)
- Evaluation loss (every 500 steps)
- Learning rate schedules (per parameter group)
- Gradient norms

**Hebrew ASR Metrics:**
- WER (Word Error Rate)
- CER (Character Error Rate)
- Per-epoch performance

**Configuration:**
- All hyperparameters
- Model architecture details
- Freezing strategy (B→A switch at epoch 3)
- Hardware config (2x A100)
- Phase 0 audit results (if available)

**Events:**
- Strategy switch (B→A at epoch 3)
- Audio layer unfreezing

### Model Artifacts

- Final trained model checkpoints
- Best model based on WER
- Training configuration

## Usage Examples

### Example 1: Standard Round 2 Training

```bash
# Set up wandb
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2-baseline"

# Run Phase 0 with tracking
export WANDB_PHASE0_LOGGING="true"
uv run python scripts/phase0_align_audit.py

# Run training (wandb auto-enabled)
uv run python train_round2_gradual.py
```

### Example 2: Experiment with Different LRs

```bash
export WANDB_RUN_NAME="round2-higher-lr-experiment"

# Modify train_hebrew_asr_enhanced.py learning_rate
# Then run:
uv run python train_round2_gradual.py
```

### Example 3: Offline Mode (No Internet)

```bash
export WANDB_MODE="offline"
uv run python train_round2_gradual.py

# Later, sync results:
wandb sync runs/
```

### Example 4: Disable Tracking

```bash
export WANDB_DISABLED="true"
uv run python train_round2_gradual.py
# Falls back to TensorBoard only
```

## Viewing Results

### W&B Dashboard

1. Go to https://wandb.ai/{your-username}/qwen3-asr-hebrew
2. View runs, compare experiments, visualize metrics
3. Filter by tags: `round2`, `gradual-unfreezing`, `2xA100`

### Key Visualizations

**Training Progress:**
- Loss curves (training + eval)
- WER/CER over epochs
- Learning rate schedules

**Hardware Monitoring:**
- GPU memory utilization
- Training throughput

**Comparison:**
- Compare multiple runs side-by-side
- Parallel coordinate plots for hyperparameter optimization

**Phase 0 Integration:**
- Data quality metrics linked to training runs
- Coverage distribution histograms
- Per-domain quality breakdown

## Comparing Round 1 vs Round 2

Tag your runs appropriately:

```bash
# Round 1 baseline (for reference)
export WANDB_RUN_NAME="round1-baseline"
export WANDB_TAGS="round1,baseline,8xA100"

# Round 2 experiments
export WANDB_RUN_NAME="round2-gradual-unfreezing"
export WANDB_TAGS="round2,gradual-unfreezing,2xA100"
```

Then use W&B's comparison view to analyze:
- WER improvement trajectory
- Training efficiency (time to target WER)
- Memory usage optimization
- Impact of freezing strategies

## Cost & Quotas

**Free Tier Includes:**
- 100 GB storage
- Unlimited runs
- 1 team member
- Public/private projects

**Pro Tier ($50/month if needed):**
- Unlimited storage
- Team collaboration
- Advanced features

For this project, the **free tier is sufficient**.

## Troubleshooting

### Issue: "wandb not logged in"

```bash
wandb login
```

### Issue: "Rate limited"

```bash
export WANDB_MODE="offline"
# Sync later when rate limit resets
```

### Issue: "Too much data"

```bash
# Reduce logging frequency
export WANDB_LOG_INTERVAL=100  # Default: 50
```

### Issue: "Don't want to use wandb"

```bash
export WANDB_DISABLED="true"
# Or modify train_hebrew_asr_enhanced.py:
# report_to=["tensorboard"]  # Remove "wandb"
```

## Additional Resources

- [W&B Documentation](https://docs.wandb.ai)
- [HuggingFace Trainer Integration](https://docs.wandb.ai/guides/integrations/huggingface)
- [W&B Python API Reference](https://docs.wandb.ai/ref/python)
