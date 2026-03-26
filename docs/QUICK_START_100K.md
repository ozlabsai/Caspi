# Quick Start: 100k Muon + SpecAugment Training

**Goal**: Beat SOTA (5.1% WER) with 100k balanced dataset + Muon optimizer + SpecAugment

**Strategy**: Data efficiency over scale - smart techniques beat brute force

---

## Prerequisites

**You need**:
- Lambda Labs account with access to 8x A100 GPU instance
- HuggingFace account (for Knesset dataset access)
- W&B account (optional but recommended for tracking)

---

## Step-by-Step Guide

### 1. Prepare Local Code

On your Mac:
```bash
cd ~/Documents/ozlabs/labs/caspi

# Files to upload:
# - train_100k_muon.py (training script with Muon + SpecAugment)
# - prepare_100k_dataset.py (dataset preparation)
# - qwen3_asr_data/ (existing 77k crowd-transcribe data)
```

### 2. Upload to Lambda Server

```bash
# Replace <lambda-ip> with your Lambda server IP
scp train_100k_muon.py prepare_100k_dataset.py ubuntu@<lambda-ip>:~/caspi/
scp -r qwen3_asr_data/ ubuntu@<lambda-ip>:~/caspi/
```

### 3. Setup on Lambda Server

SSH into Lambda:
```bash
ssh ubuntu@<lambda-ip>
cd ~/caspi
```

Install dependencies:
```bash
# Install Muon optimizer
uv pip install muon-optimizer

# Verify
uv run python -c "from muon import Muon; print('✓ Muon ready')"
```

### 4. Prepare 100k Dataset

**Option A: With Knesset (Recommended - 100k total)**

```bash
# Login to HuggingFace
hf auth login
# Enter your token from: https://hf.co/settings/tokens

# Prepare 100k dataset (77k crowd + 23k Knesset)
uv run python prepare_100k_dataset.py

# Verify
wc -l qwen3_asr_data/train_100k.jsonl
# Should show: ~100,000 lines
```

**Option B: Without Knesset (77k only - faster start)**

```bash
# Symlink existing data
ln -s qwen3_asr_data/train_ivrit-ai_crowd-transcribe-v5.jsonl qwen3_asr_data/train_100k.jsonl

# Verify
wc -l qwen3_asr_data/train_100k.jsonl
# Should show: 77,166 lines
```

### 5. Setup W&B (Optional)

```bash
# Login
wandb login
# Enter your API key from: https://wandb.ai/authorize

# Set environment
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round25-100k-muon"
```

### 6. Launch Training

```bash
# Launch with torchrun (8x A100)
torchrun --nproc_per_node=8 train_100k_muon.py \
  --train_file ./qwen3_asr_data/train_100k.jsonl \
  --output_dir ./qwen3-asr-hebrew-round25 \
  --batch_size 4 \
  --grad_acc 2 \
  --lr 2e-5 \
  --epochs 3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --use_muon 1 \
  --spec_augment 1 \
  --num_workers 8 \
  --save_steps 500 \
  --wandb_project qwen3-asr-hebrew \
  --wandb_run_name round25-100k-muon
```

**Expected output**:
```
====================================
Loading model: Qwen/Qwen3-ASR-1.7B
====================================

✓ Train examples: 100,000
✓ SpecAugment enabled (time_mask=80, freq_mask=27)
✓ Using Muon optimizer (lr=2e-5)

====================================
Training Configuration
====================================
  GPUs: 8
  Batch size per GPU: 4
  Gradient accumulation: 2
  Effective batch size: 64
  Examples: 100,000
  Epochs: 3
  Steps per epoch: 1,563
  Total steps: 4,689
  LR: 2e-5
  LR scheduler: cosine
  Warmup ratio: 0.1
  Optimizer: Muon
  SpecAugment: Enabled
  Precision: BF16
====================================

Training...
```

### 7. Monitor Training

**Option A: W&B Dashboard**
- Visit: https://wandb.ai/your-username/qwen3-asr-hebrew
- Real-time: loss, LR, GPU utilization

**Option B: Terminal**
```bash
# In a separate SSH session
tail -f qwen3-asr-hebrew-round25/trainer_log.txt

# GPU monitoring
watch -n 1 nvidia-smi
```

### 8. Training Complete

After ~6-8 hours:
```bash
# Model saved at:
ls -lh qwen3-asr-hebrew-round25/

# Should contain:
# - pytorch_model.bin (or model.safetensors)
# - config.json
# - preprocessor_config.json
# - tokenizer files
```

---

## What's Different (vs Previous Rounds)

| Feature | Round 1 | Round 2.5 | This Round |
|---------|---------|-----------|------------|
| Dataset size | ~50k | 3.3M | 100k (balanced) |
| Training method | LoRA | Full FT | Full FT |
| Optimizer | AdamW | AdamW | **Muon** (2× faster) |
| Augmentation | None | SpecAugment | **SpecAugment (on-the-fly)** |
| GPU hours | ~4 | 230+ | **6-8** (Muon speedup) |
| Cost | ~$256 | ~$15,000 | **$400-500** |

---

## Success Criteria

**Training health**:
- ✅ Loss decreasing steadily
- ✅ GPU utilization >85%
- ✅ No OOM errors
- ✅ Checkpoints saved every 500 steps

**Target performance**:
- **eval-d1**: <5.1% WER (beat SOTA)
- **eval-whatsapp**: <8% WER
- **Common Voice Hebrew**: <10% WER

---

## Quick Commands Reference

```bash
# Check training progress
tail -n 50 qwen3-asr-hebrew-round25/trainer_log.txt

# GPU usage
nvidia-smi

# Disk space
df -h

# Kill training (if needed)
pkill -f train_100k_muon

# Resume training
torchrun --nproc_per_node=8 train_100k_muon.py \
  --train_file ./qwen3_asr_data/train_100k.jsonl \
  --output_dir ./qwen3-asr-hebrew-round25 \
  --resume 1  # Auto-finds latest checkpoint
```

---

## Troubleshooting

**"Muon not available"**:
```bash
uv pip install muon-optimizer
# Training will fall back to AdamW automatically
```

**"Knesset dataset access denied"**:
```bash
# Request access: https://huggingface.co/datasets/ivrit-ai/knesset-plenums-whisper-training
# Or use 77k crowd-transcribe only (see Step 4, Option B)
```

**"OOM error"**:
```bash
# Halve batch size, double grad_acc
--batch_size 2 --grad_acc 4  # Maintains effective batch = 64
```

**"Training too slow"**:
```bash
# Increase workers (if CPU has headroom)
--num_workers 16
```

---

## Next Steps

After training:

1. **Download model** from Lambda to local Mac:
   ```bash
   scp -r ubuntu@<lambda-ip>:~/caspi/qwen3-asr-hebrew-round25/ ./
   ```

2. **Evaluate on benchmarks**:
   ```bash
   # (Create evaluation script or use existing benchmarking tools)
   uv run python scripts/eval_model.py \
     --model ./qwen3-asr-hebrew-round25 \
     --test-set eval-d1
   ```

3. **Analyze results**:
   - If WER <5.1% → **DONE! We beat SOTA!**
   - If WER 5.1-6.5% → **Close! Try Phase 2** (core-set + phoneme-aware aug + MWER)
   - If WER >6.5% → **Debug and iterate**

4. **Upload to HuggingFace Hub**:
   ```bash
   huggingface-cli upload OzLabs/Qwen3-ASR-Hebrew-100k ./qwen3-asr-hebrew-round25/
   ```

---

## Estimated Costs

**8x A100 Lambda Labs**:
- Hourly rate: ~$8-12/hour
- Training time: 6-8 hours
- **Total: $48-96**

**Much cheaper than**:
- Round 2.5 (3.3M dataset): $15,000+
- Even with 2× Muon speedup claim, this is 150× more cost-effective!

---

**Ready? Start with Step 1!** 🚀
