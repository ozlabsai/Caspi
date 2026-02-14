# Complete Deployment Guide: Dataset Prep → Training → SOTA

**Goal**: Beat 5.1% WER with 100k balanced dataset + Muon optimizer + SpecAugment

**Total Time**: ~8-10 hours | **Total Cost**: ~$60-110

---

## Overview

1. **Dataset Preparation** (Linux/Lambda): Create 100k balanced dataset → Upload to HF Hub
2. **Training** (Lambda 8xA100): Train with Muon + SpecAugment for 6-8 hours
3. **Evaluation**: Test on ivrit.ai benchmarks, upload model

---

## Phase 1: Dataset Preparation (On Linux/Lambda)

### Prerequisites
- Linux machine or Lambda instance (Mac won't work - torchcodec issues)
- HuggingFace account + token: https://hf.co/settings/tokens
- Access to `ivrit-ai/knesset-plenums-whisper-training` (may need to request)

### Step 1: Setup on Linux/Lambda

```bash
# SSH into Linux/Lambda server
ssh ubuntu@<your-server>
cd ~
git clone <your-caspi-repo> caspi  # Or upload files
cd caspi

# Install dependencies
uv sync

# Authenticate with HuggingFace
hf auth login
# Paste your token from https://hf.co/settings/tokens
```

### Step 2: Run Dataset Preparation

```bash
# This will:
# - Extract 23k Knesset samples with context
# - Load 77k crowd-transcribe samples
# - Normalize text, split train/eval
# - Upload to HuggingFace Hub

uv run python prepare_hf_dataset_linux.py
```

**Expected Output**:
```
============================================================
Qwen3-ASR Hebrew 100k Dataset Preparation
============================================================

[1/3] Extracting 23,000 Knesset samples with context...
Knesset: 100%|████████| 23000/23000
✓ Extracted 23,000 Knesset samples
  Samples with context: 11,500 (50.0%)

[2/3] Loading 77,166 crowd-transcribe samples...
Crowd-transcribe: 100%|████████| 77166/77166
✓ Loaded 77,166 crowd-transcribe samples

[3/3] Creating HuggingFace Dataset with train/eval split...
✓ Dataset created successfully

Dataset info:
  Train: 90,149 examples (90%)
  Validation: 10,017 examples (10%)
  Total: 100,166 examples

Source distribution (train):
  crowd-transcribe: 69,449 (77.1%)
  knesset: 20,700 (23.0%)

Context statistics (train):
  Samples with context: 10,350 (11.5%)

============================================================
Uploading to HuggingFace Hub
============================================================
Repository: OzLabs/qwen3-asr-hebrew-100k

✓ Dataset uploaded successfully!
Dataset URL: https://huggingface.co/datasets/OzLabs/qwen3-asr-hebrew-100k
```

**Time**: 1-2 hours
**Cost**: Minimal (can use cheap CPU instance)

### Step 3: Verify Upload

Visit: https://huggingface.co/datasets/OzLabs/qwen3-asr-hebrew-100k

You should see:
- Train split: ~90k examples
- Validation split: ~10k examples
- Features: audio, text, context, source, duration

---

## Phase 2: Training (On Lambda 8xA100)

### Step 1: Update Training Script

The `train_100k_muon.py` needs a small update to load from HF Hub.

**Find this section** (around line 451):
```python
raw_ds = load_dataset(
    "json",
    data_files={
        "train": args.train_file,
        ...
    },
)
```

**Replace with**:
```python
# Check if HuggingFace dataset or local JSONL
if "/" in args.train_file and not args.train_file.endswith(".jsonl"):
    # HuggingFace dataset (e.g., "OzLabs/qwen3-asr-hebrew-100k")
    print(f"Loading from HuggingFace Hub: {args.train_file}")
    raw_ds = load_dataset(args.train_file)
else:
    # Local JSONL file
    print(f"Loading from local file: {args.train_file}")
    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            **({"validation": args.eval_file} if args.eval_file else {}),
        },
    )
```

### Step 2: Upload Training Script

```bash
# From your Mac
scp train_100k_muon.py ubuntu@<lambda-ip>:~/caspi/
```

### Step 3: Setup W&B (Optional but Recommended)

```bash
# On Lambda server
wandb login
# Enter your API key from https://wandb.ai/authorize

# Set environment
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="100k-muon-final"
```

### Step 4: Install Muon Optimizer

```bash
# On Lambda server
uv pip install muon-optimizer

# Verify
uv run python -c "from muon import Muon; print('Muon ready!')"
```

### Step 5: Launch Training

```bash
# On Lambda server (8x A100)
torchrun --nproc_per_node=8 train_100k_muon.py \
  --train_file OzLabs/qwen3-asr-hebrew-100k \
  --output_dir ./qwen3-asr-hebrew-100k-final \
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
  --wandb_run_name 100k-muon-final
```

**Expected Output**:
```
====================================
Loading model: Qwen/Qwen3-ASR-1.7B
====================================

Loading from HuggingFace Hub: OzLabs/qwen3-asr-hebrew-100k
✓ Train examples: 90,149
✓ Eval examples: 10,017
✓ SpecAugment enabled (time_mask=80, freq_mask=27)
✓ Using Muon optimizer (lr=2e-5)

====================================
Training Configuration
====================================
  GPUs: 8
  Batch size per GPU: 4
  Gradient accumulation: 2
  Effective batch size: 64
  Examples: 90,149
  Epochs: 3
  Steps per epoch: 1,408
  Total steps: 4,224
  LR: 2e-5
  Optimizer: Muon
  SpecAugment: Enabled
  Precision: BF16
====================================

Training started...
```

**Time**: 6-8 hours
**Cost**: $48-96 (Lambda 8xA100 @ ~$8-12/hr)

### Step 6: Monitor Training

**Option A: W&B Dashboard**
- Visit: https://wandb.ai/your-username/qwen3-asr-hebrew
- Monitor: loss, learning rate, GPU utilization in real-time

**Option B: Terminal**
```bash
# In another SSH session
tail -f qwen3-asr-hebrew-100k-final/trainer_log.txt

# GPU monitoring
watch -n 1 nvidia-smi
```

**Health Checks**:
- ✅ Loss decreasing steadily (no divergence)
- ✅ GPU utilization >85%
- ✅ No OOM errors
- ✅ Checkpoints saved every 500 steps

---

## Phase 3: Evaluation & Deployment

### Step 1: Download Model from Lambda

```bash
# From your Mac
scp -r ubuntu@<lambda-ip>:~/caspi/qwen3-asr-hebrew-100k-final/ ./
```

### Step 2: Evaluate on Benchmarks

```bash
# On Mac or Lambda
uv run python scripts/eval_model.py \
  --model ./qwen3-asr-hebrew-100k-final \
  --test-set eval-d1

# Target: WER < 5.1% (beat SOTA!)
```

### Step 3: Upload to HuggingFace Hub

```bash
# Upload final model
huggingface-cli upload OzLabs/Qwen3-ASR-Hebrew-100k ./qwen3-asr-hebrew-100k-final/

# Add model card with results
# Visit: https://huggingface.co/OzLabs/Qwen3-ASR-Hebrew-100k
# Click "Edit model card" and add WER results
```

---

## Summary

### What We Built

**Dataset**: `OzLabs/qwen3-asr-hebrew-100k`
- 100k balanced samples (77% crowd, 23% Knesset)
- Context-aware (11% samples have previous transcript)
- Normalized Hebrew text
- 90/10 train/eval split

**Model**: `OzLabs/Qwen3-ASR-Hebrew-100k`
- Qwen3-ASR-1.7B fine-tuned on Hebrew
- Muon optimizer (2× training speedup)
- SpecAugment on-the-fly (5-15% WER improvement)
- Cosine LR schedule with warmup

### Cost Breakdown

| Phase | Time | Cost |
|-------|------|------|
| Dataset prep (Lambda CPU) | 1-2 hrs | ~$2-4 |
| Training (8xA100) | 6-8 hrs | $48-96 |
| **Total** | **8-10 hrs** | **$50-100** |

**Compare to**: Round 2.5 with 3.3M dataset would cost $15,000+

### Files Created

- ✅ `prepare_hf_dataset_linux.py` - Dataset preparation script
- ✅ `train_100k_muon.py` - Training script with Muon + SpecAugment
- ✅ `DEPLOYMENT_GUIDE.md` - This guide
- ⬜ `OzLabs/qwen3-asr-hebrew-100k` - HF dataset (after Phase 1)
- ⬜ `OzLabs/Qwen3-ASR-Hebrew-100k` - Trained model (after Phase 3)

---

## Troubleshooting

**Dataset Prep Issues**:

```bash
# Problem: Knesset dataset access denied
# Solution: Request access at:
https://huggingface.co/datasets/ivrit-ai/knesset-plenums-whisper-training

# Problem: Out of disk space
# Solution: Check available space
df -h
# Clean HuggingFace cache if needed
rm -rf ~/.cache/huggingface/datasets/*
```

**Training Issues**:

```bash
# Problem: Muon not found
# Solution: Install it
uv pip install muon-optimizer

# Problem: OOM (Out of Memory)
# Solution: Reduce batch size, increase grad_acc
--batch_size 2 --grad_acc 4  # Maintains effective batch = 64

# Problem: Slow data loading
# Solution: Increase workers
--num_workers 16
```

---

## Quick Reference

```bash
# === PHASE 1: Dataset Prep (Linux) ===
hf auth login
uv run python prepare_hf_dataset_linux.py

# === PHASE 2: Training (Lambda 8xA100) ===
uv pip install muon-optimizer
wandb login
torchrun --nproc_per_node=8 train_100k_muon.py \
  --train_file OzLabs/qwen3-asr-hebrew-100k \
  --output_dir ./qwen3-asr-hebrew-100k-final \
  --batch_size 4 --grad_acc 2 --epochs 3 \
  --use_muon 1 --spec_augment 1

# === PHASE 3: Evaluation (Mac/Lambda) ===
scp -r ubuntu@<lambda>:~/caspi/qwen3-asr-hebrew-100k-final/ ./
uv run python scripts/eval_model.py \
  --model ./qwen3-asr-hebrew-100k-final
huggingface-cli upload OzLabs/Qwen3-ASR-Hebrew-100k \
  ./qwen3-asr-hebrew-100k-final/
```

---

**Ready to start? Begin with Phase 1 on a Linux server!** 🚀
