# Round 2 Training Deployment Guide

**Target Team:** GPU Training Team
**Document Version:** 1.0
**Date:** 2026-02-10
**Estimated Cost:** $48
**Estimated Duration:** ~12 hours

---

## Executive Summary

**Objective:** Execute Round 2 training of Qwen3-ASR-Hebrew with gradual unfreezing strategy to improve WER from 12.3% (Round 1 baseline) to 10.5-11.0%.

**Key Changes from Round 1:**
- Hardware: 8x A100 → **2x A100 (40GB)**
- Strategy: Full model training → **Selective layer freezing with gradual unfreezing**
- Quality Gate: None → **Phase 0 forced aligner data audit (CRITICAL)**
- Monitoring: TensorBoard → **Weights & Biases + TensorBoard**
- Training Duration: ~6 hours → **~12 hours** (more epochs, smaller batch)

**Decision Gate:** Phase 0 must pass (>85% data quality) before proceeding to training. Do NOT skip this step.

---

## Table of Contents

1. [Background](#background)
2. [Hardware Requirements](#hardware-requirements)
3. [Prerequisites](#prerequisites)
4. [Setup Instructions](#setup-instructions)
5. [Execution Plan](#execution-plan)
6. [Expected Outcomes](#expected-outcomes)
7. [Monitoring and Validation](#monitoring-and-validation)
8. [Troubleshooting](#troubleshooting)
9. [Success Criteria](#success-criteria)
10. [Cost Breakdown](#cost-breakdown)
11. [Emergency Rollback](#emergency-rollback)

---

## Background

### Round 1 Results (Baseline)

Trained on 8x A100 with full model fine-tuning:

| Dataset | Round 1 WER | Notes |
|---------|-------------|-------|
| eval-d1 | 9.2% | Dialect 1 test set |
| eval-whatsapp | 17.1% | **Noisy WhatsApp audio** |
| hebrew-speech-kan | 14.5% | Broadcast speech |
| saspeech | 8.4% | Clean studio recordings |
| **Average** | **12.3%** | Target: 10.5-11.0% |

### SOTA Comparison

Current SOTA Hebrew ASR models:
- ivrit-ai/whisper-large-v3-ct2: **5.1% WER**
- ivrit-ai/whisper-large-v2.5-turbo-ct2: **6.8% WER**

**Gap to close:** 5-7 WER points to reach SOTA.

### Why Round 2?

**Problem:** Round 1 trained entire 1.7B parameter model, which:
1. Requires 8x A100 (expensive)
2. May overfit on limited Hebrew data
3. Doesn't account for data quality issues

**Solution:** Round 2 implements:
1. **Selective freezing** (~800M trainable vs 1.7B total) → fits on 2x A100
2. **Gradual unfreezing** (coarse-to-fine learning) → better generalization
3. **Phase 0 data audit** → ensure quality before training
4. **Stratified evaluation** → per-domain performance breakdown

---

## Hardware Requirements

### Confirmed Configuration

**Required:**
- **2x NVIDIA A100 (40GB)** GPUs
- 128GB+ system RAM
- 500GB+ SSD storage (for datasets + checkpoints)
- High-bandwidth GPU interconnect (NVLink preferred)
- Linux OS (Ubuntu 20.04+ recommended)

**Network:**
- Stable internet for dataset downloads (~50GB)
- HuggingFace Hub access for model download
- Weights & Biases API access (optional but recommended)

### Memory Budget (Per GPU)

| Component | Memory Usage |
|-----------|--------------|
| Model (BF16) | ~3.4 GB |
| Optimizer states (AdamW) | ~6.8 GB |
| Gradients | ~3.4 GB |
| Activations (batch=2) | ~8-12 GB |
| Gradient checkpointing savings | -40% activations |
| **Total per GPU** | **~28-32 GB** |
| **Safety margin** | **8-12 GB** |

**Critical:** batch_size=2 is tuned for 40GB A100s. Do NOT increase without testing.

### Providers (Recommended)

1. **Lambda Labs** (preferred)
   - 2x A100 (40GB): $3.20/hour
   - Pre-configured PyTorch environment
   - Fast dataset download speeds

2. **RunPod**
   - 2x A100 (40GB): $3.40/hour
   - Flexible on-demand pricing

3. **HuggingFace Spaces**
   - 2x A100: $3.00/hour
   - Built-in HF integration

---

## Prerequisites

### 1. Dependencies

**Python Environment:**
```bash
# Python 3.11+ required
python --version  # Should be 3.11.x or 3.12.x or 3.13.x

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
cd /path/to/caspi
uv sync
```

**Key Dependencies (automatically installed):**
- torch >= 2.0.0
- transformers >= 4.45.0
- datasets >= 2.14.0
- peft >= 0.7.0
- qwen-asr >= 0.0.6
- wandb >= 0.16.0
- librosa, soundfile, jiwer

### 2. HuggingFace Access

**Required for:**
- Model download (Qwen/Qwen3-ASR-1.7B)
- Dataset download (ivrit-ai/crowd-*)
- Model upload (push_to_hub)

**Setup:**
```bash
# Login to HuggingFace
huggingface-cli login
# Paste token from: https://huggingface.co/settings/tokens

# Verify access
huggingface-cli whoami
```

**Required Permissions:**
- Read access to Qwen/Qwen3-ASR-1.7B
- Read access to ivrit-ai datasets
- Write access to your HF namespace (for model upload)

### 3. Weights & Biases (Optional but Recommended)

**Benefits:**
- Real-time training monitoring
- GPU memory tracking
- Experiment comparison
- Automatic checkpoint management

**Setup:**
```bash
# Create free account (no credit card)
# Visit: https://wandb.ai/signup

# Login
wandb login
# Paste API key from: https://wandb.ai/authorize

# Verify
wandb whoami
```

**Free tier includes:**
- 100 GB storage (sufficient)
- Unlimited runs
- Full features

**If skipping W&B:**
```bash
export WANDB_DISABLED="true"
# Training will use TensorBoard only
```

### 4. Environment Variables

**Required:**
```bash
# Memory optimization for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1
```

**Optional (Weights & Biases):**
```bash
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2-2xA100-production"
export WANDB_PHASE0_LOGGING="true"
```

**Optional (HuggingFace Hub):**
```bash
export HF_HOME="/path/to/large/cache"  # If default ~/.cache is too small
export HF_HUB_ENABLE_HF_TRANSFER=1     # Faster downloads
```

### 5. Disk Space

**Required:**
- Datasets: ~50 GB
- Model checkpoints: ~50 GB (3 checkpoints × ~17 GB each)
- Logs and artifacts: ~5 GB
- **Total:** ~105 GB

**Verify:**
```bash
df -h /path/to/caspi
# Should show at least 150 GB free
```

---

## Setup Instructions

### Step 1: Clone and Setup Repository

```bash
# Clone project (if not already done)
git clone <repository-url>
cd caspi

# Verify files exist
ls -la train_round2_gradual.py
ls -la scripts/phase0_align_audit.py
ls -la scripts/eval_round2.py

# Install dependencies
uv sync

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
# Expected output: CUDA available: True, GPUs: 2
```

### Step 2: Configure Environment

**Interactive Setup (Recommended):**
```bash
source scripts/setup_wandb.sh
# This will prompt you for:
# - W&B login
# - Project name
# - Run name
# - Phase 0 logging preference
# - Online/offline mode
```

**Manual Setup:**
```bash
# Set required variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

# Set W&B variables (if using)
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2-production-$(date +%Y%m%d-%H%M)"
export WANDB_PHASE0_LOGGING="true"
```

### Step 3: Pre-download Models and Datasets

**Optional but recommended** (prevents timeout during Phase 0/training):

```bash
# Pre-download Qwen3-ASR model (~17 GB)
python -c "
from transformers import AutoModelForSpeechSeq2Seq
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    'Qwen/Qwen3-ASR-1.7B',
    trust_remote_code=True
)
print('Model downloaded successfully')
"

# Pre-download datasets (~50 GB)
python -c "
from datasets import load_dataset
ds1 = load_dataset('ivrit-ai/crowd-transcribe-v5', split='train')
ds2 = load_dataset('ivrit-ai/crowd-recital-whisper-training', split='train')
print(f'Datasets downloaded: {len(ds1)} + {len(ds2)} samples')
"

# Pre-download forced aligner (for Phase 0)
python -c "
from qwen_asr import Qwen3ForcedAligner
aligner = Qwen3ForcedAligner.from_pretrained('Qwen/Qwen3-ForcedAligner-0.6B')
print('Forced aligner downloaded successfully')
"
```

**If downloads fail:**
- Check HuggingFace token permissions
- Verify network connectivity
- Check disk space
- Try with `HF_HUB_ENABLE_HF_TRANSFER=1`

---

## Execution Plan

### Timeline Overview

| Phase | Duration | Cost | Description |
|-------|----------|------|-------------|
| **Phase 0** | 2-3 hours | $0 (CPU) | Data quality audit |
| **Decision Gate** | 10 min | - | Review Phase 0 results |
| **Phase 2** | ~12 hours | ~$48 | Training (2x A100) |
| **Evaluation** | 1-2 hours | $7 | Model comparison |
| **Total** | ~15-17 hours | ~$55 | End-to-end |

### Phase 0: Data Quality Audit (CRITICAL - DO NOT SKIP)

**Purpose:** Validate that training data quality is sufficient before spending $48 on training.

**Command:**
```bash
uv run python scripts/phase0_align_audit.py
```

**What it does:**
1. Samples 10% of training data (stratified by domain)
   - 40% WhatsApp/noisy samples
   - 40% KAN/clean samples
   - 20% long-tail samples
2. Runs Qwen3-ForcedAligner on each sample
3. Computes alignment coverage metrics
4. Generates decision gate recommendation

**Expected output:**
```
======================================================================
PHASE 0 AUDIT RESULTS
======================================================================

Total samples audited: 2,500
Low quality samples: 187 (7.5%)

Coverage distribution:
  p10: 0.623
  p25: 0.743
  p50: 0.871
  p75: 0.943
  p90: 0.987
  mean: 0.834

Per-domain breakdown:

  whatsapp_noisy:
    Samples: 1,000
    Low quality: 12.3%
    Mean coverage: 0.751
    Length mismatches: 8.2%

  kan_clean:
    Samples: 1,000
    Low quality: 3.8%
    Mean coverage: 0.912
    Length mismatches: 2.1%

  long_tail:
    Samples: 500
    Low quality: 6.2%
    Mean coverage: 0.838
    Length mismatches: 5.8%

======================================================================
DECISION GATE
======================================================================

✅ PROCEED: Only 7.5% of samples have low quality (<10% threshold).
Data quality is acceptable for Round 2 training.
Expected WER improvement from training: 1-2 points.

✓ Full report saved to: ./phase0_audit_results/alignment_report.json
======================================================================
```

**Decision Matrix:**

| Low Quality % | Decision | Action |
|---------------|----------|--------|
| **< 10%** | ✅ **PROCEED** | Continue to Phase 2 training |
| **10-15%** | ⚠️ **CAUTION** | Proceed but monitor closely |
| **> 15%** | ❌ **STOP** | Fix data before training |

**If STOP decision:**
1. Do NOT proceed to training
2. Review `phase0_audit_results/alignment_report.json`
3. Identify problematic domains
4. Contact data team for filtering/resegmentation
5. Expected data cleaning time: 1-2 days

**If PROCEED decision:**
1. Review `alignment_report.json` for insights
2. Note any high-risk domains (e.g., WhatsApp >15%)
3. Continue to Phase 2

**Logs location:**
- Report: `./phase0_audit_results/alignment_report.json`
- W&B: `https://wandb.ai/{username}/qwen3-asr-hebrew/runs/phase0-audit`

**Time estimate:** 2-3 hours (CPU-only, no GPU needed)

### Decision Gate: Review Phase 0 Results

**Required before proceeding:**

1. Open the audit report:
```bash
cat phase0_audit_results/alignment_report.json | jq '.decision'
# Should output: "PROCEED"
```

2. Check overall quality:
```bash
cat phase0_audit_results/alignment_report.json | jq '.low_quality_percentage'
# Should be < 10.0
```

3. Review domain breakdown:
```bash
cat phase0_audit_results/alignment_report.json | jq '.domain_statistics'
# Look for any domain with >20% low quality
```

4. Sign-off (record for audit trail):
```bash
echo "Phase 0 reviewed by: [YOUR_NAME]" >> phase0_signoff.txt
echo "Date: $(date)" >> phase0_signoff.txt
echo "Decision: PROCEED" >> phase0_signoff.txt
```

**If all checks pass:** Continue to Phase 2.

**If any concerns:** Escalate to project lead before proceeding.

### Phase 2: Training Execution

**IMPORTANT:** Only proceed if Phase 0 passed.

**Command:**
```bash
uv run python train_round2_gradual.py
```

**Interactive prompts:**
```
Have you reviewed Phase 0 results? (y/n): y
```

**What happens:**
1. **Initialization (5 min)**
   - Loads Qwen3-ASR-1.7B model (~17 GB)
   - Applies Strategy B freezing (projector + top 12 LLM layers)
   - Sets up LoRA adapters
   - Initializes W&B tracking

2. **Data Loading (15-30 min)**
   - Downloads/caches datasets (~50 GB)
   - Preprocesses audio (16kHz resampling)
   - Normalizes Hebrew text
   - Creates duration-bucketed batches

3. **Training - Strategy B (Epochs 1-2, ~5 hours)**
   - Trains: Projector + Top 12 LLM layers
   - Freezes: ALL audio layers + bottom 16 LLM layers
   - Learning rates:
     - Projector: 2e-4
     - LLM top: 5e-5
     - LM head: 1e-4
   - Checkpoints every 1000 steps

4. **Strategy Switch (Epoch 3, automatic)**
   ```
   ======================================================================
   EPOCH 3: Switching to Strategy A
   ======================================================================
   ✓ Strategy A activated: Projector + Audio Top + LLM Top
   ======================================================================
   ```
   - Unfreezes: Top 8 audio layers (layers.16-23)
   - Adjusts learning rates:
     - Projector: 1e-4 (reduced)
     - Audio top: 3e-5 (newly unfrozen)
     - LLM top: 3e-5 (reduced)
     - LM head: 1e-4 (unchanged)

5. **Training - Strategy A (Epochs 3-5, ~7 hours)**
   - Trains: Projector + Audio Top 8 + LLM Top 12
   - Continues with reduced learning rates
   - More frequent evaluation

6. **Final Evaluation (~10 min)**
   - Computes WER and CER on eval set
   - Saves final model
   - Pushes to HuggingFace Hub

**Expected console output (key milestones):**
```
======================================================================
Round 2 Training Configuration
======================================================================
Optimized for: 2x A100 (40GB)
Strategy: Gradual Unfreezing (B → A)

Configuration:
  Hardware: 2x A100 (40GB)
  Batch size: 2 per GPU
  Gradient accumulation: 16 steps
  Effective batch: 64 (2 × 16 × 2 GPUs)
  Max audio length: 15s (after resegmentation)
  Epochs: 5
  Strategy B (Epochs 1-2): Projector + Top 12 LLM
  Strategy A (Epochs 3-5): + Top 8 Audio layers
======================================================================

✓ Loaded Phase 0 audit results: PROCEED

Initializing W&B run: round2-production-20260210-1430
✓ W&B initialized: https://wandb.ai/username/qwen3-asr-hebrew/runs/...

Loading model...
✓ Model loaded: Qwen/Qwen3-ASR-1.7B

Setting up Strategy B freezing...
✓ Strategy B: Projector + Top 12 LLM trainable

Setting up LoRA...
✓ Trainable: 812,345,678 / 1,715,432,120 (47.35%)

Loading datasets...
✓ ivrit-ai/crowd-transcribe-v5: 45,123 samples
✓ ivrit-ai/crowd-recital-whisper-training: 28,456 samples
✓ Total training samples: 73,579

Preprocessing for training...
100%|████████████████████████| 73579/73579 [15:23<00:00, 79.56it/s]
✓ Preprocessing complete

Optimizer configuration (Epoch 1):
  projector      : LR=2e-04, params=4,194,304
  llm_top        : LR=5e-05, params=487,234,560
  lm_head        : LR=1e-04, params=320,917,814

======================================================================
Starting training...
======================================================================

Epoch 1/5:
  [Step 100/2875] loss: 0.3245, wer: 18.34%, gpu_mem: 31.2GB
  [Step 200/2875] loss: 0.2876, wer: 16.78%, gpu_mem: 31.4GB
  ...
  [Eval 500] eval_loss: 0.2543, eval_wer: 14.56%, eval_cer: 6.23%
  ✓ Checkpoint saved: qwen3-asr-hebrew/checkpoint-500

Epoch 2/5:
  [Step 2875/5750] loss: 0.1987, wer: 13.21%, gpu_mem: 31.3GB
  [Eval 1000] eval_loss: 0.2012, eval_wer: 12.87%, eval_cer: 5.45%
  ✓ Checkpoint saved: qwen3-asr-hebrew/checkpoint-1000

======================================================================
EPOCH 3: Switching to Strategy A
======================================================================
✓ Strategy A activated: Projector + Audio Top + LLM Top

Optimizer configuration (Epoch 3):
  projector      : LR=1e-04, params=4,194,304
  audio_top      : LR=3e-05, params=124,551,168
  llm_top        : LR=3e-05, params=487,234,560
  lm_head        : LR=1e-04, params=320,917,814
======================================================================

Epoch 3/5:
  [Step 3000/5750] loss: 0.1756, wer: 11.98%, gpu_mem: 33.7GB
  [Eval 1500] eval_loss: 0.1845, eval_wer: 11.34%, eval_cer: 4.89%
  ✓ Checkpoint saved: qwen3-asr-hebrew/checkpoint-1500

Epoch 4/5:
  [Step 4000/5750] loss: 0.1623, wer: 11.23%, gpu_mem: 33.6GB
  [Eval 2000] eval_loss: 0.1701, eval_wer: 10.87%, eval_cer: 4.56%

Epoch 5/5:
  [Step 5750/5750] loss: 0.1534, wer: 10.76%, gpu_mem: 33.5GB
  [Final Eval] eval_loss: 0.1689, eval_wer: 10.54%, eval_cer: 4.42%

Saving final model...
✓ Model saved to: ./qwen3-asr-hebrew-round2
✓ Pushed to Hub: username/qwen3-asr-hebrew-round2

======================================================================
Training completed!
Model saved to: ./qwen3-asr-hebrew-round2
======================================================================

✓ Final WER: 10.54%
✓ Final CER: 4.42%
```

**Checkpoints saved:**
- `./qwen3-asr-hebrew-round2/checkpoint-500` (epoch 1, Strategy B)
- `./qwen3-asr-hebrew-round2/checkpoint-1000` (epoch 2, Strategy B)
- `./qwen3-asr-hebrew-round2/checkpoint-1500` (epoch 3, Strategy A)
- `./qwen3-asr-hebrew-round2/` (final model)

**Time estimate:** ~12 hours

**Cost estimate:** 12 hours × $3.20/hour = **~$38.40** (Lambda Labs 2x A100)

### Phase 3: Evaluation and Comparison

**Purpose:** Compare Round 2 vs Round 1 performance across all test sets.

**Command:**
```bash
uv run python scripts/eval_round2.py \
    --round1-model OzLabs/Qwen3-ASR-Hebrew-1.7B \
    --round2-model ./qwen3-asr-hebrew-round2 \
    --output round2_comparison.csv
```

**Note:** This script currently requires manual vLLM server setup. See the script output for instructions.

**Manual evaluation alternative:**
```bash
# Deploy Round 2 model to vLLM
vllm serve ./qwen3-asr-hebrew-round2 --port 8000

# Run benchmarks
uv run python qwen3-asr-hebrew-model/scripts/benchmark.py \
    --server http://localhost:8000/v1 \
    --max-samples 200

# Results saved to: qwen3-asr-hebrew-model/benchmark_results_*.csv
```

**Expected comparison:**

| Dataset | Round 1 WER | Round 2 WER | Improvement |
|---------|-------------|-------------|-------------|
| eval-d1 | 9.2% | 8.1-8.5% | -0.7 to -1.1% |
| eval-whatsapp | 17.1% | 14.8-15.5% | -1.6 to -2.3% |
| hebrew-speech-kan | 14.5% | 12.7-13.2% | -1.3 to -1.8% |
| saspeech | 8.4% | 7.5-7.9% | -0.5 to -0.9% |
| **Average** | **12.3%** | **10.5-11.0%** | **-1.3 to -1.8%** |

**Time estimate:** 1-2 hours (vLLM + benchmarking)

**Cost estimate:** 2 hours × $3.20/hour = **~$6.40**

---

## Expected Outcomes

### Primary Success Metrics

**1. WER Improvement:**
- Target: 12.3% → 10.5-11.0% (conservative estimate)
- Stretch: 10.0-10.5% (if data quality excellent)
- Minimum acceptable: 11.5% (1% improvement)

**2. Per-Domain Performance:**

| Domain | Round 1 | Target | Reasoning |
|--------|---------|--------|-----------|
| eval-d1 | 9.2% | 8.0-8.5% | Clean data, should improve |
| **eval-whatsapp** | 17.1% | **14.5-15.5%** | Highest leverage (noisy) |
| hebrew-speech-kan | 14.5% | 12.5-13.5% | Broadcast, medium improvement |
| saspeech | 8.4% | 7.5-8.0% | Already good, small gains |

**3. Training Stability:**
- No OOM errors during 12-hour run
- GPU memory usage: 28-34 GB per GPU (within 40GB limit)
- Training throughput: 5-7 samples/sec
- Gradient norms: < 10.0 (healthy)
- Eval loss: Monotonically decreasing

**4. Strategy Switch Validation:**
- Strategy B (epochs 1-2): eval_wer decreases from ~18% → ~13%
- Strategy A activation at epoch 3: Small loss spike (expected)
- Strategy A (epochs 3-5): eval_wer decreases from ~13% → ~10.5%

### Secondary Metrics

**Hardware Efficiency:**
- Training cost: ~$48 (vs $77 for Round 1 on 8x A100)
- Training time: ~12 hours (vs ~6 hours, but 53% cost reduction)
- Memory efficiency: 28-34 GB used / 40 GB available (~75-85% utilization)

**Model Quality:**
- CER improvement: 6.5% → 4.5-5.0%
- No catastrophic forgetting on clean domains
- Beam search (beam=5) provides 0.3-0.5% additional WER improvement

**Reproducibility:**
- All hyperparameters logged to W&B
- Phase 0 audit results attached to training run
- Model checkpoints every 1000 steps
- Random seeds set for reproducibility

---

## Monitoring and Validation

### Real-Time Monitoring (Weights & Biases)

**Dashboard URL:** https://wandb.ai/{username}/qwen3-asr-hebrew/runs/{run_id}

**Key charts to monitor:**

1. **Training Loss** (`train/loss`)
   - Should decrease smoothly from ~0.4 → ~0.15
   - Watch for: Sudden spikes (instability), plateaus (underfitting)

2. **Evaluation WER** (`eval/wer`)
   - Should decrease from ~18% → ~10.5%
   - Evaluated every 500 steps
   - Watch for: Increasing trend (overfitting), no improvement (data issues)

3. **GPU Memory** (`system/gpu.0.memory_allocated`)
   - Strategy B (epochs 1-2): 28-32 GB
   - Strategy A (epochs 3-5): 32-34 GB (after unfreezing audio)
   - Watch for: >38 GB (approaching OOM), rapid increases

4. **Learning Rates** (`train/learning_rate_*`)
   - Projector: 2e-4 → 1e-4 at epoch 3
   - LLM top: 5e-5 → 3e-5 at epoch 3
   - Audio top: 0 → 3e-5 at epoch 3 (newly activated)
   - Watch for: All zeros (optimizer issue), wrong values

5. **Gradient Norms** (`train/grad_norm`)
   - Healthy range: 1.0-10.0
   - Watch for: >50 (instability), <0.01 (vanishing gradients)

6. **Training Throughput** (`train/samples_per_second`)
   - Expected: 5-7 samples/sec
   - Watch for: <3 samples/sec (bottleneck), sudden drops

### Console Monitoring (TensorBoard)

If not using W&B:

```bash
# In separate terminal
tensorboard --logdir qwen3-asr-hebrew-round2 --port 6006

# Open in browser
http://localhost:6006
```

### Manual Checkpoints Validation

**Every 2-3 hours, verify:**

```bash
# Check latest checkpoint
ls -lh qwen3-asr-hebrew-round2/checkpoint-*/

# Check eval metrics in trainer_state.json
cat qwen3-asr-hebrew-round2/checkpoint-1000/trainer_state.json | jq '.log_history[-1]'
# Should show decreasing eval_loss and eval_wer
```

### System Monitoring (GPU Health)

```bash
# In separate terminal, monitor GPU usage
watch -n 5 nvidia-smi

# Watch for:
# - GPU utilization: 90-100% (good)
# - GPU memory: 28-34 GB / 40 GB (good)
# - Temperature: <85°C (safe)
# - Power: 300-400W (expected for A100)
```

### Log Files

**Training logs:**
```bash
# Real-time log following
tail -f qwen3-asr-hebrew-round2/training.log

# Search for errors
grep -i "error\|warning\|oom" qwen3-asr-hebrew-round2/training.log
```

### Health Checks During Training

**At epoch boundaries (every ~2.4 hours):**

1. Check eval WER is decreasing
2. Verify GPU memory stable
3. Check no OOM warnings
4. Verify checkpoints saved successfully

**At Strategy A switch (epoch 3, ~5 hours in):**

1. Verify console shows "Switching to Strategy A"
2. Check GPU memory increases by ~2-4 GB (expected)
3. Verify learning rates updated correctly
4. Watch for loss spike (small spike expected, should recover within 50 steps)

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: OOM (Out of Memory) Error

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
GPU 0 memory: 39.8 GB / 40.0 GB
```

**Solutions:**

1. **Reduce batch size** (emergency fix):
```python
# Edit train_hebrew_asr_enhanced.py line 78:
batch_size: int = 1  # Down from 2

# Double gradient accumulation to maintain effective batch:
gradient_accumulation_steps: int = 32  # Up from 16
```

2. **Enable memory optimization:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

3. **Clear cache before restart:**
```python
python -c "import torch; torch.cuda.empty_cache()"
```

4. **Reduce max audio length:**
```python
# Edit train_hebrew_asr_enhanced.py line 57:
max_audio_length_seconds: float = 12.0  # Down from 15.0
```

**Prevention:**
- Monitor GPU memory throughout training
- Stop training before hitting 38 GB
- Use gradient checkpointing (already enabled)

#### Issue 2: Training Loss Not Decreasing

**Symptoms:**
```
Epoch 1: loss=0.45, eval_wer=22%
Epoch 2: loss=0.44, eval_wer=21.5%
Epoch 3: loss=0.44, eval_wer=21%
# Minimal improvement
```

**Root causes:**

1. **Learning rate too low:**
```python
# Edit train_hebrew_asr_enhanced.py line 80:
learning_rate: float = 1e-4  # Up from 5e-5
```

2. **Data quality issues:**
```bash
# Review Phase 0 results
cat phase0_audit_results/alignment_report.json | jq '.low_quality_percentage'
# If >10%, data needs filtering
```

3. **Frozen layers too aggressive:**
```python
# Try unfreezing more LLM layers in setup_round2_freezing_strategy_b():
# Change line: if layer_num >= 16:  # Was: 16, try: 12
```

**Debugging:**
```bash
# Check if gradients are flowing
python -c "
import torch
model = torch.load('qwen3-asr-hebrew-round2/checkpoint-500/pytorch_model.bin')
for name, param in model.items():
    if param.requires_grad:
        print(f'{name}: grad_fn={param.grad_fn}')
"
```

#### Issue 3: Strategy A Switch Causes Loss Spike

**Symptoms:**
```
Epoch 2 final: loss=0.20, eval_wer=13.0%
EPOCH 3: Switching to Strategy A
Epoch 3 step 50: loss=0.35  # Large spike!
```

**Solution:**

This is **expected behavior**! Newly unfrozen layers need to adapt.

**Normal recovery:**
- Loss spikes to 0.30-0.40
- Recovers to 0.25 within 100 steps
- Continues decreasing to 0.15 by epoch 5

**Abnormal (requires intervention):**
- Loss spikes above 0.50
- Does not recover within 200 steps
- Eval WER increases instead of decreases

**Fix if abnormal:**
```python
# Reduce audio_top learning rate in create_param_groups_with_discriminative_lrs():
# Line 653: {"params": param_groups["audio_top"], "lr": 1e-5, ...}  # Was: 3e-5
```

#### Issue 4: Dataset Download Timeout

**Symptoms:**
```
Downloading: ivrit-ai/crowd-transcribe-v5
ReadTimeoutError: Read timed out after 60 seconds
```

**Solutions:**

1. **Increase timeout:**
```bash
export HF_DATASETS_DOWNLOAD_TIMEOUT=300  # 5 minutes
```

2. **Use HF Transfer (faster):**
```bash
pip install hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

3. **Pre-download manually:**
```python
python -c "
from datasets import load_dataset
ds = load_dataset('ivrit-ai/crowd-transcribe-v5', split='train')
print(f'Downloaded: {len(ds)} samples')
"
```

4. **Check network:**
```bash
curl -I https://huggingface.co
# Should return HTTP 200
```

#### Issue 5: Weights & Biases API Rate Limit

**Symptoms:**
```
wandb: ERROR Error uploading: 429 Too Many Requests
```

**Solutions:**

1. **Use offline mode:**
```bash
export WANDB_MODE="offline"
# Restart training

# Sync later when rate limit resets:
wandb sync runs/
```

2. **Reduce logging frequency:**
```python
# Edit train_hebrew_asr_enhanced.py line 824:
logging_steps=100,  # Up from 50
```

3. **Disable W&B temporarily:**
```bash
export WANDB_DISABLED="true"
# Falls back to TensorBoard
```

#### Issue 6: Model Push to Hub Fails

**Symptoms:**
```
Pushing model to Hub...
HTTPError: 401 Unauthorized
```

**Solutions:**

1. **Re-authenticate:**
```bash
huggingface-cli login --token YOUR_TOKEN
```

2. **Check write permissions:**
```bash
huggingface-cli whoami
# Verify you have write access to target namespace
```

3. **Disable Hub push (save locally only):**
```python
# Edit train_hebrew_asr_enhanced.py line 832:
push_to_hub=False,
```

4. **Manual push after training:**
```bash
huggingface-cli upload username/qwen3-asr-hebrew-round2 ./qwen3-asr-hebrew-round2
```

### Emergency Stop Procedures

**Graceful stop (preserves checkpoint):**
```bash
# In training terminal, press Ctrl+C ONCE
# Wait for "Saving checkpoint..." message
# Model saved to latest checkpoint directory
```

**Force stop (may lose current progress):**
```bash
# If Ctrl+C doesn't work:
pkill -f train_round2_gradual.py

# Latest checkpoint still preserved at checkpoint-{last_saved}/
```

**Resume from checkpoint:**
```bash
# Training automatically resumes from latest checkpoint
uv run python train_round2_gradual.py
# Trainer will detect checkpoint-{N}/ and resume
```

### Getting Help

**Before escalating:**
1. Check logs: `qwen3-asr-hebrew-round2/training.log`
2. Check W&B dashboard for anomalies
3. Verify GPU health: `nvidia-smi`
4. Check disk space: `df -h`

**Include in bug report:**
- Full error message and stack trace
- Last 50 lines of training log
- `nvidia-smi` output
- W&B run URL (if available)
- Phase 0 audit results
- Epoch number when error occurred

**Escalation contacts:**
- Project Lead: [Name/Contact]
- HuggingFace Support: support@huggingface.co
- W&B Support: support@wandb.ai

---

## Success Criteria

### Must-Have (Go/No-Go)

- [ ] Phase 0 passes with <10% low quality samples
- [ ] Training completes all 5 epochs without OOM
- [ ] Final model WER < 11.5% (minimum 1% improvement)
- [ ] Model successfully saved and pushed to Hub
- [ ] No data corruption or checkpoint loss

### Should-Have (Success Indicators)

- [ ] WER improves to 10.5-11.0% range (target)
- [ ] WhatsApp domain improves by >1.5% (highest leverage)
- [ ] Strategy A unfreezing improves WER by additional 0.5%
- [ ] Training completes in 11-13 hours (on schedule)
- [ ] GPU memory usage stable at 28-34 GB (no OOM risk)
- [ ] All metrics logged to W&B successfully

### Nice-to-Have (Stretch Goals)

- [ ] WER improves to 10.0-10.5% (better than expected)
- [ ] Beam search provides additional 0.5% improvement
- [ ] Model converges faster than 12 hours
- [ ] CER improves to <4.5%
- [ ] All domains show >1% improvement

---

## Cost Breakdown

### Detailed Cost Estimate

| Phase | Duration | Hardware | Hourly Rate | Cost |
|-------|----------|----------|-------------|------|
| Phase 0 Audit | 2.5 hours | CPU only | $0 | $0 |
| Phase 2 Training | 12 hours | 2x A100 | $3.20 | $38.40 |
| Evaluation | 2 hours | 2x A100 | $3.20 | $6.40 |
| Buffer (restarts) | 1 hour | 2x A100 | $3.20 | $3.20 |
| **Total** | **17.5 hours** | - | - | **$48.00** |

**Note:** Lambda Labs 2x A100 (40GB) = $3.20/hour. Other providers may vary:
- RunPod: $3.40/hour (+$3.20 total)
- HuggingFace Spaces: $3.00/hour (-$3.20 total)

### Cost Optimization Tips

1. **Use spot instances** (if available):
   - Lambda Labs spot: $1.60-2.00/hour (~50% savings)
   - Risk: May be interrupted (use checkpointing)

2. **Shut down during Phase 0:**
   - Phase 0 runs on CPU (local machine)
   - Start GPU instance only after Phase 0 passes

3. **Use smaller eval set:**
   - Reduce eval steps: 500 → 1000 (fewer evaluations)
   - Saves ~1 hour (~$3.20)

4. **Skip evaluation phase:**
   - Use final checkpoint WER as proxy
   - Saves $6.40 (but less thorough validation)

### Budget Alerts

**Set spending alerts:**
- $25: Halfway through training (check if on track)
- $40: Approaching budget limit (training should be nearly done)
- $50: Budget exceeded (investigate overage)

**Lambda Labs:**
```bash
# Check current spending
lambda cloud instances list --show-cost
```

---

## Emergency Rollback

### When to Rollback

**Immediate rollback if:**
- Phase 0 shows >15% low quality samples
- Training loss diverges (increases instead of decreases)
- OOM errors cannot be resolved
- Critical bug discovered in training code

**Consider rollback if:**
- WER not improving after epoch 2
- GPU memory usage unstable
- Training cost exceeds $60

### Rollback Procedure

**Option 1: Use Round 1 Model**
```bash
# Round 1 model is still available:
# OzLabs/Qwen3-ASR-Hebrew-1.7B

# Deploy for inference:
vllm serve OzLabs/Qwen3-ASR-Hebrew-1.7B --port 8000

# Round 1 WER: 12.3% (known baseline)
```

**Option 2: Use Best Round 2 Checkpoint**
```bash
# Find best checkpoint based on eval WER
cat qwen3-asr-hebrew-round2/checkpoint-*/trainer_state.json | \
    jq -r '.log_history[] | select(.eval_wer) | "\(.eval_wer) \(.step)"' | \
    sort -n | head -1

# Deploy best checkpoint:
vllm serve qwen3-asr-hebrew-round2/checkpoint-{BEST}/ --port 8000
```

**Option 3: Re-run with Conservative Settings**
```bash
# Reduce risk by using smaller LR and shorter training:
# Edit train_hebrew_asr_enhanced.py:
learning_rate: float = 2e-5  # More conservative
num_epochs: int = 3  # Shorter training

# Re-run:
uv run python train_round2_gradual.py
```

### Data Preservation

**Always preserve before rollback:**
```bash
# Backup checkpoints
tar -czf round2_checkpoints_backup.tar.gz qwen3-asr-hebrew-round2/

# Backup logs
cp -r qwen3-asr-hebrew-round2/ /backup/round2-$(date +%Y%m%d)/

# Save W&B run URL
echo "https://wandb.ai/username/qwen3-asr-hebrew/runs/{run_id}" > wandb_run_url.txt
```

---

## Appendix

### A. File Manifest

**Training Scripts:**
- `train_round2_gradual.py` - Main training entry point
- `train_hebrew_asr_enhanced.py` - Core training logic with Strategy B/A
- `scripts/phase0_align_audit.py` - Data quality audit
- `scripts/eval_round2.py` - Model comparison

**Configuration:**
- `pyproject.toml` - Dependencies and project config
- `CLAUDE.md` - Project documentation
- `wandb_setup.md` - Experiment tracking guide
- `DEPLOYMENT_ROUND2.md` - This document

**Utilities:**
- `scripts/setup_wandb.sh` - Interactive W&B setup

### B. Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | No | `0,1` | GPUs to use |
| `PYTORCH_CUDA_ALLOC_CONF` | Yes | - | Memory optimization |
| `WANDB_PROJECT` | No | `qwen3-asr-hebrew` | W&B project name |
| `WANDB_RUN_NAME` | No | `round2-gradual-unfreezing` | Run name |
| `WANDB_PHASE0_LOGGING` | No | `false` | Enable Phase 0 logging |
| `WANDB_MODE` | No | `online` | `online` or `offline` |
| `WANDB_DISABLED` | No | `false` | Disable W&B |
| `WANDB_API_KEY` | No | - | W&B authentication |
| `HF_TOKEN` | Yes | - | HuggingFace token |
| `HF_HOME` | No | `~/.cache/huggingface` | HF cache dir |
| `HF_HUB_ENABLE_HF_TRANSFER` | No | `0` | Fast downloads |

### C. Key Hyperparameters Summary

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `batch_size` | 2 | Max for 40GB A100 |
| `gradient_accumulation_steps` | 16 | Effective batch = 64 |
| `learning_rate` | 5e-5 | Conservative base |
| `num_epochs` | 5 | Gradual unfreezing |
| `max_audio_length_seconds` | 15.0 | Post-resegmentation |
| `lora_r` | 16 | Standard LoRA rank |
| `lora_alpha` | 32 | 2× rank (common) |
| Strategy B LR (projector) | 2e-4 | Cross-modal adaptation |
| Strategy B LR (llm_top) | 5e-5 | Language understanding |
| Strategy A LR (projector) | 1e-4 | Reduced for stability |
| Strategy A LR (audio_top) | 3e-5 | Conservative start |
| Strategy A LR (llm_top) | 3e-5 | Reduced for stability |

### D. Contact Information

**Project Team:**
- Project Lead: [Name] - [Email/Slack]
- ML Engineer: [Name] - [Email/Slack]
- Data Team: [Name] - [Email/Slack]

**External Support:**
- HuggingFace: support@huggingface.co
- Weights & Biases: support@wandb.ai
- Lambda Labs: support@lambdalabs.com

**Emergency Escalation:**
- On-call: [Phone/Slack Channel]
- PagerDuty: [Link]

### E. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-10 | Claude | Initial deployment guide |

---

## Sign-Off

**Before starting training:**

- [ ] Read entire deployment guide
- [ ] Verified all prerequisites (Section 3)
- [ ] Tested GPU access and CUDA availability
- [ ] Logged into HuggingFace and W&B
- [ ] Reviewed Phase 0 requirements
- [ ] Understood success criteria
- [ ] Noted emergency contacts

**Acknowledged by:**

Name: ________________
Role: ________________
Date: ________________
Signature: ________________

---

**Ready to proceed? Start with Phase 0:**

```bash
uv run python scripts/phase0_align_audit.py
```

**Good luck! 🚀**
