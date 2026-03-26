# Round 2.5 Training Launch Guide

**Goal:** Beat ivrit.ai SOTA (5.1% WER on eval-d1) using full fine-tuning + 5 SOTA methods

**Expected Result:** 7-9% WER (30-40% relative improvement over Round 1's 12.3%)

---

## Quick Start

```bash
# 1. Prepare dataset locally (Mac or cheap CPU instance)
uv run python prepare_qwen_data.py
./scripts/upload_to_lambda_storage.sh

# 2. On GPU instance
git clone https://github.com/OzLabs/caspi.git && cd caspi
# Copy .env file to GPU instance
./scripts/download_from_lambda_storage.sh
wandb login && export WANDB_PROJECT="qwen3-asr-hebrew" && export WANDB_RUN_NAME="round2.5"
uv run python train_hebrew_asr_enhanced.py
```

---

## What We Implemented (Changes from Round 1)

### 1. Full Fine-Tuning (No LoRA)
- **Why:** Maximum WER performance, standalone model output
- **Cost:** More VRAM, but we have 8x A100 (40GB each) = plenty of headroom
- **Result:** Complete model checkpoint (not adapters)

### 2. Synthetic Audio Augmentation
- **Speed perturbation:** 0.9x, 1.0x, 1.1x (70% probability)
- **Pitch shifting:** ±2 semitones (30% probability)
- **Background noise:** SNR 15-30 dB (40% probability)
- **Expected impact:** 2-5% relative WER reduction

### 3. Timestamp Preservation (ivrit.ai method)
- **40% of samples keep Whisper-style timestamps** `<|0.00|>...<|2.34|>`
- **Why:** Helps model learn temporal alignment
- **Expected impact:** 1-3% relative WER reduction

### 4. Previous Context (ivrit.ai method)
- **50% of samples use previous transcript as context**
- **Why:** Better fluency and speaker continuity
- **Expected impact:** 2-4% relative WER reduction

### 5. Balanced Dataset Sampling (ivrit.ai method)
- **50% Knesset** (formal, high-quality)
- **30% Transcribe** (crowd, medium quality)
- **20% Recital** (crowd, informal)
- **Why:** Better generalization vs ivrit.ai's 90% Knesset bias
- **Expected impact:** 5-10% relative WER reduction on informal test sets

### 6. Model Averaging (ivrit.ai method)
- **Average 3 best checkpoints by eval loss**
- **Why:** Ensemble-like benefits, smoother predictions
- **Expected impact:** 2-5% relative WER reduction

### 7. W&B Live Tracking
- **WER/CER tracked during training** (not just at eval steps)
- **Visible in W&B dashboard** alongside loss curves
- **Final summary metrics** on run page for easy comparison

---

## Dataset Overview

**Full Training Set (Round 2.5):**
- **Knesset:** ~13,000 hours (150 GB) - Formal Hebrew, parliamentary proceedings
- **Transcribe:** ~200 hours - Crowd-sourced diverse content
- **Recital:** ~100 hours - Informal spoken Hebrew
- **Total:** ~13,300 hours

**Sampling Strategy:**
- 50% Knesset (not 90% like ivrit.ai - better generalization)
- 30% Transcribe
- 20% Recital

**On-the-fly Augmentation:**
- Audio augmentation applied during training (speed/pitch/noise)
- Timestamp preservation randomized per sample (40% keep)
- Previous context randomized per sample (50% include)

---

## Cost-Optimized Workflow with Lambda Storage

### Why Lambda Storage?

**Problem:** Downloading Knesset directly on GPU wastes $256 (4 hours × $64/hr)

**Solution:** Use Lambda Cloud Storage as intermediate cache
- Prep data on cheap CPU or local Mac (free or ~$0.20)
- Upload to Lambda storage once (10-30 min)
- Download to GPU blazing fast (internal Lambda network, 5-10 min)
- Reuse for future training runs

**Total Cost Savings:** $250+ per training run

### Phase 1: Data Preparation (Local Mac or Cheap CPU)

**Option A: Prepare on Mac (RECOMMENDED for first time)**

```bash
# 1. Prepare dataset (downloads Knesset + Transcribe + Recital)
uv run python prepare_qwen_data.py

# Expected output: ./qwen3_asr_data/ (~150 GB)
# Time: 4-8 hours (downloading Knesset is the bottleneck)
# Cost: FREE (runs on your Mac)

# 2. Upload to Lambda storage
./scripts/upload_to_lambda_storage.sh

# Time: 10-30 minutes (depends on your upload speed)
# Cost: FREE (Lambda storage is included)
```

**Option B: Prepare on Cheap CPU Instance (for faster upload)**

```bash
# 1. Launch Lambda CPU instance (8 vCPU, ~$0.04/hr)
# Or any cheap cloud CPU instance

# 2. Clone repo and install deps
git clone https://github.com/OzLabs/caspi.git && cd caspi
uv sync

# 3. Copy .env file (contains Lambda storage creds)
# scp .env cpu-instance:/workspace/caspi/.env

# 4. Prepare dataset
uv run python prepare_qwen_data.py

# 5. Upload to Lambda storage
./scripts/upload_to_lambda_storage.sh

# 6. Terminate CPU instance
# Total time: 4-8 hours
# Total cost: $0.16-0.32
```

### Phase 2: Training (8x A100 or 8x H100 GPU Instance)

**1. Launch GPU Instance**

```bash
# Lambda Labs: 8x A100 (40GB) - $64/hr
# Or: 8x H100 (80GB) - higher cost but faster

# Launch via Lambda dashboard or CLI
```

**2. Setup on GPU Instance**

```bash
# Clone repo
git clone https://github.com/OzLabs/caspi.git && cd caspi

# Install dependencies
uv sync

# Copy .env file from local machine
# scp .env gpu-instance:/workspace/caspi/.env
# Or manually create .env with Lambda storage credentials
```

**3. Download Dataset from Lambda Storage**

```bash
# This is FAST (internal Lambda network, 5-10 minutes)
./scripts/download_from_lambda_storage.sh

# The script will:
# - Download qwen3_asr_data.tar.gz from Lambda storage (~5-10 min)
# - Extract dataset (~5-15 min)
# - Verify structure
# - Clean up compressed file
```

**4. Setup W&B Tracking**

```bash
# Login to W&B
wandb login

# Set experiment name
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2.5-full-finetuning"

# Optional: Add notes
export WANDB_NOTES="Round 2.5: Full fine-tuning + 5 SOTA methods (audio aug, timestamps, context, balanced sampling, model averaging)"
```

**5. Launch Training**

```bash
# Start training (8-12 hours)
uv run python train_hebrew_asr_enhanced.py

# The script will:
# - Load datasets with balanced sampling (50-30-20)
# - Apply audio augmentation on-the-fly
# - Use gradual unfreezing (Strategy B→A at epoch 3)
# - Log WER/CER live to W&B
# - Save checkpoints every 100 steps
# - Average best 3 checkpoints at end
```

**6. Monitor Training**

```bash
# W&B dashboard: https://wandb.ai/your-username/qwen3-asr-hebrew
# Watch for:
# - Loss breaking through 11.3 plateau (Round 1 got stuck here)
# - WER dropping below 10% by epoch 3-4
# - CER tracking WER (roughly WER × 0.5-0.6)
# - Strategy B→A switch at epoch 3 (slight loss spike is normal)

# Console output shows eval results every 100 steps:
# ======================================================================
# Evaluation at step 500 (epoch 2.1):
#   WER: 9.8%
#   CER: 5.2%
#   Loss: 0.0823
# ======================================================================
```

**7. Training Completion**

```bash
# After 5 epochs (~8-12 hours):
# - Final model: ./qwen3-asr-hebrew-round2.5/
# - Averaged model: ./qwen3-asr-hebrew-round2.5-averaged/
# - Use averaged model for evaluation (best performance)
```

### Phase 3: Evaluation

**1. Download Model to Local Machine**

```bash
# Option A: rsync from GPU instance
rsync -avz --progress gpu-instance:/workspace/caspi/qwen3-asr-hebrew-round2.5-averaged/ ./qwen3-asr-hebrew-round2.5-averaged/

# Option B: Upload to HF Hub first, then download
# (On GPU instance)
huggingface-cli upload OzLabs/Qwen3-ASR-Hebrew-Round2.5 ./qwen3-asr-hebrew-round2.5-averaged/

# (On local Mac)
huggingface-cli download OzLabs/Qwen3-ASR-Hebrew-Round2.5 --local-dir ./qwen3-asr-hebrew-round2.5-averaged/
```

**2. Run Full Benchmark**

```bash
# Evaluate on all 6 ivrit.ai test sets
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model ./qwen3-asr-hebrew-round2.5-averaged/ \
    --output results_round2.5.json

# Expected output (target):
# {
#   "eval-d1": {"wer": 0.078, "cer": 0.042, "sota": 0.051, "vs_sota": "+53%"},
#   "whatsapp": {"wer": 0.095, "cer": 0.055, "sota": 0.072, "vs_sota": "+32%"},
#   "saspeech": {"wer": 0.088, "cer": 0.051, "sota": 0.064, "vs_sota": "+38%"},
#   "fleurs": {"wer": 0.195, "cer": 0.112, "sota": 0.174, "vs_sota": "+12%"},
#   "commonvoice": {"wer": 0.168, "cer": 0.095, "sota": 0.149, "vs_sota": "+13%"},
#   "kan": {"wer": 0.104, "cer": 0.061, "sota": 0.081, "vs_sota": "+28%"}
# }

# Time: 2-4 hours (all test sets)
# Can run on Mac (CPU inference is fine for eval)
```

**3. Quick Validation (Optional)**

```bash
# Fast check on 100 samples from eval-d1 (1-2 minutes)
uv run python scripts/quick_eval.py \
    --model ./qwen3-asr-hebrew-round2.5-averaged/ \
    --test-set eval-d1 \
    --max-samples 100

# Use this for quick sanity check before full benchmark
```

---

## Training Configuration

**Hardware:**
- 8x A100 (40GB) via Lambda Labs
- Cost: ~$64/hr
- Expected runtime: 8-12 hours (training only, dataset already prepped)

**Hyperparameters:**
- Learning rate: 5e-5 (discriminative layer-wise)
  - Projector: 5e-5
  - Audio encoder: 2.5e-5
  - LLM: 1e-5
- Batch size: 4 per GPU × 8 accumulation × 8 GPUs = 256 effective
- Epochs: 5 (gradual unfreezing)
  - Epochs 1-2: Strategy B (freeze audio encoder)
  - Epochs 3-5: Strategy A (unfreeze all)
- Eval steps: 100
- Save steps: 100
- Warmup ratio: 0.05

**Full Fine-Tuning (No LoRA):**
- All model weights trainable
- Outputs standalone model (not adapters)
- Ready for production deployment

---

## Success Criteria

### Target Performance (Round 2.5)
- **eval-d1 WER: 7-9%** (vs Round 1: 12.3%, SOTA: 5.1%)
- **Relative improvement: 30-40%** over Round 1
- **Gap to SOTA: Reduced from 7.2% → 2-4%**

### What to Watch During Training

**Good signs:**
- ✅ Loss breaks through 11.3 plateau (Round 1 got stuck here)
- ✅ WER drops below 10% by epoch 3-4
- ✅ CER tracks WER (roughly WER × 0.5-0.6)
- ✅ Eval loss continues improving (no overfitting)
- ✅ Strategy B→A switch at epoch 3 causes slight loss spike (expected)

**Bad signs:**
- ❌ Loss stuck at 11.3 for >2 epochs (same as Round 1)
- ❌ WER above 12% after epoch 3 (no improvement)
- ❌ Eval loss increasing while train loss decreasing (overfitting)
- ❌ GPU memory errors (shouldn't happen with 8x A100)

### Decision Point: Round 3?

**If Round 2.5 achieves 7-9% WER:**
- **Gap to SOTA: 2-4%** (still significant but within reach)
- **Round 3 options:**
  - GRPO (RL-based WER optimization) - most promising
  - Self-training (pseudo-label unlabeled Hebrew audio)
  - Text augmentation (LLM-generated transcript variants)

**If Round 2.5 achieves <7% WER:**
- **SOTA is within reach!** (~2% gap)
- Focus on GRPO for final push (direct WER optimization)

**If Round 2.5 still at 10-12% WER:**
- Something wrong with implementation (debug before Round 3)
- Check augmentation actually applies
- Verify balanced sampling works
- Review W&B logs for anomalies

---

## Cost Summary

### Total Estimated Cost (with Lambda Storage)

**Data Preparation:**
- Option A (Mac): FREE, 4-8 hours download + 10-30 min upload
- Option B (Cheap CPU): $0.16-0.32, 4-8 hours total

**Lambda Storage:**
- Storage cost: ~$0.10/month for 150GB (negligible)
- Transfer cost: FREE (internal Lambda network)

**GPU Training:**
- Download from Lambda storage: ~10-15 min (~$16)
- Training: 8-12 hours (~$512-768)
- **Total GPU cost: $528-784**

**Total Cost (all phases): $528-784**

**Savings vs Direct Download on GPU:** $256

**Bonus:** Dataset cached in Lambda storage for future training runs!

---

## Troubleshooting

### "RuntimeError: CUDA out of memory"
- Shouldn't happen with 8x A100 (40GB each)
- If it does: halve batch_size (4→2) and double gradient_accumulation (8→16)
- Check no other processes using GPU memory

### "Lambda storage download fails"
- Verify .env file copied to GPU instance
- Check credentials: `aws s3 ls s3://ozlabs-qwen3-asr --endpoint-url "$S3_ENDPOINT_URL"`
- If bucket doesn't exist: Upload from local machine first

### "Dataset download hanging"
- From Lambda storage: Shouldn't happen (internal network is fast)
- If it does: Check Lambda network status, try again

### "WER stuck at 12% (same as Round 1)"
- Check augmentation is applying: Look for "Applying audio augmentation" in logs
- Check balanced sampling: Verify dataset interleaving probabilities in W&B config
- Check W&B config: Should show `use_lora=False`, `use_audio_augmentation=True`

### "Loss stuck at 11.3"
- This happened in Round 1
- Should break through by epoch 2-3 with new methods
- If still stuck at epoch 4: Stop training, debug implementation

### "Model averaging fails"
- Need at least 3 checkpoints saved
- Check `save_steps=100` and training completes >300 steps
- Fallback: Use best checkpoint without averaging

---

## Quick Reference Commands

### Data Prep (Local Mac)
```bash
uv run python prepare_qwen_data.py
./scripts/upload_to_lambda_storage.sh
```

### Training (GPU Instance)
```bash
git clone https://github.com/OzLabs/caspi.git && cd caspi
# Copy .env file
./scripts/download_from_lambda_storage.sh
wandb login
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2.5-full-finetuning"
uv run python train_hebrew_asr_enhanced.py
```

### Evaluation (Local Mac)
```bash
# Full benchmark
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model ./qwen3-asr-hebrew-round2.5-averaged/ \
    --output results_round2.5.json

# Quick check
uv run python scripts/quick_eval.py \
    --model ./qwen3-asr-hebrew-round2.5-averaged/ \
    --test-set eval-d1 \
    --max-samples 100
```

---

## Next Steps After Training

1. **Evaluate on all 6 test sets** (see EVALUATION_GUIDE.md)
2. **Upload model to HuggingFace Hub:**
   ```bash
   huggingface-cli upload OzLabs/Qwen3-ASR-Hebrew-Round2.5 ./qwen3-asr-hebrew-round2.5-averaged/
   ```
3. **Compare results to Round 1 and SOTA** (see BEATING_SOTA_ANALYSIS.md)
4. **Decide on Round 3** based on gap to SOTA
5. **Update README** with Round 2.5 results

---

## Files Modified/Created for Round 2.5

**Training:**
- `train_hebrew_asr_enhanced.py` - Main training script with all SOTA methods
- `config.yaml` - Updated with new hyperparameters

**Scripts:**
- `scripts/upload_to_lambda_storage.sh` - Upload dataset to Lambda storage
- `scripts/download_from_lambda_storage.sh` - Download dataset on GPU instance
- `scripts/evaluate_ivrit_benchmarks.py` - Benchmark on all 6 test sets
- `scripts/quick_eval.py` - Fast validation on single test set

**Documentation:**
- `ROUND2.5_LAUNCH_GUIDE.md` - This file
- `ROUND2.5_IMPLEMENTATION.md` - Technical details of SOTA methods
- `EVALUATION_GUIDE.md` - How to benchmark models
- `BEATING_SOTA_ANALYSIS.md` - Gap analysis and roadmap
- `PRETRAINING_CHECKLIST.md` - Pre-flight checklist

---

**Ready to launch?** Follow Phase 1 → Phase 2 → Phase 3 above. Good luck beating SOTA! 🚀
