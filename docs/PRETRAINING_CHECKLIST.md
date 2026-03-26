# Pre-Training Checklist for Round 2.5

**Goal:** Maximize GPU efficiency, minimize wasted time

---

## ✅ **Things to Do BEFORE Training**

### **1. Pre-build Dataset** 🔴 CRITICAL (Saves 2-6 hours)

**Why:** Knesset dataset is ~150-300GB, downloading during training wastes GPU time

**Steps:**
```bash
# On machine with good internet (can be different from training machine)
cd ~/caspi
uv run python scripts/prebuild_round25_dataset.py

# This will:
# - Download all 3 datasets (~5,050 hours)
# - Apply balanced interleaving (50-30-20)
# - Deduplicate by audio hash
# - Save as JSONL + WAV files (~150-200GB)
# - Create train/eval splits (95/5)
```

**Transfer to training machine:**
```bash
# Check size first
du -sh qwen3_asr_round25_data/

# Rsync to training machine
rsync -avz --progress qwen3_asr_round25_data/ ubuntu@<training-ip>:~/caspi/qwen3_asr_round25_data/
```

**Update training script** to load from local JSONL instead of HuggingFace:
```python
# In train_hebrew_asr_enhanced.py - load_datasets()
# Replace:
ds = load_dataset(dataset_name, split="train")

# With:
from datasets import Dataset
ds = Dataset.from_json("./qwen3_asr_round25_data/train.jsonl")
```

---

### **2. Optional: Pre-compute Augmented Samples** 🟡 MEDIUM (Saves compute during training)

**Why:** Audio augmentation (speed/pitch/noise) is CPU-intensive during training

**Trade-off:**
- **Pre-compute:** Faster training, but larger dataset (~3-5x size)
- **On-the-fly:** Slower training, but saves disk space

**Recommendation:** Keep on-the-fly for now (simpler), optimize later if bottleneck

---

### **3. Verify Dependencies** ✅ QUICK (5 minutes)

```bash
# Check torchaudio (for audio augmentation)
python3 -c "import torchaudio; print(torchaudio.__version__)"

# Check sox (for speed perturbation)
python3 -c "import torchaudio; torchaudio.sox_effects.apply_effects_tensor"

# If sox fails, install:
# Ubuntu: sudo apt-get install sox libsox-fmt-all
# Mac: brew install sox
```

---

### **4. Set Environment Variables** ✅ QUICK (1 minute)

```bash
# HuggingFace Hub (for auto-upload)
export HF_TOKEN="<your-token>"  # Get from https://hf.co/settings/tokens
export HF_REPO_ID="OzLabs/Qwen3-ASR-Hebrew-Round2.5"

# Weights & Biases (for experiment tracking)
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2.5-balanced-$(date +%Y%m%d-%H%M)"

# Optional: Skip HF upload if testing
# export SKIP_HF_UPLOAD="true"
```

---

### **5. Verify GPU Setup** ✅ QUICK (2 minutes)

```bash
# Check GPUs available
nvidia-smi

# Check PyTorch sees GPUs
python3 -c "import torch; print(f'{torch.cuda.device_count()} GPUs: {torch.cuda.get_device_name(0)}')"

# Expected: 8 GPUs on A100 machine
```

---

### **6. Estimate Training Time & Cost** 💰 IMPORTANT

**With pre-built dataset (recommended):**
- **8x A100:** 8-12 hours (~$64-96 on Lambda Labs)
- **2x A100:** 30-40 hours (~$120-160 on Lambda Labs)

**Without pre-built dataset (NOT recommended):**
- Add 2-6 hours download time = wasted GPU cost

**Recommendation:** Use 8x A100 with pre-built dataset

---

## 🚀 **Training Launch Checklist**

### **Option A: Quick Validation (Without Knesset)** ⚡ 2-3 hours

**Purpose:** Verify all methods work before full training

**Steps:**
```bash
# 1. Temporarily disable Knesset in config
# Edit train_hebrew_asr_enhanced.py, comment out:
#   "ivrit-ai/knesset-plenums-whisper-training"

# 2. Launch training
export WANDB_RUN_NAME="round2.5-validation-$(date +%Y%m%d)"
uv run torchrun --nproc_per_node=8 train_round2_gradual.py

# 3. Expected WER: 7.5-9.0% (still better than 12.3% baseline)
```

**What to verify:**
- ✅ All 5 ivrit.ai methods apply (check logs)
- ✅ Synthetic augmentation works (speed/pitch/noise)
- ✅ No errors during training
- ✅ WER < 9.0% on eval set
- ✅ Model averaging produces `-averaged` variant

**If validation passes → Proceed to Option B**

---

### **Option B: Full Training (With Knesset)** 🎯 8-12 hours

**Prerequisites:**
- ✅ Dataset pre-built (150-200GB)
- ✅ Transferred to training machine
- ✅ Dependencies verified
- ✅ Environment variables set

**Steps:**
```bash
# 1. Verify dataset exists
ls -lh qwen3_asr_round25_data/
# Should see: train.jsonl, eval.jsonl, wavs/, dataset_info.json

# 2. Set environment
export HF_TOKEN="<your-token>"
export HF_REPO_ID="OzLabs/Qwen3-ASR-Hebrew-Round2.5"
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2.5-full-$(date +%Y%m%d-%H%M)"

# 3. Launch training
uv run torchrun --nproc_per_node=8 train_round2_gradual.py

# 4. Monitor progress
# - W&B dashboard: https://wandb.ai/OzLabs/qwen3-asr-hebrew
# - Check logs for augmentation, balanced sampling, model averaging
```

**Expected results:**
- **WER:** 6.0-7.5% (competitive with SOTA 5.1%)
- **Training time:** 8-12 hours on 8x A100
- **Cost:** ~$64-96 on Lambda Labs ($8/hr per A100)

---

## 📊 **Monitoring During Training**

### **Key Metrics to Watch:**

1. **Dataset Loading (First 5 minutes):**
   ```
   Loading datasets with balanced sampling...
   knesset-plenums-whisper-training: 50.0% (XXX,XXX samples)
   crowd-transcribe-v5:              30.0% (XXX,XXX samples)
   crowd-recital-whisper-training:   20.0% (XXX,XXX samples)
   ```

2. **Loss Convergence:**
   - Should start ~450-500 (random init)
   - Drop to ~220-230 by epoch 2 (Strategy B)
   - Drop to ~200-210 by epoch 5 (Strategy A)
   - **Should NOT plateau at 11.3** (fixed in Round 2)

3. **WER Metrics:**
   - Should work (no 100.0% errors)
   - Eval every 500 steps
   - Target: < 9.0% by end of training

4. **Augmentation Logs:**
   ```
   Applying synthetic augmentation: speed=0.9
   Applying synthetic augmentation: pitch=+2
   Applying synthetic augmentation: noise SNR=22.5dB
   ```

5. **Model Averaging (After training):**
   ```
   Averaging 3 best checkpoints:
     checkpoint-1000: eval_loss=0.152
     checkpoint-2000: eval_loss=0.149
     checkpoint-3000: eval_loss=0.151
   ✓ Saved averaged model to: ./qwen3-asr-hebrew-averaged
   ```

---

## ⚠️ **Common Issues & Solutions**

### **Issue 1: Dataset download hangs**
**Solution:** Use pre-built dataset (see Section 1)

### **Issue 2: Audio augmentation fails**
```
Error: torchaudio.sox_effects not available
```
**Solution:**
```bash
# Ubuntu
sudo apt-get install sox libsox-fmt-all

# Mac
brew install sox

# Then verify
python3 -c "import torchaudio; torchaudio.sox_effects.apply_effects_tensor"
```

### **Issue 3: OOM (Out of Memory)**
**Solution:**
```python
# Reduce batch size in config
batch_size: int = 2  → 1
gradient_accumulation_steps: int = 4  → 8
```

### **Issue 4: WER still showing 100.0%**
**Cause:** BF16 dtype mismatch (should be fixed in Round 2.5)
**Solution:** Check logs for "Input type (float) and bias type (BFloat16)" error

### **Issue 5: Model averaging fails**
```
Warning: Only 2 checkpoints found, need 3+ for averaging
```
**Cause:** Training ended early or eval_steps too large
**Solution:** Check training completed, or reduce eval_steps from 500 → 250

---

## 💡 **Additional Optimizations (Optional)**

### **1. Gradient Checkpointing Tuning**
- **Current:** Enabled globally
- **Optimization:** Selective checkpointing (skip last N layers)
- **Impact:** 10-15% faster training, same memory

### **2. Mixed Precision Tuning**
- **Current:** BF16 everywhere
- **Optimization:** FP32 for critical ops (loss, metrics)
- **Impact:** More stable training

### **3. Data Loading Optimization**
- **Current:** num_workers=auto
- **Optimization:** Set num_workers=8 (match GPU count)
- **Impact:** Faster data loading

### **4. Checkpoint Saving Strategy**
- **Current:** Save every 1000 steps
- **Optimization:** Save top 5 by eval loss, delete rest
- **Impact:** Save disk space (~10GB per checkpoint)

---

## 📈 **Success Criteria**

### **Round 2.5 is successful if:**

1. ✅ **Training completes** without errors
2. ✅ **All 5 ivrit.ai methods** apply (logs show timestamps, context, augmentation, averaging, balanced sampling)
3. ✅ **Loss converges** below 210 (no plateau at 11.3)
4. ✅ **WER < 9.0%** on eval set (without Knesset)
5. ✅ **WER < 7.5%** on eval set (with Knesset)
6. ✅ **Model averaging** produces `-averaged` variant
7. ✅ **Auto-upload** to HuggingFace Hub succeeds

### **Beat SOTA if:**
- WER < 5.1% on eval-d1 (ivrit.ai SOTA)
- Competitive across all 6 test sets

---

## 🎯 **Recommended Workflow**

**Day 1: Preparation**
1. ✅ Pre-build dataset (2-6 hours on good internet)
2. ✅ Transfer to training machine (1-2 hours)
3. ✅ Verify dependencies (10 minutes)
4. ✅ Run Option A validation (2-3 hours training)

**Day 2: Full Training**
1. ✅ Verify validation passed (WER < 9.0%)
2. ✅ Launch Option B full training (8-12 hours)
3. ✅ Monitor W&B dashboard
4. ✅ Benchmark on all 6 test sets

**Day 3: Analysis**
1. ✅ Compare Round 2.5 vs Round 1 (12.3% baseline)
2. ✅ Benchmark against ivrit.ai leaderboard
3. ✅ Identify remaining gaps to SOTA
4. ✅ Plan Round 3 if needed

---

## 📝 **Final Pre-Flight Check**

Before clicking "Launch":

- [ ] Dataset pre-built and transferred (or streaming from HF is acceptable)
- [ ] HF_TOKEN set (for auto-upload)
- [ ] WANDB_PROJECT and WANDB_RUN_NAME set
- [ ] GPU count verified (8x A100 recommended)
- [ ] Dependencies verified (torchaudio, sox)
- [ ] Config reviewed (balanced sampling 50-30-20)
- [ ] Backup plan if training fails (checkpoints every 1000 steps)
- [ ] Monitoring ready (W&B dashboard, logs)

**Estimated total time:** 10-18 hours (prep + training)
**Estimated total cost:** $80-160 (Lambda Labs 8x A100)

---

**You're ready to train!** 🚀

Next step: Run `uv run python scripts/prebuild_round25_dataset.py` on a machine with good internet.
