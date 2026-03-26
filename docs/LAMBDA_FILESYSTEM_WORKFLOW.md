# Lambda Filesystem Workflow (No Compression Needed!)

**Key Insight:** Lambda Cloud Storage works as a **persistent filesystem** that mounts at `/lambda/nfs/persistent-storage`. This means **no compression, no upload/download** - just direct file access across instances!

**Savings:**
- ❌ No compression time (saves ~1-2 hours)
- ❌ No upload time (saves ~10-30 minutes)
- ❌ No download time on GPU (saves ~10-15 minutes)
- ✅ **Total time saved: ~1.5-2 hours**
- ✅ **Simpler workflow: Just attach → use**

---

## How Lambda Filesystems Work

**Mount Location:** `/lambda/nfs/persistent-storage`
**Capacity:** 8 EB (essentially unlimited)
**Billing:** Per GB used per month (incremental, hourly)
**Access:** Like any regular directory - read/write files directly

**Source:** [Lambda Docs - Filesystems](https://docs.lambda.ai/public-cloud/filesystems/)

---

## Setup (One-Time)

### 1. Create Lambda Filesystem

Via Lambda Cloud Dashboard:
```
1. Go to https://cloud.lambdalabs.com/filesystems
2. Click "Create Filesystem"
3. Name: qwen3-asr-data
4. Region: us-east-2 (or your preferred region)
5. Click "Create"
```

Via Lambda CLI (if available):
```bash
lambda filesystem create qwen3-asr-data --region us-east-2
```

**Cost:** ~$0.05/GB/month (for 363 GB ≈ $18/month)

---

## Workflow

### Phase 1: Dataset Preparation (CPU Instance with Filesystem)

**1. Launch CPU Instance with Filesystem Attached**

```
Dashboard: https://cloud.lambdalabs.com/instances
1. Click "Launch Instance"
2. Select: CPU instance (16 vCPU, $0.08/hr recommended)
3. Attach filesystem: Select "qwen3-asr-data"
4. Region: us-east-2 (must match filesystem region)
5. Launch
```

**2. SSH into CPU Instance**

```bash
ssh ubuntu@<cpu-instance-ip>

# Verify filesystem is mounted
ls -lh /lambda/nfs/persistent-storage/
# Should show empty qwen3-asr-data/ directory
```

**3. Clone Repo and Install Dependencies**

```bash
cd /lambda/nfs/persistent-storage/qwen3-asr-data/

git clone https://github.com/OzLabs/caspi.git
cd caspi
uv sync
```

**4. Prepare Dataset (Writes Directly to Filesystem)**

```bash
# Modify output to write to filesystem root (not ./qwen3_asr_data)
uv run python prepare_qwen_data.py

# This creates:
# /lambda/nfs/persistent-storage/qwen3-asr-data/qwen3_asr_data/
#   ├── train.jsonl
#   ├── eval.jsonl
#   └── audio/
#       ├── train/
#       │   ├── ivrit-ai_knesset-plenums-whisper-training/
#       │   ├── ivrit-ai_crowd-transcribe-v5/
#       │   └── ivrit-ai_crowd-recital-whisper-training/
#       └── eval/ (...)

# Time: ~1-2 hours with 16 workers (multiprocessing)
# Size: ~363 GB on filesystem
```

**5. Verify Dataset**

```bash
cd /lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data/

# Check file counts
wc -l train.jsonl eval.jsonl
# Should show ~500K+ train samples, ~25K+ eval samples

# Check audio files
find audio/ -name "*.wav" | wc -l
# Should show ~525K+ WAV files

# Check total size
du -sh .
# Should show ~363 GB
```

**6. Terminate CPU Instance**

```bash
# From your local machine
lambda instance terminate <cpu-instance-id>

# The filesystem persists! Data is safe.
# Cost: ~$0.16 (2 hours × $0.08/hr)
```

---

### Phase 2: Training (GPU Instance with Same Filesystem)

**1. Launch GPU Instance with Filesystem Attached**

```
Dashboard: https://cloud.lambdalabs.com/instances
1. Click "Launch Instance"
2. Select: 8x A100 (40GB) - $64/hr
3. Attach filesystem: Select "qwen3-asr-data" (same filesystem!)
4. Region: us-east-2 (must match filesystem region)
5. Launch
```

**2. SSH into GPU Instance**

```bash
ssh ubuntu@<gpu-instance-ip>

# Verify filesystem is mounted
ls -lh /lambda/nfs/persistent-storage/qwen3-asr-data/
# Should show caspi/ directory with all your data!
```

**3. Navigate to Dataset**

```bash
cd /lambda/nfs/persistent-storage/qwen3-asr-data/caspi/

# Data is INSTANTLY available (no download!)
ls -lh qwen3_asr_data/
# train.jsonl  eval.jsonl  audio/

# Verify dataset integrity
wc -l qwen3_asr_data/train.jsonl
# Should show ~500K+ lines
```

**4. Setup W&B Tracking**

```bash
wandb login
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2.5-full-finetuning"
export WANDB_NOTES="Round 2.5: Full fine-tuning + Knesset + 5 SOTA methods"
```

**5. Launch Training**

```bash
# Training script will read from qwen3_asr_data/ (already on filesystem)
uv run python train_hebrew_asr_enhanced.py

# Time: 8-12 hours
# Output: ./qwen3-asr-hebrew-round2.5/ (also on filesystem)
```

**6. Monitor Training**

```bash
# W&B dashboard: https://wandb.ai/your-username/qwen3-asr-hebrew
# Watch for:
# - Loss breaking through 11.3 plateau
# - WER dropping below 10% by epoch 3-4
# - No GPU memory errors
```

**7. After Training Completes**

```bash
# Model saved to filesystem
ls -lh qwen3-asr-hebrew-round2.5-averaged/

# Option A: Copy to local machine
rsync -avz --progress ubuntu@<gpu-ip>:/lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3-asr-hebrew-round2.5-averaged/ ./

# Option B: Upload to HuggingFace Hub
huggingface-cli upload OzLabs/Qwen3-ASR-Hebrew-Round2.5 ./qwen3-asr-hebrew-round2.5-averaged/

# Option C: Keep on filesystem for future use
# (Filesystem persists even after instance termination)
```

**8. Terminate GPU Instance**

```bash
# Keep filesystem running if you want to reuse data
# Or delete filesystem to stop billing

lambda instance terminate <gpu-instance-id>
```

---

## Cost Breakdown (With Filesystem)

### Data Preparation
- CPU instance (16 vCPU, 2 hours): **$0.16**
- Filesystem storage (363 GB, 1 month): **~$18/month**

### Training
- GPU instance (8x A100, 10 hours): **$640**
- Filesystem access: **FREE** (no transfer costs)

### Total First Training Run
- **$640.16** (GPU) + **~$18** (storage/month)

### Future Training Runs (Reuse Dataset)
- **$640** (GPU only, data already on filesystem!)
- **Savings per run:** $0.16 + time saved (no prep needed)

---

## Advantages Over S3 Workflow

**Old S3 Workflow:**
1. Prep data locally (4-8 hours)
2. Compress (1-2 hours)
3. Upload to S3 (10-30 minutes)
4. Download on GPU (10-15 minutes)
5. Extract (5-15 minutes)
6. **Total prep overhead:** ~6-10 hours

**New Filesystem Workflow:**
1. Prep data on CPU with filesystem (1-2 hours)
2. Attach filesystem to GPU (instant)
3. **Total prep overhead:** ~1-2 hours

**Time saved:** 4-8 hours
**Complexity:** Much simpler (no compression/upload/download steps)
**Reusability:** Dataset stays on filesystem forever (reuse for Round 3, Round 4, etc.)

---

## Updated Training Script Path

Since the dataset is now on the filesystem, update `train_hebrew_asr_enhanced.py` to point to the correct path:

```python
# In config or training script:
TRAIN_JSONL = "/lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data/train.jsonl"
EVAL_JSONL = "/lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data/eval.jsonl"

# Or use relative path if running from caspi/ directory:
TRAIN_JSONL = "./qwen3_asr_data/train.jsonl"
EVAL_JSONL = "./qwen3_asr_data/eval.jsonl"
```

No other changes needed!

---

## Troubleshooting

### "Filesystem not mounted"
```bash
# Check mount point
ls /lambda/nfs/persistent-storage/

# If empty, filesystem wasn't attached during instance creation
# Solution: Terminate instance, create new one with filesystem attached
```

### "Permission denied"
```bash
# Filesystem is writable by ubuntu user
whoami  # Should show: ubuntu

# If permission issues:
sudo chown -R ubuntu:ubuntu /lambda/nfs/persistent-storage/qwen3-asr-data/
```

### "No space left on device"
```bash
# Check filesystem usage
df -h /lambda/nfs/persistent-storage/

# Lambda filesystems have 8 EB capacity (won't run out)
# If you see space issues, it's likely local disk, not the filesystem
```

### "Slow file access"
```bash
# Lambda filesystems are network-attached (like NFS)
# Expected throughput: ~1-10 GB/s
# For audio files (16kHz WAV), this is plenty fast
# If slow, check network issues or Lambda status page
```

---

## Quick Reference

### Prep on CPU with Filesystem
```bash
# Launch CPU instance with qwen3-asr-data filesystem attached
ssh ubuntu@<cpu-ip>
cd /lambda/nfs/persistent-storage/qwen3-asr-data/
git clone https://github.com/OzLabs/caspi.git && cd caspi
uv sync
uv run python prepare_qwen_data.py  # ~1-2 hours
# Terminate CPU instance (filesystem persists)
```

### Train on GPU with Same Filesystem
```bash
# Launch GPU instance with qwen3-asr-data filesystem attached
ssh ubuntu@<gpu-ip>
cd /lambda/nfs/persistent-storage/qwen3-asr-data/caspi/
wandb login && export WANDB_PROJECT="qwen3-asr-hebrew" && export WANDB_RUN_NAME="round2.5"
uv run python train_hebrew_asr_enhanced.py  # 8-12 hours
# Copy model or upload to HF Hub
```

---

## Next Steps

1. **Create Lambda filesystem** (one-time): `qwen3-asr-data` in us-east-2
2. **Run dataset prep** (Phase 1 above): 1-2 hours on CPU instance
3. **Launch training** (Phase 2 above): 8-12 hours on GPU instance
4. **Evaluate results**: See EVALUATION_GUIDE.md

**Ready to start!** 🚀

---

**Sources:**
- [Lambda Docs - Filesystems](https://docs.lambda.ai/public-cloud/filesystems/)
- [Lambda Blog - Persistent Storage Beta](https://lambda.ai/blog/persistent-storage-beta)
