# GCP VM → Lambda GPU Workflow

**Your Setup:**
- Prep data on **GCP VM with 2 GPUs** (actually uses CPUs for prep, GPUs unused)
- Train on **Lambda 8x A100** with Lambda filesystem

**Key Decision:** How to transfer 363 GB from GCP to Lambda?

---

## Transfer Options

### Option 1: Direct rsync to Lambda Filesystem (RECOMMENDED)

**Workflow:**
1. Create Lambda filesystem
2. Launch small Lambda CPU instance with filesystem attached
3. Prep data on GCP VM
4. rsync from GCP VM → Lambda CPU instance (writes to filesystem)
5. Terminate Lambda CPU instance
6. Launch Lambda GPU instance with filesystem attached → Train

**Pros:**
- Direct transfer, no intermediate storage
- Simple workflow
- Data ends up on Lambda filesystem (ready for training)

**Cons:**
- Need to keep Lambda CPU instance running during transfer (~2-3 hours)
- Cost: ~$0.24 (3 hours × $0.08/hr)

**Speed:** ~2-3 hours for 363 GB (depends on GCP → Lambda network speed)

---

### Option 2: GCP Cloud Storage → Lambda Download

**Workflow:**
1. Prep data on GCP VM
2. Upload to GCS bucket (~1-2 hours, fast internal network)
3. Download from GCS to Lambda GPU instance (~2-3 hours)
4. Train

**Pros:**
- No Lambda CPU instance needed
- GCS is cheap storage (~$0.02/GB/month)
- Can reuse GCS bucket for future runs

**Cons:**
- Two-step transfer (GCP → GCS → Lambda)
- Data not on Lambda filesystem (uses GPU instance local disk)
- Need to delete/manage GCS bucket

**Speed:** ~3-5 hours total (upload + download)
**Cost:** ~$7/month for GCS storage

---

### Option 3: Compress + Transfer (Hybrid)

**Workflow:**
1. Prep data on GCP VM
2. Compress (~1-2 hours with pigz)
3. rsync compressed file to Lambda (~1-2 hours for 100-150 GB compressed)
4. Extract on Lambda GPU instance (~30-60 minutes)
5. Train

**Pros:**
- Faster transfer (100-150 GB vs 363 GB)
- Less bandwidth usage

**Cons:**
- Compression time (1-2 hours)
- Extraction time on GPU (wastes GPU money)
- More complex workflow

**Speed:** ~3-5 hours total

---

## Recommendation: Option 1 (Direct rsync)

**Why:**
- Simplest workflow
- Data ends up on Lambda filesystem (reusable)
- No intermediate storage needed
- Only costs ~$0.24 for Lambda CPU instance

---

## Detailed Workflow (Option 1)

### Phase 1: Create Lambda Filesystem (One-Time)

```bash
# Via Lambda Dashboard: https://cloud.lambdalabs.com/filesystems
# 1. Click "Create Filesystem"
# 2. Name: qwen3-asr-data
# 3. Region: us-east-2 (or your preferred region)
# 4. Create
```

---

### Phase 2: Launch Lambda CPU Instance with Filesystem

```bash
# Via Lambda Dashboard: https://cloud.lambdalabs.com/instances
# 1. Click "Launch Instance"
# 2. Select: CPU instance (8 vCPU, $0.04/hr)
# 3. Attach filesystem: qwen3-asr-data
# 4. Region: us-east-2 (must match filesystem)
# 5. Launch

# Note the instance IP for rsync
LAMBDA_CPU_IP="<instance-ip>"
```

---

### Phase 3: Prep Data on GCP VM

**On your GCP VM:**

```bash
# Check CPU cores (multiprocessing will use all)
nproc
# Should show 16-32 cores on typical 2-GPU VM

# Clone repo if not already done
git clone https://github.com/OzLabs/caspi.git
cd caspi
uv sync

# Run data prep with multiprocessing
uv run python prepare_qwen_data.py

# This creates:
# ./qwen3_asr_data/
#   ├── train.jsonl (~500K samples)
#   ├── eval.jsonl (~25K samples)
#   └── audio/ (~525K WAV files, 363 GB)

# Time: 1-2 hours with multiprocessing
# The 2 GPUs are NOT used (audio processing is CPU-bound)
```

---

### Phase 4: Transfer to Lambda Filesystem via rsync

**On your GCP VM:**

```bash
# Set Lambda CPU instance IP
LAMBDA_CPU_IP="<lambda-cpu-instance-ip>"

# Test SSH connection
ssh ubuntu@$LAMBDA_CPU_IP "ls /lambda/nfs/persistent-storage/"
# Should show qwen3-asr-data/ directory

# Create target directory
ssh ubuntu@$LAMBDA_CPU_IP "mkdir -p /lambda/nfs/persistent-storage/qwen3-asr-data/caspi/"

# rsync with progress (this will take 2-3 hours for 363 GB)
rsync -avz --progress \
    ./qwen3_asr_data/ \
    ubuntu@$LAMBDA_CPU_IP:/lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data/

# Options explained:
# -a: archive mode (preserves permissions, timestamps)
# -v: verbose
# -z: compress during transfer (saves bandwidth)
# --progress: show progress

# Alternative without compression (if network is fast):
rsync -av --progress \
    ./qwen3_asr_data/ \
    ubuntu@$LAMBDA_CPU_IP:/lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data/

# Expected transfer speed: 50-200 MB/s
# Time: 2-3 hours for 363 GB
```

**Monitor transfer:**

```bash
# On Lambda CPU instance (in another terminal):
ssh ubuntu@$LAMBDA_CPU_IP

# Watch disk usage
watch -n 10 'du -sh /lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data/'

# Count files transferred
watch -n 10 'find /lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data/ -name "*.wav" | wc -l'
# Should eventually show ~525K files
```

---

### Phase 5: Verify Transfer

**On Lambda CPU instance:**

```bash
ssh ubuntu@$LAMBDA_CPU_IP

cd /lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data/

# Check file counts
wc -l train.jsonl eval.jsonl
# Should show ~500K train, ~25K eval

# Check audio files
find audio/ -name "*.wav" | wc -l
# Should show ~525K files

# Check total size
du -sh .
# Should show ~363 GB

# Verify a few random audio files
ls audio/train/ivrit-ai_knesset-plenums-whisper-training/ | head -5
```

---

### Phase 6: Terminate Lambda CPU Instance

```bash
# From your local machine or GCP VM
# The filesystem persists!

# Via Lambda dashboard: Terminate the CPU instance
# Or via CLI:
lambda instance terminate <cpu-instance-id>

# Cost: ~$0.16-0.24 (3-4 hours × $0.04-0.08/hr)
```

---

### Phase 7: Launch Lambda GPU Instance with Filesystem

```bash
# Via Lambda Dashboard: https://cloud.lambdalabs.com/instances
# 1. Click "Launch Instance"
# 2. Select: 8x A100 (40GB) - $64/hr
# 3. Attach filesystem: qwen3-asr-data (SAME filesystem!)
# 4. Region: us-east-2 (must match)
# 5. Launch

# Note GPU instance IP
LAMBDA_GPU_IP="<gpu-instance-ip>"
```

---

### Phase 8: Setup and Train on Lambda GPU

**SSH into Lambda GPU instance:**

```bash
ssh ubuntu@$LAMBDA_GPU_IP

# Verify filesystem is mounted
ls -lh /lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data/
# Should show train.jsonl, eval.jsonl, audio/

# Clone training repo
cd /lambda/nfs/persistent-storage/qwen3-asr-data/
git clone https://github.com/OzLabs/caspi.git caspi-training
cd caspi-training

# Install deps
uv sync

# Setup W&B
wandb login
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2.5-full-finetuning"

# Update training script to point to filesystem data
# (Or use symlink if script expects ./qwen3_asr_data)
ln -s /lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data ./qwen3_asr_data

# Launch training
uv run python train_hebrew_asr_enhanced.py

# Time: 8-12 hours
# Output: ./qwen3-asr-hebrew-round2.5-averaged/
```

---

### Phase 9: After Training

**Copy model back to GCP or upload to HF:**

```bash
# Option A: Copy to GCP VM
# (On GCP VM)
rsync -avz --progress \
    ubuntu@$LAMBDA_GPU_IP:/lambda/nfs/persistent-storage/qwen3-asr-data/caspi-training/qwen3-asr-hebrew-round2.5-averaged/ \
    ./qwen3-asr-hebrew-round2.5-averaged/

# Option B: Upload to HuggingFace Hub
# (On Lambda GPU instance)
huggingface-cli upload OzLabs/Qwen3-ASR-Hebrew-Round2.5 ./qwen3-asr-hebrew-round2.5-averaged/

# Option C: Keep on Lambda filesystem
# (For future experiments, no transfer needed)
```

---

## Cost Breakdown (GCP → Lambda Workflow)

### Data Preparation
- **GCP VM:** Already running (no additional cost)
- **Lambda CPU instance (rsync target):** $0.16-0.24 (3-4 hours)
- **Lambda filesystem storage:** ~$18/month (363 GB)

### Training
- **Lambda GPU (8x A100, 10 hours):** $640
- **Filesystem access:** FREE

### Total First Run
- **$640.16-0.24** (GPU + CPU transfer instance)
- **~$18/month** (filesystem storage)

### Future Runs (Reuse Dataset)
- **$640** (GPU only, data already on filesystem)

---

## Alternative: Skip Lambda Filesystem (Use GPU Local Disk)

If you don't want to use Lambda filesystem, you can rsync directly to GPU instance local disk:

```bash
# Launch Lambda GPU instance WITHOUT filesystem

# rsync directly from GCP to GPU
rsync -avz --progress \
    ./qwen3_asr_data/ \
    ubuntu@$LAMBDA_GPU_IP:/workspace/qwen3_asr_data/

# Train as normal
# Model saved to /workspace/

# Note: Data deleted when GPU instance terminates
# Need to re-transfer for future training runs
```

**Pros:** No filesystem costs (~$18/month)
**Cons:** Need to re-transfer for each training run (2-3 hours + $0 transfer cost but wastes time)

---

## Quick Reference

### On GCP VM
```bash
git clone https://github.com/OzLabs/caspi.git && cd caspi
uv sync
uv run python prepare_qwen_data.py  # 1-2 hours

# Transfer to Lambda
rsync -avz --progress ./qwen3_asr_data/ ubuntu@<lambda-ip>:/lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data/
```

### On Lambda GPU (After Transfer)
```bash
cd /lambda/nfs/persistent-storage/qwen3-asr-data/
git clone https://github.com/OzLabs/caspi.git caspi-training && cd caspi-training
ln -s /lambda/nfs/persistent-storage/qwen3-asr-data/caspi/qwen3_asr_data ./qwen3_asr_data
wandb login && export WANDB_PROJECT="qwen3-asr-hebrew" && export WANDB_RUN_NAME="round2.5"
uv run python train_hebrew_asr_enhanced.py  # 8-12 hours
```

---

## Next Steps

1. **Prep data on GCP VM** (1-2 hours): `uv run python prepare_qwen_data.py`
2. **Create Lambda filesystem** (one-time): Via dashboard
3. **Launch Lambda CPU** with filesystem attached
4. **rsync GCP → Lambda** (2-3 hours)
5. **Launch Lambda GPU** with filesystem attached
6. **Train!** (8-12 hours)

Ready to start the prep on your GCP VM? 🚀
