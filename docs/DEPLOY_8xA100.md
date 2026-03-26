# Deploy Round 2 Training on 8x A100

## Machine Info
- **GPUs:** 8x A100 (80GB or 40GB)
- **Expected training time:** ~2-2.5 hours
- **Expected cost:** ~$25-30

## What This Training Does
Fine-tunes Qwen3-ASR-1.7B for Hebrew ASR using gradual unfreezing:
- **Epochs 1-2 (Strategy B):** Train projector + top 12 LLM layers + LM head
- **Epochs 3-5 (Strategy A):** Also unfreeze top 8 audio encoder layers
- **Data augmentation:** SpecAugment (freq+time masking) + speed perturbation (0.9x/1.0x/1.1x)
- **WER/CER eval:** 100 samples evaluated every 500 steps
- **Target:** 12.3% WER → 10.5-11.0% WER

## Setup Steps

```bash
# 1. Clone and install
git clone <repo-url> && cd Caspi
uv sync

# 2. Login
huggingface-cli login
wandb login

# 3. Verify GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Should output: GPUs: 8
```

## Config Changes Needed for 8 GPUs

Edit `train_hebrew_asr_enhanced.py` — the `TrainingConfig` class (around line 100):

```python
# CURRENT (2x A100):
batch_size: int = 4
gradient_accumulation_steps: int = 8  # effective = 4 × 8 × 2 = 64

# CHANGE TO (8x A100):
batch_size: int = 4
gradient_accumulation_steps: int = 2  # effective = 4 × 2 × 8 = 64
```

This keeps the same effective batch size (64) while leveraging 8 GPUs.

Also update the print statement in `main()` around line 931:
```python
# Change "2 GPUs" references to use actual GPU count
```

And in `train_round2_gradual.py`, update the configure_round2() prints to reflect 8 GPUs.

## Run Training

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_ENTITY="OzLabs"
export WANDB_PROJECT="qwen3-asr-hebrew"

torchrun --nproc_per_node=8 train_round2_gradual.py
```

## Data

Training data is generated automatically on first run:
- Downloads `ivrit-ai/crowd-transcribe-v5` and `ivrit-ai/crowd-recital-whisper-training`
- Exports ~196K WAV files (~27GB) to `qwen3_asr_round2_data/`
- Creates `train.jsonl` (186K samples) and `eval.jsonl` (9.8K samples)
- **First run takes ~2-3 hours for data prep** (CPU-bound WAV export)

If you want to skip data prep, copy `qwen3_asr_round2_data/` from the current machine.

## Expected Metrics During Training

| Checkpoint | Loss (per-example) | Per-token est. | Notes |
|-----------|-------------------|----------------|-------|
| Step 100 | ~100-120 | ~5-6 | Early, near random |
| Step 500 | ~50-55 | ~2.5-2.8 | Settling |
| Step 2000 | ~30-40 | ~1.5-2.0 | Good progress |
| Step 3600 (epoch 3) | spike then recover | — | Strategy A kicks in |
| Final | ~15-25 | ~0.7-1.2 | Target |

## Expected Timeline (8x A100)

| Phase | Duration |
|-------|----------|
| Data prep (if from scratch) | 2-3 hours |
| Training (5 epochs) | 2-2.5 hours |
| Total steps | ~3,650 (vs 14,595 on 2 GPUs) |

Steps = ceil(186812 / (4 × 2 × 8)) × 5 = ceil(186812 / 64) × 5 ≈ 2920 × 5 / 5 ≈ 14,595 total steps across epochs... actually:
- Steps per epoch = ceil(186812 / (batch × grad_acc × gpus)) = ceil(186812 / 64) = 2,919
- Total = 2,919 × 5 = 14,595 steps (same total, but each step is ~4x faster)

## Monitoring

```bash
# Watch GPU usage
watch -n 5 nvidia-smi

# wandb dashboard
# https://wandb.ai/OzLabs/qwen3-asr-hebrew
```

## Key Checkpoints
- **Epoch 3 (~step 8,757):** Console prints "Strategy A: Unfreezing Top 8 Audio Layers", GPU memory increases +2-4GB, small loss spike is expected
- **Eval WER:** Should decrease each eval; target < 11.5%

## Troubleshooting

**OOM:** Reduce batch_size to 2, increase grad_acc to 4
**Slow data loading:** Increase `dataloader_num_workers` (currently 8, try 16)
**NCCL timeout:** `export NCCL_TIMEOUT=1800`

## Output

Model saved to `./qwen3-asr-hebrew-round2/` with checkpoints every 500 steps.
