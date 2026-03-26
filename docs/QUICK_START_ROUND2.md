# Round 2 Training - Quick Start Guide

**TL;DR:** Run Phase 0 audit → Review results → Train if approved → Expected: 12.3% → 10.5% WER

---

## Prerequisites (5 min)

```bash
# Install dependencies
cd caspi && uv sync

# Login to services
huggingface-cli login  # https://huggingface.co/settings/tokens
wandb login           # https://wandb.ai/authorize

# Verify GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Should output: GPUs: 2
```

---

## Step 1: Phase 0 Audit (2-3 hours, CPU only)

```bash
# Set up environment
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2-$(date +%Y%m%d)"
export WANDB_PHASE0_LOGGING="true"

# Run audit
uv run python scripts/phase0_align_audit.py
```

**Check results:**
```bash
cat phase0_audit_results/alignment_report.json | jq '.decision'
```

**Decision:**
- `"PROCEED"` → Continue to Step 2
- `"CAUTION"` → Review details, proceed with monitoring
- `"STOP"` → Fix data first, do NOT train

---

## Step 2: Training (12 hours, 2x A100, ~$48)

**ONLY if Phase 0 passed!**

```bash
# Set environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

# Run training
uv run python train_round2_gradual.py
```

**When prompted:**
```
Have you reviewed Phase 0 results? (y/n): y
```

**Monitor:** https://wandb.ai/{username}/qwen3-asr-hebrew

---

## Step 3: Evaluate (1-2 hours, optional)

```bash
uv run python scripts/eval_round2.py \
    --round1-model OzLabs/Qwen3-ASR-Hebrew-1.7B \
    --round2-model ./qwen3-asr-hebrew-round2
```

---

## Expected Timeline

| Phase | Duration | Cost | Output |
|-------|----------|------|--------|
| Phase 0 | 2-3 hours | $0 | `alignment_report.json` |
| Training | ~12 hours | ~$48 | `qwen3-asr-hebrew-round2/` |
| Eval | 1-2 hours | ~$7 | `round2_comparison.csv` |

---

## Key Checkpoints

**After Phase 0:**
- Low quality % < 10%
- Coverage mean > 0.80
- Decision: PROCEED

**During Training (every 2 hours):**
- GPU memory: 28-34 GB (safe)
- Eval WER: decreasing
- No OOM warnings

**At Epoch 3 (~5 hours in):**
- Console shows "Switching to Strategy A"
- GPU memory increases +2-4 GB (expected)
- Small loss spike recovers within 100 steps

**Final Results:**
- WER: 10.5-11.0% (target)
- Model saved to `./qwen3-asr-hebrew-round2/`
- Pushed to HuggingFace Hub

---

## Troubleshooting

**OOM Error:**
```python
# Edit train_hebrew_asr_enhanced.py line 78:
batch_size: int = 1  # Down from 2
gradient_accumulation_steps: int = 32  # Up from 16
```

**Training not improving:**
- Check Phase 0 results (data quality)
- Increase LR: `learning_rate: float = 1e-4`

**Dataset download timeout:**
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
python -c "from datasets import load_dataset; load_dataset('ivrit-ai/crowd-transcribe-v5', split='train')"
```

**W&B rate limit:**
```bash
export WANDB_MODE="offline"
# Sync later: wandb sync runs/
```

---

## Emergency Stop

**Graceful:**
- Press Ctrl+C ONCE
- Wait for "Saving checkpoint..."

**Force:**
- `pkill -f train_round2_gradual.py`
- Latest checkpoint preserved

**Resume:**
- Re-run `uv run python train_round2_gradual.py`
- Automatically resumes from latest checkpoint

---

## Success Criteria

- [ ] Phase 0: <10% low quality
- [ ] Training: Completes 5 epochs without OOM
- [ ] Results: WER < 11.5% (minimum)
- [ ] Target: WER 10.5-11.0%

---

## Contact

**Before escalating:** Check `DEPLOYMENT_ROUND2.md` for detailed troubleshooting.

**Emergency:** [Contact info]

---

**Full documentation:** See `DEPLOYMENT_ROUND2.md` (35+ pages)
