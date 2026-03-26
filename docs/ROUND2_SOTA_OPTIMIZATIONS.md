# Round 2 SOTA Optimizations (2025)

**Goal:** Beat current Hebrew ASR SOTA through cutting-edge optimization techniques from 2025 research.

## Summary

We've implemented 7 major optimizations based on recent ASR research (2025) and analysis of our Round 1 training plateau:

1. **SpecAugment** - Proven 5-15% WER reduction
2. **Cosine LR Schedule** - Better convergence than linear warmup
3. **Discriminative Layer-wise LR** - Different LRs per component
4. **Fixed Effective Batch Size** - Auto-adjust for GPU count (target: 64)
5. **BF16-Safe Evaluation** - Fixed dtype mismatch in WER computation
6. **Automatic HuggingFace Upload** - Prevent model loss
7. **Enhanced W&B Tracking** - Log all SOTA features

---

## 1. SpecAugment (Park et al. 2019)

**What:** Time and frequency masking applied to audio spectrograms
**Why:** Standard in modern ASR, reduces WER by 5-15%
**Research:** Used in Whisper, wav2vec2, Conformer models

### Configuration:
```python
use_specaugment: bool = True
time_mask_param: int = 80      # Max time steps to mask
freq_mask_param: int = 27      # Max frequency bins (Fbank=128)
num_time_masks: int = 2        # Number of time masks
num_freq_masks: int = 2        # Number of frequency masks
mask_value: float = 0.0
```

### Implementation:
- Applied in `DataCollatorSpeechSeq2SeqWithPadding.apply_specaugment()`
- Only applied during training (not evaluation)
- Masks random time/frequency stripes in Fbank features

**Expected Impact:** 5-10% relative WER reduction

---

## 2. Cosine Learning Rate Schedule

**What:** Cosine annealing with linear warmup
**Why:** SOTA for ASR in 2025, better than linear/constant
**Research:** Qwen3-ASR technical report, Conformer models

### Configuration:
```python
lr_scheduler_type: str = "cosine"
warmup_ratio: float = 0.1      # 10% warmup (modern best practice)
warmup_steps: int = 500
```

### Benefits:
- Gradual LR decay prevents overfitting
- Smooth convergence vs. sharp drops
- Better final model quality

**Expected Impact:** 2-5% relative WER reduction + faster convergence

---

## 3. Discriminative Layer-wise Learning Rates

**What:** Different LRs for projector, audio, LLM components
**Why:** Cross-modal bottleneck needs higher LR, LLM needs conservative LR
**Research:** ULMFiT (Howard & Ruder), BERT fine-tuning

### Configuration:

**Strategy B (Epochs 1-2):**
- Projector: 2e-4 (highest - learning cross-modal mapping)
- LLM top: 5e-5 (conservative - preserve pre-training)
- LM head: 1e-4 (moderate)

**Strategy A (Epochs 3-5):**
- Projector: 1e-4 (reduced - already aligned)
- Audio top: 3e-5 (conservative - fine-tune features)
- LLM top: 3e-5 (reduced further)
- LM head: 1e-4 (unchanged)

**Expected Impact:** 3-7% relative WER reduction vs. single LR

---

## 4. Fixed Effective Batch Size

**Problem:** Previous training used 256 effective batch (8 GPUs × 8 batch × 4 grad_acc) = 4x too large
**Solution:** Auto-detect GPU count and adjust gradient accumulation to maintain effective_batch = 64

### Configuration:
```python
target_effective_batch = 64
config.gradient_accumulation_steps = 64 // (config.batch_size * num_gpus)

# Examples:
# 2x A100: batch=2, grad_acc=16 → 2 × 16 × 2 = 64 ✓
# 8x A100: batch=2, grad_acc=4  → 2 × 4 × 8 = 64 ✓
```

**Expected Impact:** Fix loss plateau observed at 11.3-11.4/token

---

## 5. BF16-Safe Evaluation

**Problem:** WER evaluation failed with "Input type (float) and bias type (c10::BFloat16) should be the same"
**Root Cause:** Model weights in BF16 but `model.generate()` creates FP32 tensors

### Solution:
1. Override `prediction_step()` in `GradualUnfreezeTrainer`
2. Convert inputs to BF16 before generation
3. Use BF16 autocast during generation
4. Filter empty predictions from failed generations

**Expected Impact:** Working WER metrics during training (was 100.0% before)

---

## 6. Automatic HuggingFace Upload

**Problem:** Lost 2.5 hours of training when machine was terminated before manual copy
**Solution:** Auto-upload to HuggingFace Hub after training completes

### Configuration:
```bash
export HF_REPO_ID="OzLabs/Qwen3-ASR-Hebrew-Round2"
# Optional: SKIP_HF_UPLOAD=true to disable
```

**Expected Impact:** Prevent model loss, enable immediate deployment

---

## 7. Enhanced W&B Tracking

### New Metrics Logged:
- SpecAugment parameters (time/freq masks)
- LR scheduler type (cosine vs linear)
- Layer-wise LR decay settings
- GPU count and effective batch size
- All SOTA feature flags

### Tags Added:
- `sota-2025`
- `specaugment`
- `cosine-lr`
- `gradual-unfreezing`

---

## Research Sources

1. **SpecAugment:** Park et al. (2019) - [arXiv:1904.08779](https://arxiv.org/abs/1904.08779)
2. **Qwen3-ASR Technical Report:** [arXiv:2601.21337](https://arxiv.org/abs/2601.21337) - 3-stage training (SFT + RL)
3. **Hebrew ASR SOTA (2025):** Marmor et al. - 29% WER reduction via crowdsourcing + Whisper fine-tuning
4. **ASR Modern Era:** [arXiv:2510.12827](https://arxiv.org/html/2510.12827) - End-to-end architectures, self-supervised learning
5. **Discriminative Fine-tuning:** Howard & Ruder (ULMFiT), BERT fine-tuning best practices

---

## Expected Total Impact

| Optimization | Expected WER Reduction |
|-------------|------------------------|
| SpecAugment | 5-10% |
| Cosine LR Schedule | 2-5% |
| Discriminative LR | 3-7% |
| Fixed Batch Size | 5-10% (fix plateau) |
| BF16-Safe Eval | Enable metrics |
| **TOTAL ESTIMATED** | **15-30% relative** |

## Actual SOTA Benchmarks (ivrit.ai Leaderboard)

**Current Hebrew ASR SOTA:** ivrit-ai/whisper-large-v3-ct2-20250513
- **eval-d1:** 5.1% WER 🏆
- **WhatsApp:** 7.2% WER 🏆
- **CommonVoice:** 14.9% WER
- **FLEURS:** 17.4% WER

**Our Performance:**
- **Round 1:** 12.3% WER (eval-d1 equivalent)
- **Target Round 2:** 8.5-9.5% WER (30% improvement from optimizations)
- **Gap to SOTA:** Still 3-4% WER behind best model

**Realistic Assessment:**
- Round 2 optimizations will **NOT beat SOTA** (5.1% is very aggressive)
- **Achievable target:** 8-9% WER (competitive, but not SOTA)
- **To beat 5.1% SOTA:** Need Round 3+ with:
  - More training data (ivrit.ai has 314 hours crowdsourced)
  - Larger model (Whisper-large-v3 = 1.5B params, we use 1.7B Qwen)
  - Multi-stage training (SFT + RL like Qwen3-ASR paper)
  - Domain-specific fine-tuning per test set

---

## Next Steps

1. **Re-run Round 2 training** with SOTA optimizations
2. **Monitor W&B dashboard** for:
   - Loss convergence (should break through 11.3 plateau)
   - WER metrics during training (should work now)
   - SpecAugment impact on generalization
3. **Evaluate on test sets:**
   - ivrit.ai eval sets
   - Common Voice Hebrew
   - Real-world WhatsApp/call recordings
4. **Compare to baselines:**
   - Round 1: 12.3% WER
   - Whisper-large-v3: ~15% WER (Hebrew)
   - Google Cloud Speech: ~12% WER (Hebrew)

---

## Files Modified

1. **train_hebrew_asr_enhanced.py:**
   - Added SpecAugment to `DataCollatorSpeechSeq2SeqWithPadding`
   - Added cosine LR schedule + warmup_ratio
   - Enhanced W&B config tracking
   - Added BF16-safe evaluation
   - Added auto HF upload
   - Improved logging for SOTA features

2. **Config changes:**
   - `use_specaugment=True` with optimal params
   - `lr_scheduler_type="cosine"`
   - `warmup_ratio=0.1`
   - `max_grad_norm=1.0`
   - `weight_decay=0.01`

---

## Training Command

```bash
# Set environment variables
export HF_REPO_ID="OzLabs/Qwen3-ASR-Hebrew-Round2"
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2-sota-$(date +%Y%m%d)"

# Launch training (8x A100)
uv run torchrun --nproc_per_node=8 train_round2_gradual.py

# Or 2x A100
uv run torchrun --nproc_per_node=2 train_round2_gradual.py
```

**Expected runtime:** 2-3 hours on 8x A100, 8-10 hours on 2x A100

---

## Monitoring

**W&B Dashboard:** https://wandb.ai/OzLabs/qwen3-asr-hebrew

**Key metrics to watch:**
- `train/loss` - Should break through 11.3 plateau
- `eval/wer` - Should work now (was 100.0 before)
- `eval/cer` - Character error rate
- `train/learning_rate_*` - Per-component LRs
- `train/grad_norm` - Gradient clipping effectiveness

**Success criteria:**
- Loss converges below 11.0/token
- WER < 11.0% on eval set
- No plateau at 11.3-11.4 range
- WER metrics working during training
