# Beating Hebrew ASR SOTA: Gap Analysis & Strategy

**Goal:** Beat ivrit-ai's SOTA of **5.1% WER** on eval-d1

---

## Current Performance Gap

| Model | eval-d1 WER | WhatsApp WER | CommonVoice WER | FLEURS WER |
|-------|-------------|--------------|-----------------|------------|
| **ivrit-ai SOTA** (whisper-large-v3-ct2-20250513) | **5.1%** | **7.2%** | **14.9%** | **17.4%** |
| **Our Round 1** (Qwen3-ASR-Hebrew) | **12.3%** | N/A | N/A | N/A |
| **Gap to SOTA** | **-7.2% absolute** | N/A | N/A | N/A |

**Relative improvement needed:** 58% WER reduction (12.3% → 5.1%)

---

## What Makes ivrit-ai SOTA?

Based on [ivrit.ai's training blog](https://www.ivrit.ai/en/2025/02/13/training-whisper/):

### 1. **Timestamp Preservation** (Critical for Whisper)
- **What:** 40% of training samples include timestamp tokens
- **Why:** Prevents catastrophic forgetting of long-form transcription
- **Impact:** Whisper models without timestamps forget how to handle >30s audio
- **Our status:** ❌ We strip timestamps (see `HebrewTextNormalizer.normalize()`)

### 2. **Previous Text Context**
- **What:** 50% of training samples include previous context
- **Why:** Improves coherence in long-form transcription
- **Impact:** Better handling of multi-turn conversations
- **Our status:** ❌ Not implemented

### 3. **Model Averaging**
- **What:** Weighted average of 3 best checkpoints (by eval loss)
- **Why:** Ensemble-like benefits without inference cost
- **Impact:** 2-5% relative WER reduction typical for model averaging
- **Our status:** ❌ We keep only best single checkpoint

### 4. **Massive Domain-Specific Data**
- **Knesset:** ~4,700 hours (90% of training)
- **Crowd Recital:** ~50 hours (7.5% of training)
- **Crowd Transcribe v5:** ~300 hours (2.5% of training)
- **Total:** ~5,050 hours
- **Our status:** ✅ We use ~450 hours (Crowd Transcribe v5 + Recital)

### 5. **Two-Phase Training**
- **Phase 1:** Pretrain on Knesset (4,700 hours) × 3 epochs
- **Phase 2:** Mixed training (all datasets) × 2 epochs
- **Our status:** ✅ Similar (gradual unfreezing = 2-phase)

### 6. **Hardware & Scale**
- **GPUs:** 8x Nvidia A40 (48GB each)
- **Training time:** ~55 hours total
- **Batch size:** 32 (vs our 2-8)
- **Our status:** ⚠️ We have 2-8x A100, but smaller batch size

---

## Why We Can't Beat SOTA with Current Approach

### Critical Missing Pieces:

1. **No Timestamp Preservation**
   - Qwen3-ASR uses different format than Whisper
   - Our `HebrewTextNormalizer` strips timestamps
   - Qwen doesn't have catastrophic forgetting issue like Whisper
   - **But:** We still lose temporal information that helps WER

2. **Limited Training Data**
   - ivrit-ai: **5,050 hours** (90% Knesset formal speech)
   - Us: **450 hours** (crowd-sourced informal speech)
   - **Gap:** 11x less data, different domain

3. **No Model Averaging**
   - Easy to implement (merge 3 best checkpoints)
   - **Expected gain:** 2-5% relative WER reduction
   - **Action:** Can implement for Round 2/3

4. **Architecture Difference**
   - SOTA: Whisper-large-v3 (1.55B params, encoder-decoder)
   - Us: Qwen3-ASR-1.7B (2.0B params, audio-LLM hybrid)
   - Whisper has 2 years of production optimization
   - Qwen3-ASR is newer (Jan 2025), less battle-tested

---

## Realistic Round 2 Expectations

### With Current SOTA Optimizations:
- SpecAugment: 5-10% relative reduction
- Cosine LR: 2-5% relative reduction
- Fixed batch size: 5-10% relative reduction (fix plateau)
- Layer-wise LR: 3-7% relative reduction

**Total expected:** 15-30% relative reduction
**From 12.3% WER → 8.6-10.5% WER**

### Still Missing vs SOTA:
- 8.6% (our best case) vs 5.1% (SOTA) = **3.5% gap**
- **40% worse than SOTA** even with all optimizations

---

## Path to Beating SOTA (Round 3+)

### Immediate Wins (Round 2.5 - Can implement now):

#### 1. Model Averaging (Easy)
```python
# After training, merge 3 best checkpoints
from transformers import AutoModel
import torch

checkpoints = [
    "./qwen3-asr-hebrew/checkpoint-1000",
    "./qwen3-asr-hebrew/checkpoint-2000",
    "./qwen3-asr-hebrew/checkpoint-3000"
]

# Average weights
state_dicts = [torch.load(f"{cp}/model.safetensors") for cp in checkpoints]
avg_state = {k: sum(sd[k] for sd in state_dicts) / len(state_dicts)
             for k in state_dicts[0].keys()}
```
**Expected gain:** +2-5% relative WER reduction
**Effort:** 1 hour implementation

#### 2. Timestamp Preservation (Medium difficulty)
- Modify `HebrewTextNormalizer` to preserve timestamps for 40% of samples
- Requires understanding Qwen3-ASR timestamp format
- May need to regenerate training data
**Expected gain:** +3-7% relative WER reduction
**Effort:** 1-2 days

#### 3. Speed Perturbation (Easy with torchaudio)
```python
# Add to DataCollator
import torchaudio.transforms as T

speed_perturb = T.SpeedPerturbation(sample_rate=16000, factors=[0.9, 1.0, 1.1])
```
**Expected gain:** +2-4% relative WER reduction
**Effort:** 2 hours

### Longer-term (Round 3):

#### 4. Add Knesset Data (~4,700 hours)
- Domain: Formal Israeli parliamentary speech
- Quality: Professional recordings, accurate transcripts
- **Challenge:** May need to request access from ivrit.ai
**Expected gain:** +10-20% relative WER reduction (domain-specific)
**Effort:** Data acquisition + 1 week retraining

#### 5. Multi-Stage Training (SFT + RL)
- Follow Qwen3-ASR paper methodology
- Stage 1: Supervised fine-tuning (current)
- Stage 2: GRPO (Group Relative Policy Optimization)
**Expected gain:** +5-10% relative WER reduction
**Effort:** 2-3 weeks implementation + experimentation

#### 6. Larger Batch Size
- ivrit-ai uses batch=32, we use batch=2-8
- Larger batches → more stable gradients
- **Challenge:** Requires more GPU memory or smaller model
**Expected gain:** +2-4% relative WER reduction
**Effort:** Optimize memory usage (gradient checkpointing, activation checkpointing)

---

## Recommended Strategy

### **Phase 1: Round 2 (Current - Run Today)**
✅ SpecAugment
✅ Cosine LR schedule
✅ Fixed batch size (64 effective)
✅ Layer-wise discriminative LR
✅ BF16-safe evaluation
✅ Auto HF upload

**Expected:** 12.3% → 8.6-10.5% WER
**Timeline:** 2-3 hours training on 8x A100

### **Phase 2: Round 2.5 (This week)**
🔲 Model averaging (3 best checkpoints)
🔲 Speed perturbation augmentation
🔲 Timestamp preservation (40% of samples)

**Expected:** 8.6% → 7.0-7.5% WER
**Timeline:** 2-3 days implementation + 3 hours retraining

### **Phase 3: Round 3 (Next 2-4 weeks)**
🔲 Add Knesset data (~4,700 hours)
🔲 Multi-stage training (SFT + GRPO)
🔲 Larger batch size optimization
🔲 Hyperparameter search (LR, LoRA rank, warmup)

**Expected:** 7.0% → **5.0-6.0% WER** (competitive with SOTA)
**Timeline:** 2-4 weeks

---

## Why This is Hard

1. **ivrit-ai has 11x more data** (5,050 vs 450 hours)
2. **Whisper architecture is battle-tested** (2 years production)
3. **Domain mismatch** (we train on crowd-sourced, SOTA uses formal Knesset)
4. **Different model families** (Whisper encoder-decoder vs Qwen audio-LLM)

**Realistic goal for Round 2:** Get competitive (7-9% WER), not SOTA
**Realistic goal for Round 3:** Match or beat SOTA (5-6% WER)

---

## Immediate Next Steps

1. ✅ **Run Round 2 training** with current SOTA optimizations
   - Validate that optimizations work (loss convergence, WER metrics)
   - Benchmark on eval-d1, WhatsApp, CommonVoice

2. **Implement quick wins** (model averaging, speed perturbation)
   - Can be done in parallel while Round 2 trains
   - Retrain as Round 2.5 with these additions

3. **Plan Round 3 data strategy**
   - Contact ivrit.ai about Knesset data access
   - Investigate other Hebrew ASR datasets
   - Consider synthetic data generation

4. **Benchmark rigorously**
   - Test on all 6 ivrit.ai leaderboard datasets
   - Compare apples-to-apples (same test sets)
   - Document results for reproducibility

---

## Sources

- [ivrit.ai Whisper Training Blog (Feb 2025)](https://www.ivrit.ai/en/2025/02/13/training-whisper/)
- [ivrit.ai Hebrew ASR Leaderboard](https://huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard)
- [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)
- [Whisper Fine-Tuning Best Practices](https://huggingface.co/blog/fine-tune-whisper)
- [Model Averaging for ASR (Springer)](https://link.springer.com/article/10.1186/s13636-024-00349-3)
