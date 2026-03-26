# Round 2 Training - Technical Specification

**Document Type:** Technical Design Document
**Version:** 1.0
**Date:** 2026-02-10
**Authors:** ML Team

---

## Executive Summary

Round 2 implements a gradual unfreezing training strategy optimized for 2x A100 (40GB) GPUs, targeting 1-2 WER point improvement over Round 1. Key innovations include Phase 0 forced aligner data quality audit, selective layer freezing with discriminative learning rates, and comprehensive experiment tracking.

**Key Technical Achievements:**
- 53% cost reduction: $77 → $38 (8x A100 → 2x A100)
- Memory optimization: Selective freezing reduces trainable params by 53%
- Data quality gate: Phase 0 audit prevents wasted compute on bad data
- Reproducibility: Full W&B experiment tracking with Phase 0 integration

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Architecture Details](#model-architecture-details)
3. [Training Strategy](#training-strategy)
4. [Data Pipeline](#data-pipeline)
5. [Memory Management](#memory-management)
6. [Optimization Details](#optimization-details)
7. [Evaluation Methodology](#evaluation-methodology)
8. [Implementation Details](#implementation-details)
9. [Risk Analysis](#risk-analysis)
10. [Future Work](#future-work)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Round 2 Training Pipeline                 │
└─────────────────────────────────────────────────────────────┘

┌───────────────┐
│   Phase 0     │  Stratified data quality audit
│ Forced Aligner│  - 10% sample (40% noisy, 40% clean, 20% tail)
│   Audit       │  - Alignment coverage metrics
└───────┬───────┘  - Decision gate: <10% low quality → PROCEED
        │
        ▼
┌───────────────┐
│ Decision Gate │  Review alignment_report.json
│   Review      │  - Coverage distribution
└───────┬───────┘  - Per-domain breakdown
        │
        ▼ (if PROCEED)
┌───────────────────────────────────────────────────────────────┐
│                    Phase 2: Training                          │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Epochs 1-2: Strategy B                               │   │
│  │ ┌────────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │ │   Audio    │  │ Projector│  │   LLM Decoder    │  │   │
│  │ │   Tower    │→ │ (proj1,  │→ │                  │  │   │
│  │ │ 24 layers  │  │  proj2,  │  │ 28 layers        │  │   │
│  │ │            │  │  ln_post)│  │                  │  │   │
│  │ │ ❄️ FROZEN  │  │ 🔥 LR:2e-4│  │ Layers 0-15: ❄️  │  │   │
│  │ │            │  │          │  │ Layers 16-27: 🔥 │  │   │
│  │ └────────────┘  └──────────┘  │  (LR: 5e-5)      │  │   │
│  │                                └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼ (Epoch 3)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Epochs 3-5: Strategy A                               │   │
│  │ ┌────────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │ │   Audio    │  │ Projector│  │   LLM Decoder    │  │   │
│  │ │   Tower    │→ │ (proj1,  │→ │                  │  │   │
│  │ │            │  │  proj2,  │  │ 28 layers        │  │   │
│  │ │ Layers 0-15│  │  ln_post)│  │                  │  │   │
│  │ │ ❄️ FROZEN  │  │ 🔥 LR:1e-4│  │ Layers 0-15: ❄️  │  │   │
│  │ │ Layers16-23│  │          │  │ Layers 16-27: 🔥 │  │   │
│  │ │ 🔥 LR:3e-5 │  │          │  │  (LR: 3e-5)      │  │   │
│  │ └────────────┘  └──────────┘  └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────┐
│  Evaluation   │  Benchmark on eval sets
│ & Comparison  │  - Round 1 vs Round 2 WER
└───────────────┘  - Per-domain breakdown
```

### Component Breakdown

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Phase 0 Auditor** | Qwen3-ForcedAligner-0.6B | Data quality assessment |
| **Model Base** | Qwen3-ASR-1.7B | Pretrained Hebrew ASR |
| **Freezing Logic** | PyTorch `requires_grad` | Selective parameter training |
| **Optimizer** | AdamW (fused) | Gradient-based optimization |
| **LoRA Adapters** | PEFT library | Low-rank adaptation layers |
| **Experiment Tracking** | Weights & Biases | Metrics, logs, checkpoints |
| **Data Loading** | HuggingFace Datasets | Streaming audio datasets |
| **Training Framework** | HuggingFace Trainer | Training loop orchestration |

---

## Model Architecture Details

### Qwen3-ASR-1.7B Structure

**Total Parameters:** 1,715,432,120 (~1.7B)

**Module Hierarchy:**
```
thinker
├── audio_tower
│   ├── layers.0-23 (24 layers)          [24 × Whisper-style encoder blocks]
│   │   ├── self_attn
│   │   ├── layer_norm1
│   │   ├── mlp
│   │   └── layer_norm2
│   ├── proj1 (Linear: 1280 → 2560)      [First projection layer]
│   ├── proj2 (Linear: 2560 → 1536)      [Second projection layer]
│   └── ln_post (LayerNorm: 1536)        [Post-projection normalization]
│
└── model (LLM decoder)
    ├── embed_tokens (Embedding)          [Token embeddings]
    ├── layers.0-27 (28 layers)          [28 × Qwen decoder blocks]
    │   ├── self_attn (QKV attention)
    │   ├── mlp (SwiGLU feed-forward)
    │   └── input_layernorm, post_attention_layernorm
    ├── norm (RMSNorm)                    [Final layer norm]
    └── lm_head (Linear: 1536 → 151936)  [Output projection]
```

### Parameter Distribution

| Module | Parameters | Percentage | Round 2 Status |
|--------|-----------|------------|----------------|
| Audio tower (layers.0-15) | ~400M | 23.3% | ❄️ FROZEN (both strategies) |
| Audio tower (layers.16-23) | ~124M | 7.2% | ❄️ FROZEN (B), 🔥 TRAIN (A) |
| Projector (proj1, proj2, ln_post) | ~4M | 0.2% | 🔥 TRAIN (both) |
| LLM layers.0-15 (bottom) | ~650M | 37.9% | ❄️ FROZEN (both) |
| LLM layers.16-27 (top) | ~487M | 28.4% | 🔥 TRAIN (both) |
| LM head | ~321M | 18.7% | 🔥 TRAIN (both) |
| **Total** | **1,715M** | **100%** | - |

### Trainable Parameters by Strategy

**Strategy B (Epochs 1-2):**
- Projector: 4M
- LLM top 12 layers: 487M
- LM head: 321M
- **Total trainable:** 812M (47.3%)
- **Frozen:** 903M (52.7%)

**Strategy A (Epochs 3-5):**
- Audio top 8 layers: 124M (newly unfrozen)
- Projector: 4M
- LLM top 12 layers: 487M
- LM head: 321M
- **Total trainable:** 936M (54.6%)
- **Frozen:** 779M (45.4%)

### Cross-Modal Architecture

**Audio → Text Flow:**
```
Input Audio (16kHz WAV)
    ↓
Audio Tower (24 Whisper-style layers)
    ├─ Layer 0-23: Self-attention over audio features
    ├─ Output: [batch, audio_seq_len, 1280] features
    ↓
Projector (proj1 → proj2 → ln_post)
    ├─ proj1: 1280 → 2560 (expansion)
    ├─ proj2: 2560 → 1536 (compression)
    ├─ ln_post: LayerNorm(1536)
    ├─ Output: [batch, audio_seq_len, 1536] aligned features
    ↓
LLM Decoder (28 Qwen layers)
    ├─ Layers 0-27: Causal self-attention
    ├─ Input: Concatenate([BOS], audio_features, text_tokens)
    ├─ Output: [batch, seq_len, 1536] contextualized
    ↓
LM Head (1536 → 151936 vocab)
    ├─ Linear projection to logits
    └─ Output: [batch, seq_len, vocab_size]
```

**Key Insight:** The projector (proj1, proj2, ln_post) is the **cross-modal bottleneck**. This is why we:
1. Always train it (both strategies)
2. Use higher learning rate (2e-4 → 1e-4)
3. It adapts Hebrew phonemes → LLM token space

---

## Training Strategy

### Gradual Unfreezing Rationale

**Why Not Full Fine-Tuning?**
- Requires 8x A100 ($77 for 12h)
- Risk of catastrophic forgetting
- May overfit on limited Hebrew data (~73k samples)

**Why Not Pure LoRA?**
- Hebrew phoneme space may differ from pretraining
- Cross-modal adaptation requires projector training
- Top LLM layers need fine-grained Hebrew patterns

**Why Gradual Unfreezing?**
1. **Phase 1 (Strategy B):** Adapt cross-modal interface and high-level language understanding
2. **Phase 2 (Strategy A):** Refine acoustic feature extraction once language model knows what to look for
3. **Coarse-to-fine:** Prevents low-level audio features from disrupting high-level language patterns

### Strategy B (Epochs 1-2): Language Adaptation

**Trainable:**
- Projector (4M params)
- LLM top 12 layers (487M params)
- LM head (321M params)

**Frozen:**
- Audio tower (all 24 layers)
- LLM bottom 16 layers

**Learning Rates:**
- Projector: 2e-4 (highest - cross-modal bottleneck)
- LLM top: 5e-5 (standard fine-tuning rate)
- LM head: 1e-4 (output layer)

**Rationale:**
- Focus on mapping audio features → Hebrew text
- Audio tower features are good (pretrained on multilingual data)
- Bottom LLM preserves general language understanding
- Top LLM learns Hebrew-specific patterns

**Expected Progress:**
- Epoch 1: WER drops from ~22% → ~16%
- Epoch 2: WER drops to ~13%

### Strategy A (Epochs 3-5): Acoustic Refinement

**Newly Unfrozen:**
- Audio tower top 8 layers (layers.16-23, 124M params)

**Learning Rates (adjusted):**
- Projector: 1e-4 (reduced for stability)
- Audio top: 3e-5 (conservative start)
- LLM top: 3e-5 (reduced)
- LM head: 1e-4 (unchanged)

**Rationale:**
- Now that LLM knows Hebrew patterns, refine audio features
- Top audio layers capture high-level acoustic patterns
- Lower LRs prevent disrupting learned language mappings
- Bottom audio layers stay frozen (preserve low-level features)

**Expected Progress:**
- Epoch 3: Small loss spike (0.20 → 0.30, recovers quickly)
- Epochs 3-5: WER drops from ~13% → ~10.5%

### Discriminative Learning Rates

**Why different LRs per layer group?**

| Layer Group | LR Range | Justification |
|-------------|----------|---------------|
| Projector | 1e-4 to 2e-4 | Cross-modal bottleneck, needs fastest adaptation |
| Audio top | 3e-5 | High-level acoustic features, conservative |
| LLM top | 3e-5 to 5e-5 | Task-specific patterns, moderate |
| LM head | 1e-4 | Output layer, higher LR standard practice |
| Audio bottom | 0 (frozen) | Low-level features universal, preserve |
| LLM bottom | 0 (frozen) | General language understanding, preserve |

**Implementation:**
```python
param_groups = [
    {"params": projector_params, "lr": 2e-4, "name": "projector"},
    {"params": audio_top_params, "lr": 3e-5, "name": "audio_top"},
    {"params": llm_top_params, "lr": 5e-5, "name": "llm_top"},
    {"params": lm_head_params, "lr": 1e-4, "name": "lm_head"},
]
optimizer = AdamW(param_groups, ...)
```

---

## Data Pipeline

### Phase 0: Stratified Quality Audit

**Objective:** Assess data quality before training to avoid wasting compute on misaligned data.

**Methodology:**

1. **Stratified Sampling (10% of training data):**
   - **40% WhatsApp/noisy domain:** High error rate, critical to validate
   - **40% KAN/clean domain:** Broadcast speech, different acoustic properties
   - **20% Long-tail:** Long segments, edge cases

2. **Domain Classification Heuristics:**
```python
def categorize_domain(dataset_name, duration, text):
    if "whatsapp" in dataset_name.lower():
        return "whatsapp_noisy"
    if any(x in dataset_name.lower() for x in ["kan", "saspeech", "broadcast"]):
        return "kan_clean"
    if duration > 20.0:
        return "long_tail"
    # Fallback logic based on duration and text length
```

3. **Forced Aligner Metrics:**
```python
result = Qwen3ForcedAligner.align(audio, transcript, language="Hebrew")

# Key metrics:
coverage = aligned_words / expected_words
confidence = mean(word_confidences)
length_mismatch = abs(audio_duration - expected_duration) > threshold

# Quality flag:
low_quality = (coverage < 0.6) or (confidence < 0.5) or length_mismatch
```

4. **Decision Gate:**
   - **<10% low quality:** ✅ PROCEED (data acceptable)
   - **10-15% low quality:** ⚠️ CAUTION (proceed with monitoring)
   - **>15% low quality:** ❌ STOP (filter/resegment data first)

**Output:** `phase0_audit_results/alignment_report.json`

**Time:** 2-3 hours (CPU only, ~2,500 samples @ 1.5 sec/sample)

### Training Data Pipeline

**Datasets:**
- `ivrit-ai/crowd-transcribe-v5`: 45,123 samples
- `ivrit-ai/crowd-recital-whisper-training`: 28,456 samples
- **Total:** 73,579 training samples

**Preprocessing Steps:**

1. **Audio Processing:**
```python
# Load with librosa (16kHz resampling)
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

# Duration filtering
if not (0.5 <= duration <= 15.0):
    skip_sample()

# Chunk long segments with overlap (if needed)
if duration > 15.0:
    chunks = chunk_with_overlap(audio, chunk_size=15.0, overlap=1.0)
```

2. **Hebrew Text Normalization:**
```python
def normalize_hebrew_text(text):
    # Remove niqqud (vowel diacritics)
    text = re.sub(r'[\u0591-\u05C7]', '', text)

    # Unify geresh and gershayim (Hebrew quotation marks)
    text = text.replace('״', '"').replace('׳', "'")

    # Remove Whisper timestamp tokens
    text = re.sub(r'<\|[\d:.]+\|>', '', text)

    # Clean duplicate punctuation
    text = re.sub(r'([.!?,])\1+', r'\1', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()
```

3. **Qwen3-ASR Prompt Format:**
```python
prompt = f"language Hebrew<asr_text>{normalized_text}"
# Example: "language Hebrew<asr_text>שלום עולם זה טקסט לדוגמה"
```

4. **Duration Bucketing:**
```python
# Group samples by duration for efficient batching
# Reduces padding overhead
dataset = dataset.sort(key=lambda x: audio_duration(x))
```

**Data Collation:**
```python
# Pad audio to longest in batch
audio_padded = pad_sequence(audio_tensors, batch_first=True)

# Pad text with processor.tokenizer.pad_token_id
text_padded = pad_sequence(text_tensors, batch_first=True)

# Attention masks
attention_mask = (audio_padded != pad_value)
```

**Memory-Efficient Loading:**
- Streaming mode for large datasets
- On-the-fly resampling (16kHz)
- Lazy audio decoding (only when sampled)

---

## Memory Management

### Memory Budget (Per GPU)

**Hardware:** A100 40GB

**Memory Allocation:**

| Component | BF16 Size | Notes |
|-----------|-----------|-------|
| **Model Weights** | | |
| Audio tower | 1.2 GB | 24 layers × 50M params each |
| Projector | 8 MB | Small projection layers |
| LLM decoder | 1.9 GB | 28 layers × 60M params each |
| LM head | 320 MB | Vocab size 151936 |
| **Subtotal: Model** | **~3.4 GB** | All parameters in BF16 |
| | | |
| **Optimizer States (AdamW)** | | |
| First moment (m) | 1.7 GB | Same size as trainable params |
| Second moment (v) | 1.7 GB | Same size as trainable params |
| Master weights | 1.7 GB | FP32 copy for BF16 training |
| **Subtotal: Optimizer** | **~5.1 GB** | For 936M trainable params (Strategy A) |
| | | |
| **Gradients** | | |
| Backward pass | 1.7 GB | Same size as trainable params |
| **Subtotal: Gradients** | **~1.7 GB** | |
| | | |
| **Activations (batch_size=2)** | | |
| Audio tower forward | 4.5 GB | Intermediate activations |
| LLM forward | 6.2 GB | Attention + FFN activations |
| Gradient checkpointing | -4.3 GB | Recompute instead of store |
| **Subtotal: Activations** | **~6.4 GB** | After checkpointing |
| | | |
| **Miscellaneous** | | |
| CUDA kernels | 1.2 GB | cuBLAS, cuDNN |
| PyTorch overhead | 0.8 GB | Framework memory |
| Data loading buffers | 0.5 GB | Batch prefetching |
| **Subtotal: Misc** | **~2.5 GB** | |
| | | |
| **Total Estimated** | **~19.1 GB** | Strategy B (812M trainable) |
| **Total Estimated** | **~20.8 GB** | Strategy A (936M trainable) |
| **Safety Margin** | **~19-20 GB** | |
| **Available** | **40 GB** | |
| **Utilization** | **52-58%** | Conservative, safe |

### Memory Optimization Techniques

**1. BF16 Mixed Precision:**
- Weights stored in BF16 (2 bytes vs 4 for FP32)
- Master weights in FP32 (optimizer requirement)
- Activations in BF16
- 50% memory savings vs FP16 training

**2. Gradient Checkpointing:**
```python
model.gradient_checkpointing_enable()
# Recompute activations during backward pass
# Trades compute for memory (40-60% activation savings)
```

**3. Selective Freezing:**
- Strategy B: Freeze 903M params → Save ~5.4 GB optimizer states
- Strategy A: Freeze 779M params → Save ~4.7 GB optimizer states

**4. Fused Optimizer:**
```python
optim="adamw_torch_fused"
# Fuses multiple CUDA kernels
# 10-15% memory reduction vs standard AdamW
```

**5. Length-Based Bucketing:**
```python
group_by_length=True
# Reduces padding overhead
# Groups similar-length samples in same batch
```

**6. Gradient Accumulation:**
```python
per_device_batch_size=2
gradient_accumulation_steps=16
# Effective batch = 64 (2 × 16 × 2 GPUs)
# Only update weights every 16 steps → amortize optimizer overhead
```

**7. PyTorch Memory Allocator:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Reduces memory fragmentation
# Better allocation efficiency
```

### OOM Prevention Strategy

**Monitoring:**
- Watch `nvidia-smi` every minute
- Alert if memory >38 GB (95% utilization)
- W&B logs GPU memory per step

**If approaching OOM:**
1. **First:** Reduce batch_size: 2 → 1, double grad_accumulation
2. **Second:** Reduce max_audio_length: 15s → 12s
3. **Third:** Enable more aggressive gradient checkpointing
4. **Last resort:** Freeze more layers (e.g., top 10 LLM instead of 12)

---

## Optimization Details

### Optimizer Configuration

**AdamW (Fused):**
```python
optimizer = torch.optim.AdamW(
    param_groups,  # Discriminative LRs
    betas=(0.9, 0.999),  # Standard Adam betas
    eps=1e-8,
    weight_decay=0.01,  # L2 regularization
)
```

**Learning Rate Schedule:**
```python
# Warmup for 500 steps
warmup_steps=500
# Linear warmup from 0 → target_lr

# Then constant LR (no decay)
# Rationale: Short training (5 epochs), decay not needed
```

**Gradient Clipping:**
```python
max_grad_norm=1.0
# Prevents gradient explosion
# Especially important after Strategy A switch
```

### Training Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `batch_size` | 2 | Max for 40GB A100 |
| `gradient_accumulation_steps` | 16 | Effective batch = 64 |
| `effective_batch_size` | 64 | Large enough for stable gradients |
| `learning_rate` | 5e-5 | Conservative base (discriminative LRs override) |
| `weight_decay` | 0.01 | Mild regularization |
| `warmup_steps` | 500 | Stabilize early training |
| `num_epochs` | 5 | 2 for Strategy B + 3 for Strategy A |
| `max_audio_length` | 15s | Post-resegmentation |
| `bf16` | True | Memory + speed |
| `gradient_checkpointing` | True | Memory savings |
| `group_by_length` | True | Reduce padding |
| `max_grad_norm` | 1.0 | Gradient stability |

### LoRA Configuration

**Why LoRA on top of freezing?**
- LoRA adds low-rank adapters to attention layers
- Complements selective freezing
- Allows fine-grained adaptation without full parameter training

**LoRA Settings:**
```python
lora_config = LoraConfig(
    r=16,              # Rank (standard)
    lora_alpha=32,     # Scaling factor (2×rank common)
    lora_dropout=0.1,  # Regularization
    target_modules=None,  # Auto-detect attention layers
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
```

**LoRA Parameter Count:**
- Per attention layer: ~2M params (vs 60M full)
- Total LoRA params: ~56M (across 28 layers)
- Combined with freezing: 812M (B) or 936M (A) trainable

---

## Evaluation Methodology

### Metrics

**Primary: Word Error Rate (WER)**
```python
def calculate_wer(references, hypotheses):
    """
    WER = (Substitutions + Insertions + Deletions) / Total_Words

    Lower is better. 0% = perfect transcription.
    """
    return jiwer.wer(references, hypotheses)
```

**Secondary: Character Error Rate (CER)**
```python
def calculate_cer(references, hypotheses):
    """
    CER = (Char_Subs + Char_Ins + Char_Dels) / Total_Chars

    More granular than WER for Hebrew (agglutinative).
    """
    return jiwer.cer(references, hypotheses)
```

### Evaluation Datasets

| Dataset | Samples | Domain | Round 1 WER |
|---------|---------|--------|-------------|
| eval-d1 | 200 | Dialect 1 test set | 9.2% |
| eval-whatsapp | 200 | Noisy WhatsApp audio | 17.1% |
| hebrew-speech-kan | 200 | Broadcast speech | 14.5% |
| saspeech | 200 | Clean studio recordings | 8.4% |

### Evaluation Schedule

**During Training:**
- Every 500 steps (eval_steps=500)
- Computes WER + CER on eval set
- Saves checkpoint if best WER

**Final Evaluation:**
- Full eval set (all samples)
- Both WER and CER
- Per-domain breakdown

**Round 2 Comparison:**
```bash
# Compare Round 1 vs Round 2 on all datasets
uv run python scripts/eval_round2.py \
    --round1-model OzLabs/Qwen3-ASR-Hebrew-1.7B \
    --round2-model ./qwen3-asr-hebrew-round2
```

### Duration-Bucket Analysis

**Buckets:**
- **Short:** 0.5-5s (quick utterances)
- **Medium:** 5-15s (standard speech)
- **Long:** 15-30s (extended segments)

**Per-bucket WER:**
```python
for duration_bucket in ["short", "medium", "long"]:
    bucket_samples = filter_by_duration(samples, bucket)
    bucket_wer = calculate_wer(bucket_samples)
    log(f"{bucket}: {bucket_wer:.2f}%")
```

**Purpose:**
- Identify weak duration ranges
- Guide future data augmentation
- Validate model generalization

---

## Implementation Details

### Key Functions

**1. setup_round2_freezing_strategy_b() (`train_hebrew_asr_enhanced.py:486-520`)**
```python
def setup_round2_freezing_strategy_b(model):
    """
    Freeze all parameters, then selectively unfreeze:
    - Projector (proj1, proj2, ln_post)
    - Top 12 LLM layers (layers.16-27)
    - LM head

    Frozen:
    - All audio tower layers (layers.0-23)
    - Bottom 16 LLM layers (layers.0-15)
    """
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(x in name for x in ["proj1", "proj2", "ln_post"]):
            param.requires_grad = True
        elif "model.layers" in name:
            layer_num = int(name.split("layers.")[1].split(".")[0])
            if layer_num >= 16:
                param.requires_grad = True
        elif "lm_head" in name:
            param.requires_grad = True

    return model
```

**2. unfreeze_audio_top_layers() (`train_hebrew_asr_enhanced.py:523-545`)**
```python
def unfreeze_audio_top_layers(model, num_layers=8):
    """
    Unfreeze top 8 audio layers (layers.16-23) for Strategy A.
    """
    start_layer = 24 - num_layers  # 16
    for name, param in model.named_parameters():
        if "audio_tower.layers" in name:
            layer_num = int(name.split("layers.")[1].split(".")[0])
            if layer_num >= start_layer:
                param.requires_grad = True
```

**3. create_param_groups_with_discriminative_lrs() (`train_hebrew_asr_enhanced.py:548-656`)**
```python
def create_param_groups_with_discriminative_lrs(model, epoch):
    """
    Create parameter groups with layer-specific learning rates.

    Strategy B (epochs 1-2):
        - projector: 2e-4
        - llm_top: 5e-5
        - lm_head: 1e-4

    Strategy A (epochs 3-5):
        - projector: 1e-4
        - audio_top: 3e-5
        - llm_top: 3e-5
        - lm_head: 1e-4
    """
    param_groups = {"projector": [], "llm_top": [], "lm_head": []}
    if epoch >= 3:
        param_groups["audio_top"] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(x in name for x in ["proj1", "proj2", "ln_post"]):
            param_groups["projector"].append(param)
        elif "audio_tower.layers" in name:
            param_groups["audio_top"].append(param)
        elif "model.layers" in name:
            param_groups["llm_top"].append(param)
        elif "lm_head" in name:
            param_groups["lm_head"].append(param)

    # Create optimizer param groups with LRs
    if epoch < 3:
        return [
            {"params": param_groups["projector"], "lr": 2e-4, "name": "projector"},
            {"params": param_groups["llm_top"], "lr": 5e-5, "name": "llm_top"},
            {"params": param_groups["lm_head"], "lr": 1e-4, "name": "lm_head"},
        ]
    else:
        return [
            {"params": param_groups["projector"], "lr": 1e-4, "name": "projector"},
            {"params": param_groups["audio_top"], "lr": 3e-5, "name": "audio_top"},
            {"params": param_groups["llm_top"], "lr": 3e-5, "name": "llm_top"},
            {"params": param_groups["lm_head"], "lr": 1e-4, "name": "lm_head"},
        ]
```

**4. GradualUnfreezeTrainer Class (`train_hebrew_asr_enhanced.py:659-721`)**
```python
class GradualUnfreezeTrainer(Seq2SeqTrainer):
    """Custom trainer with epoch-based unfreezing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfroze_audio = False
        self.strategy_a_enabled = False

    def training_step(self, model, inputs):
        """Check for epoch boundary and switch strategies."""
        current_epoch = int(self.state.epoch) if self.state.epoch else 1

        if current_epoch >= 3 and not self.unfroze_audio:
            print(f"\nEPOCH {current_epoch}: Switching to Strategy A")

            # Unfreeze audio top layers
            unfreeze_audio_top_layers(model, num_layers=8)

            # Recreate optimizer with new param groups + LRs
            self.optimizer = self.create_optimizer()

            self.unfroze_audio = True
            self.strategy_a_enabled = True

            # Log to W&B
            wandb.log({
                "training/strategy_switch": current_epoch,
                "training/strategy": "A",
            })

        return super().training_step(model, inputs)

    def create_optimizer(self):
        """Create optimizer with discriminative LRs."""
        current_epoch = int(self.state.epoch) if self.state.epoch else 1
        param_groups = create_param_groups_with_discriminative_lrs(
            self.model, current_epoch
        )

        return torch.optim.AdamW(
            param_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
```

### Critical Code Paths

**Training Initialization:**
```
main()
  → load model (Qwen3-ASR-1.7B)
  → setup_round2_freezing_strategy_b()  # Freeze + unfreeze selectively
  → setup_lora_model()  # Add LoRA adapters
  → load datasets
  → preprocess (normalize text, resample audio)
  → create GradualUnfreezeTrainer
  → trainer.train()
```

**Epoch Boundary (Strategy Switch):**
```
GradualUnfreezeTrainer.training_step()
  → Check: current_epoch >= 3?
    YES:
      → unfreeze_audio_top_layers()
      → create_optimizer() with new param groups
      → Log strategy switch to W&B
  → Continue training
```

---

## Risk Analysis

### High-Risk Items

**1. OOM During Strategy A Switch**
- **Risk:** GPU memory spikes when unfreezing audio layers
- **Likelihood:** Medium (33.5 GB → 37 GB spike possible)
- **Mitigation:** Monitor memory closely, reduce batch_size=1 if needed
- **Fallback:** Resume from checkpoint-1000 with batch_size=1

**2. Phase 0 Shows >15% Low Quality**
- **Risk:** Data quality insufficient, training will fail
- **Likelihood:** Low (datasets are curated)
- **Mitigation:** STOP training, escalate to data team
- **Impact:** 2-3 day delay for data filtering

**3. Training Loss Diverges After Strategy A**
- **Risk:** Unfreezing audio causes instability
- **Likelihood:** Low (conservative LRs mitigate)
- **Mitigation:** Reduce audio_top LR: 3e-5 → 1e-5
- **Fallback:** Revert to Strategy B only (2 epochs)

### Medium-Risk Items

**4. WER Improvement < 1% (11.5-12.3%)**
- **Risk:** Training doesn't improve over Round 1
- **Likelihood:** Medium
- **Mitigation:** Analyze per-domain results, may need more data
- **Impact:** Round 3 requires different approach

**5. WhatsApp Domain Doesn't Improve**
- **Risk:** Noisy domain too challenging
- **Likelihood:** Medium
- **Mitigation:** Expected (hardest domain), focus on clean domains
- **Impact:** May need codec augmentation (Round 3)

**6. Hub Push Fails**
- **Risk:** Model training completes but can't upload
- **Likelihood:** Low
- **Mitigation:** Local checkpoint preserved, manual upload later
- **Impact:** Minimal (workaround available)

### Low-Risk Items

**7. W&B Logging Fails**
- **Risk:** Metrics not logged, but training continues
- **Likelihood:** Low
- **Mitigation:** TensorBoard fallback available
- **Impact:** Minimal (local logs preserved)

**8. Dataset Download Timeout**
- **Risk:** Datasets fail to download during training
- **Likelihood:** Very Low (pre-download recommended)
- **Mitigation:** HF_TRANSFER=1, manual pre-download
- **Impact:** 1-2 hour delay

---

## Future Work

### Round 3 Enhancements (If Round 2 Successful)

**1. Codec Simulation Augmentation**
- Add MP3/Opus codec artifacts to training data
- Improve robustness on WhatsApp/compressed audio
- Expected: +0.5-1.0% WER improvement on noisy domains

**2. Discriminative LR Search**
- Grid search over LR combinations
- Find optimal LR per layer group
- Expected: +0.3-0.5% WER improvement

**3. Beam Search Optimization**
- Tune beam size and length penalty
- May provide +0.3-0.5% over greedy decoding

### Round 4: Distillation (If Round 3 Successful)

**Teacher-Student Distillation:**
- Teacher: Round 3 Qwen3-ASR-1.7B (10.0% WER)
- Student: Qwen3-ASR-0.6B (smaller, faster)
- Target: 10.5-11.0% WER at 3× faster inference

### Long-Term Roadmap

**Phase 5: Forced Aligner Quality Improvement**
- Use Round 2 model to re-align training data
- Filter/resegment low-quality samples
- Expected: 1-2% WER improvement from cleaner data

**Phase 6: SOTA Push**
- Ensemble models (if compute allows)
- Multi-task learning (ASR + punctuation)
- Target: 5-7% WER (competitive with Whisper-large-v3)

---

## References

**Papers:**
- Qwen3-ASR: [Architecture paper link]
- LoRA: "Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- Discriminative Fine-Tuning: "Universal Language Model Fine-tuning for Text Classification" (Howard & Ruder, 2018)
- Forced Alignment: "Forced Alignment with Transformers" (McAuliffe et al., 2022)

**Code:**
- HuggingFace Transformers: https://github.com/huggingface/transformers
- PEFT (LoRA): https://github.com/huggingface/peft
- Qwen-ASR: https://github.com/QwenLM/Qwen-Audio

**Datasets:**
- ivrit.ai datasets: https://huggingface.co/ivrit-ai
- Qwen3-ASR model: https://huggingface.co/Qwen/Qwen3-ASR-1.7B

---

**Document Version:** 1.0
**Last Updated:** 2026-02-10
**Status:** Ready for execution
