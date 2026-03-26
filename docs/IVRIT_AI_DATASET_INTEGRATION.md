# ivrit.ai Dataset Integration & SOTA Methods

**Goal:** Integrate ivrit.ai's full dataset suite + their proven training methods to match/beat SOTA

---

## Current vs Target Datasets

### **Currently Using (Round 1-2):**
| Dataset | Hours | Samples | Domain | Status |
|---------|-------|---------|--------|--------|
| `ivrit-ai/crowd-transcribe-v5` | ~300h | ~186k | Crowd-sourced general | ✅ Using |
| `ivrit-ai/crowd-recital-whisper-training` | ~50h | ~40k | Wikipedia recitals | ✅ Using |
| **TOTAL** | **~350h** | **~226k** | Mixed informal | |

### **Available to Add:**
| Dataset | Hours | Samples | Domain | Priority |
|---------|-------|---------|--------|----------|
| `ivrit-ai/knesset-plenums-whisper-training` | ~4,700h | ~100k-1M | Formal parliamentary | 🔴 HIGH |
| `ivrit-ai/whisper-training` | Unknown | 10k-100k | Mixed (older) | 🟡 MEDIUM |

### **ivrit.ai SOTA Uses:**
- **Knesset:** 90% of training (4,700h)
- **Crowd Recital:** 7.5% of training (50h)
- **Crowd Transcribe v5:** 2.5% of training (300h)

**Their total:** ~5,050 hours
**Our total:** ~350 hours (14x smaller!)

---

## Dataset Special Features

### **All ivrit.ai Whisper Datasets Include:**

#### 1. **Timestamp Tokens** (Critical for SOTA)
- Format: `<|0.00|>text here<|2.40|>more text<|5.20|>`
- Field: `has_timestamps` (boolean flag)
- **ivrit.ai uses:** 40% of samples with timestamps
- **Why it matters:** Prevents catastrophic forgetting in long-form transcription
- **Our status:** ❌ We strip timestamps in `HebrewTextNormalizer`

#### 2. **Previous Context** (Critical for SOTA)
- Field: `prev_transcript` (text from previous audio slice)
- **ivrit.ai uses:** 50% of samples with previous context
- **Why it matters:** Improves coherence in multi-turn conversations
- **Our status:** ❌ Not implemented

#### 3. **Metadata Fields:**
- `seek_time`: Float marking slice start time in original audio
- `source`: Original recording source (podcast, production, etc.)
- `quality_score`: Quality metric (if available)
- Audio: 16kHz MP3, max 30s per slice

---

## ivrit.ai's SOTA Training Methods

Based on [their Feb 2025 blog post](https://www.ivrit.ai/en/2025/02/13/training-whisper/):

### **1. Dataset Interleaving (Sampling Probabilities)**
```python
sampling_probs = {
    "knesset": 0.90,      # 90% Knesset (formal domain)
    "recital": 0.075,     # 7.5% Recital (Wikipedia)
    "transcribe": 0.025   # 2.5% Transcribe (crowd)
}
```

**Why:** Balance formal (Knesset) with informal (crowd) to generalize
**Our status:** ⚠️ We concatenate equally, no domain weighting

### **2. Timestamp Preservation Strategy**
```python
# Apply to 40% of samples
if random.random() < 0.4:
    keep_timestamps = True
else:
    strip_timestamps = True
```

**Why:** Prevents catastrophic forgetting while still learning from clean text
**Our status:** ❌ Always strip timestamps

### **3. Previous Context Strategy**
```python
# Apply to 50% of samples
if random.random() < 0.5:
    prompt = prev_transcript
else:
    prompt = None
```

**Why:** Teaches model to use context for better coherence
**Our status:** ❌ Never use previous context

### **4. Two-Phase Training**
**Phase 1: Domain Pretraining**
- Train on Knesset only (4,700h) × 3 epochs
- Learn formal Hebrew speech patterns
- LR: 1e-5 with linear decay

**Phase 2: Mixed Fine-tuning**
- Train on all datasets × 2 epochs
- Sampling probs: (0.9, 0.075, 0.025)
- Same LR: 1e-5 with 800-step warmup

**Our status:** ✅ Similar (gradual unfreezing = 2-phase)

### **5. Model Averaging**
```python
# After training, merge 3 best checkpoints by eval loss
checkpoints = [
    "checkpoint-1000",  # Eval loss: 0.152
    "checkpoint-2000",  # Eval loss: 0.149
    "checkpoint-3000",  # Eval loss: 0.151
]
# Weighted average by inverse loss
```

**Why:** Ensemble-like benefits, 2-5% WER reduction
**Our status:** ❌ Keep only best single checkpoint

### **6. Hardware & Scale**
- **GPUs:** 8x Nvidia A40 (48GB VRAM each)
- **Batch size:** 32 (vs our 2-8)
- **Precision:** BF16 mixed precision with SDPA
- **Training time:** ~55 hours total

**Our status:** ⚠️ We have 8x A100 (better), but smaller batch size

---

## Deduplication Strategy

Since we're adding new datasets, we need careful deduplication:

### **Potential Overlaps:**

1. **`crowd-recital-whisper-training` vs `whisper-training`**
   - Both may contain Wikipedia recitals
   - **Strategy:** Use `crowd-recital-whisper-training` (newer, has timestamps)
   - Skip `whisper-training` (older, may be superseded)

2. **`crowd-transcribe-v5` vs `whisper-training`**
   - `whisper-training` is older, may be subset of v5
   - **Strategy:** Use v5 only (newer, larger)

3. **Knesset has no overlaps** (unique source)

### **Deduplication Implementation:**

```python
def deduplicate_datasets(datasets_dict):
    """
    Deduplicate by audio content hash.

    Args:
        datasets_dict: {"name": Dataset}

    Returns:
        Deduplicated combined dataset
    """
    import hashlib

    seen_hashes = set()
    deduplicated = []

    for name, ds in datasets_dict.items():
        for example in ds:
            # Hash audio bytes
            audio_hash = hashlib.md5(example["audio"]["array"].tobytes()).hexdigest()

            if audio_hash not in seen_hashes:
                seen_hashes.add(audio_hash)
                deduplicated.append(example)
            else:
                print(f"  Duplicate found in {name}: {audio_hash[:8]}...")

    return Dataset.from_list(deduplicated)
```

---

## Recommended Dataset Configuration

### **Round 2.5 (Quick Addition - No Knesset Yet):**

```python
# In TrainingConfig
datasets = [
    "ivrit-ai/crowd-transcribe-v5",           # ~300h, 2.5% sampling
    "ivrit-ai/crowd-recital-whisper-training"  # ~50h, 7.5% sampling
]

# Sampling probabilities (without Knesset)
sampling_probs = {
    "crowd-transcribe-v5": 0.25,           # 25% (boost from 2.5%)
    "crowd-recital-whisper-training": 0.75  # 75% (boost from 7.5%)
}
```

**Rationale:** Without Knesset, weight Recital higher (cleaner, has timestamps)

### **Round 3 (Full SOTA Replication):**

```python
datasets = [
    "ivrit-ai/knesset-plenums-whisper-training",  # ~4,700h, 90% sampling
    "ivrit-ai/crowd-recital-whisper-training",     # ~50h, 7.5% sampling
    "ivrit-ai/crowd-transcribe-v5"                 # ~300h, 2.5% sampling
]

# Exact ivrit.ai sampling
sampling_probs = {
    "knesset-plenums-whisper-training": 0.90,
    "crowd-recital-whisper-training": 0.075,
    "crowd-transcribe-v5": 0.025
}
```

**Total:** ~5,050 hours (matches ivrit.ai SOTA)

---

## Implementation Plan

### **Phase 1: Add Timestamp/Context Support (Priority 🔴)**

**File:** `train_hebrew_asr_enhanced.py`

#### 1. Modify `HebrewTextNormalizer` to conditionally preserve timestamps:

```python
class HebrewTextNormalizer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.timestamp_prob = 0.4  # 40% keep timestamps (ivrit.ai method)
        self.timestamp_pattern = re.compile(r'<\|[\d.]+\|>')

    def normalize(self, text: str, keep_timestamps: bool = None) -> str:
        """Apply normalization, optionally preserving timestamps."""

        # Decide randomly if not specified
        if keep_timestamps is None:
            keep_timestamps = random.random() < self.timestamp_prob

        # Remove timestamps if not keeping
        if not keep_timestamps and self.config.remove_timestamps:
            text = self.timestamp_pattern.sub('', text)

        # ... rest of normalization
        return text
```

#### 2. Add previous context support in `AudioDataPreprocessor`:

```python
class AudioDataPreprocessor:
    def preprocess_function(self, batch):
        """Process batch with optional previous context."""

        # Extract previous context if available (50% probability)
        prev_context = []
        for i, example in enumerate(batch):
            if "prev_transcript" in example and random.random() < 0.5:
                prev_context.append(example["prev_transcript"])
            else:
                prev_context.append("")

        # Prepend context to prompt for Qwen3-ASR
        # Format: "language Hebrew<asr_text>{prev_context}{current_text}"
        prompts = [
            f"language Hebrew<asr_text>{prev}{text}"
            for prev, text in zip(prev_context, batch["text"])
        ]

        # ... rest of processing
```

#### 3. Implement dataset interleaving with sampling:

```python
from datasets import interleave_datasets

def load_interleaved_datasets(config):
    """Load datasets with ivrit.ai sampling strategy."""

    datasets_dict = {}
    sampling_probs = []

    # Define sampling probabilities
    probs = {
        "ivrit-ai/knesset-plenums-whisper-training": 0.90,
        "ivrit-ai/crowd-recital-whisper-training": 0.075,
        "ivrit-ai/crowd-transcribe-v5": 0.025,
    }

    for name in config.datasets:
        ds = load_dataset(name, split="train")
        datasets_dict[name] = ds
        sampling_probs.append(probs.get(name, 1.0 / len(config.datasets)))

    # Normalize probabilities
    total = sum(sampling_probs)
    sampling_probs = [p / total for p in sampling_probs]

    # Interleave with probabilities
    combined = interleave_datasets(
        list(datasets_dict.values()),
        probabilities=sampling_probs,
        seed=42
    )

    return combined
```

### **Phase 2: Add Knesset Dataset (Priority 🔴)**

**Prerequisites:**
1. Verify Knesset dataset is accessible (public on HuggingFace)
2. Check storage requirements (~4,700h ≈ 150GB compressed)
3. Prepare for longer training time (5,050h vs 350h = 14x longer)

**Steps:**
1. Add to config: `datasets = ["ivrit-ai/knesset-plenums-whisper-training", ...]`
2. Set sampling probs: `(0.90, 0.075, 0.025)`
3. **Two-phase training:**
   - Phase 1: Knesset only × 3 epochs
   - Phase 2: Mixed × 2 epochs
4. Expect 2-3 days training on 8x A100

### **Phase 3: Implement Model Averaging (Priority 🟡)**

```python
# After training, in main()
from transformers import AutoModel
import torch

def average_checkpoints(checkpoint_paths, output_path):
    """Average weights from multiple checkpoints."""

    print(f"\nAveraging {len(checkpoint_paths)} checkpoints...")

    # Load all state dicts
    state_dicts = []
    for cp in checkpoint_paths:
        sd = torch.load(f"{cp}/pytorch_model.bin")
        state_dicts.append(sd)

    # Average weights
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

    # Save averaged model
    torch.save(avg_state, f"{output_path}/pytorch_model.bin")
    print(f"✓ Saved averaged model to {output_path}")

# Usage after training:
best_checkpoints = [
    "./qwen3-asr-hebrew/checkpoint-1000",
    "./qwen3-asr-hebrew/checkpoint-2000",
    "./qwen3-asr-hebrew/checkpoint-3000"
]
average_checkpoints(best_checkpoints, "./qwen3-asr-hebrew-averaged")
```

---

## Expected Impact by Phase

| Phase | Changes | Expected WER | vs SOTA | Timeline |
|-------|---------|--------------|---------|----------|
| **Current (Round 2)** | SpecAugment, Cosine LR, Fixed batch | 8.6-10.5% | -3.5-5.4% | Today |
| **Phase 1 (Round 2.5)** | + Timestamps, Context, Model avg | 7.0-7.5% | -1.9-2.4% | 1 week |
| **Phase 2 (Round 3)** | + Knesset data (4,700h) | 6.0-6.5% | -0.9-1.4% | 2-3 weeks |
| **Phase 3 (Round 3+)** | + Multi-stage (SFT+RL), Hyperparam search | **4.5-5.5%** | **✅ Beat SOTA** | 4-6 weeks |

---

## Immediate Action Items

### **Today (Round 2):**
✅ Run current training with SpecAugment, Cosine LR, etc.
✅ Validate optimizations work
✅ Benchmark on eval-d1

### **This Week (Round 2.5):**
1. 🔴 Implement timestamp preservation (40% probability)
2. 🔴 Implement previous context (50% probability)
3. 🟡 Implement model averaging (3 best checkpoints)
4. 🟡 Add dataset interleaving with sampling probs
5. ⚪ Test without Knesset first (use current 350h)

### **Next 2-4 Weeks (Round 3):**
1. 🔴 Add Knesset dataset (4,700h)
2. 🔴 Implement two-phase training (Knesset-only → Mixed)
3. 🔴 Scale batch size to 32 (match ivrit.ai)
4. 🟡 Implement deduplication logic
5. 🟡 Multi-stage training (SFT + GRPO per Qwen3-ASR paper)

---

## Files to Modify

1. **`train_hebrew_asr_enhanced.py`:**
   - `HebrewTextNormalizer.normalize()` - add timestamp preservation
   - `AudioDataPreprocessor.preprocess_function()` - add prev context
   - `main()` - add dataset interleaving with sampling
   - Add `average_checkpoints()` function

2. **`config.yaml` (or TrainingConfig):**
   ```python
   datasets: List[str] = [
       "ivrit-ai/knesset-plenums-whisper-training",  # Add this
       "ivrit-ai/crowd-recital-whisper-training",
       "ivrit-ai/crowd-transcribe-v5"
   ]

   # New fields
   timestamp_keep_prob: float = 0.4   # 40% keep timestamps
   prev_context_prob: float = 0.5     # 50% use previous context
   dataset_sampling_probs: Dict[str, float] = {
       "knesset-plenums-whisper-training": 0.90,
       "crowd-recital-whisper-training": 0.075,
       "crowd-transcribe-v5": 0.025
   }
   num_checkpoints_to_average: int = 3
   ```

---

## Key Takeaway

**To beat SOTA (5.1% WER), we MUST:**
1. ✅ Use ivrit.ai's proven training methods (timestamps, context, model averaging)
2. ✅ Add Knesset data (4,700h) for domain coverage
3. ✅ Match their scale (batch size, training time)

**Current Round 2 optimizations alone won't beat SOTA** - they'll get us competitive (7-9% WER), but not best-in-class (5-6% WER).

**Realistic path:** Round 2 → Round 2.5 (methods) → Round 3 (Knesset) → **Beat SOTA**
