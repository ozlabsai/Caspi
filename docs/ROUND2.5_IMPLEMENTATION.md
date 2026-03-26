# Round 2.5: ivrit.ai SOTA Methods Implementation

**Status:** ✅ **Complete - Ready to Train**

**Goal:** Implement all ivrit.ai SOTA methods + synthetic augmentation with **balanced sampling** (not 90% Knesset)

---

## What Was Implemented

### 1. ✅ **Timestamp Preservation** (40% probability)
**File:** `train_hebrew_asr_enhanced.py` - `HebrewTextNormalizer.normalize()`

**What:**
- Conditionally preserve Whisper timestamp tokens like `<|0.00|>text<|2.40|>`
- Random 40% probability per sample (ivrit.ai method)
- Field: `has_timestamps` (from datasets)

**Why:** Prevents catastrophic forgetting in long-form transcription (>30s audio)

**Code:**
```python
def normalize(self, text: str, keep_timestamps: bool = None) -> str:
    # Decide randomly (40% keep)
    if keep_timestamps is None:
        import random
        keep_timestamps = random.random() < self.config.timestamp_keep_prob

    # Remove timestamps if not keeping
    if not keep_timestamps:
        text = self.timestamp_pattern.sub('', text)
```

**Expected impact:** 3-7% WER reduction

---

### 2. ✅ **Previous Context Support** (50% probability)
**File:** `train_hebrew_asr_enhanced.py` - `preprocess_function()`

**What:**
- Use `prev_transcript` field from datasets
- Prepend to current transcript with 50% probability
- Improves coherence in multi-turn conversations

**Code:**
```python
# Add previous context with probability (ivrit.ai: 50%)
if "prev_transcript" in examples and random.random() < self.config.prev_context_prob:
    prev_text = examples["prev_transcript"][i]
    if prev_text:
        text = f"{prev_text} {text}"
```

**Expected impact:** 2-4% WER reduction

---

### 3. ✅ **Synthetic Audio Augmentation**
**File:** `train_hebrew_asr_enhanced.py` - `apply_audio_augmentation()`

**What:**
Three augmentation techniques applied with 50% probability:

#### A. **Speed Perturbation** (70% of augmented samples)
- Factors: [0.9, 1.0, 1.1] (slow, normal, fast)
- Uses torchaudio sox_effects for time-stretching

#### B. **Pitch Shifting** (30% of augmented samples)
- Range: +/- 2 semitones
- Uses torchaudio PitchShift transform

#### C. **Background Noise Injection** (40% of augmented samples)
- Gaussian noise with SNR 15-30 dB
- Simulates real-world recording conditions

**Code:**
```python
def apply_audio_augmentation(self, audio_array: np.ndarray) -> np.ndarray:
    if not self.config.use_audio_augmentation:
        return audio_array

    if random.random() > self.config.audio_augmentation_prob:  # 50%
        return audio_array

    audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)

    # Speed perturbation (70%)
    if random.random() < 0.7:
        speed_factor = random.choice(self.config.speed_perturb_factors)
        # ... sox_effects apply

    # Pitch shift (30%)
    if random.random() < 0.3:
        n_steps = random.randint(-self.config.pitch_shift_steps, self.config.pitch_shift_steps)
        # ... PitchShift apply

    # Noise injection (40%)
    if random.random() < 0.4:
        snr_db = random.uniform(*self.config.noise_snr_db)
        # ... add Gaussian noise

    return audio_tensor.squeeze(0).numpy()
```

**Expected impact:** 5-10% WER reduction

---

### 4. ✅ **Model Averaging** (3 best checkpoints)
**File:** `train_hebrew_asr_enhanced.py` - `average_model_checkpoints()`

**What:**
- After training, find 3 best checkpoints by eval loss
- Average their weights (ensemble-like benefits)
- Save as `-averaged` model variant

**Code:**
```python
def average_model_checkpoints(checkpoint_paths: List[str], output_path: str, base_model):
    # Load all state dicts
    state_dicts = [torch.load(cp) for cp in checkpoint_paths]

    # Average weights
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

    # Save
    torch.save(avg_state, Path(output_path) / "pytorch_model.bin")
```

**Expected impact:** 2-5% WER reduction

---

### 5. ✅ **Balanced Dataset Sampling**
**File:** `train_hebrew_asr_enhanced.py` - `load_datasets()`

**What:**
Use `interleave_datasets()` instead of `concatenate_datasets()` with **balanced** sampling:

#### **Balanced Strategy (NOT 90% Knesset):**
```python
dataset_sampling_probs = {
    "ivrit-ai/knesset-plenums-whisper-training": 0.50,  # 50% formal
    "ivrit-ai/crowd-transcribe-v5": 0.30,               # 30% crowd
    "ivrit-ai/crowd-recital-whisper-training": 0.20,    # 20% Wikipedia
}
```

**Why balanced?**
- ivrit.ai uses 90% Knesset because they optimize for formal speech
- Our eval sets are more informal (WhatsApp, crowd-sourced)
- Balanced mix generalizes better to diverse domains

**Code:**
```python
# Interleave datasets with probabilities
combined = interleave_datasets(
    list(datasets_dict.values()),
    probabilities=sampling_probs,
    seed=42,
    stopping_strategy="all_exhausted"
)
```

**Expected impact:** 5-10% WER reduction (domain-specific)

---

## Configuration Changes

### New Config Fields:

```python
# Synthetic audio augmentation
use_audio_augmentation: bool = True
speed_perturb_factors: List[float] = [0.9, 1.0, 1.1]
pitch_shift_steps: int = 2
noise_snr_db: Tuple[float, float] = (15.0, 30.0)
audio_augmentation_prob: float = 0.5  # 50%

# ivrit.ai SOTA methods
timestamp_keep_prob: float = 0.4  # 40%
prev_context_prob: float = 0.5    # 50%

# Balanced dataset sampling
dataset_sampling_probs: Dict[str, float] = {
    "ivrit-ai/knesset-plenums-whisper-training": 0.50,
    "ivrit-ai/crowd-transcribe-v5": 0.30,
    "ivrit-ai/crowd-recital-whisper-training": 0.20,
}

# Datasets (added Knesset)
datasets: List[str] = [
    "ivrit-ai/knesset-plenums-whisper-training",  # NEW!
    "ivrit-ai/crowd-transcribe-v5",
    "ivrit-ai/crowd-recital-whisper-training"
]
```

---

## Training Data Summary

| Dataset | Hours | Samples | Sampling | Domain |
|---------|-------|---------|----------|--------|
| Knesset | ~4,700h | ~100k-1M | **50%** | Formal parliamentary |
| Transcribe v5 | ~300h | ~186k | **30%** | Crowd informal |
| Recital | ~50h | ~40k | **20%** | Wikipedia recitals |
| **TOTAL** | **~5,050h** | **~326k-1.2M** | | **Balanced** |

**vs ivrit.ai:** Same data, but 50-30-20 (balanced) instead of 90-2.5-7.5 (Knesset-heavy)

---

## Expected Total Impact

| Optimization | Expected WER Reduction |
|-------------|------------------------|
| **Round 2 (baseline):** | |
| SpecAugment | 5-10% |
| Cosine LR Schedule | 2-5% |
| Fixed batch size | 5-10% |
| Layer-wise LR | 3-7% |
| **Round 2 subtotal** | **15-30%** → 12.3% → **8.6-10.5%** |
| | |
| **Round 2.5 (added):** | |
| Timestamp preservation | 3-7% |
| Previous context | 2-4% |
| Synthetic augmentation | 5-10% |
| Model averaging | 2-5% |
| Balanced sampling | 5-10% |
| **Round 2.5 subtotal** | **17-36%** → 8.6% → **5.5-7.1%** |
| | |
| **COMBINED TOTAL** | **32-66%** → 12.3% → **4.2-8.4%** |

**Target WER:** 5.5-7.0% (competitive with SOTA 5.1%)

**Realistic expectation:** 6.0-7.5% WER (depends on Knesset data quality, training stability)

---

## W&B Tracking

New metrics logged:
```python
"audio_augmentation_enabled": True,
"speed_perturb_factors": [0.9, 1.0, 1.1],
"pitch_shift_steps": 2,
"noise_snr_db": (15.0, 30.0),
"audio_augmentation_prob": 0.5,
"timestamp_keep_prob": 0.4,
"prev_context_prob": 0.5,
"model_averaging": True,
"dataset_sampling_strategy": "balanced",
"dataset_sampling_probs": {50%, 30%, 20%},
```

New tags:
- `round2.5`
- `ivrit-ai-methods`
- `synthetic-aug`
- `timestamp-preservation`
- `model-averaging`
- `balanced-sampling`

---

## Training Command

```bash
# Set environment variables
export HF_REPO_ID="OzLabs/Qwen3-ASR-Hebrew-Round2.5"
export WANDB_PROJECT="qwen3-asr-hebrew"
export WANDB_RUN_NAME="round2.5-balanced-$(date +%Y%m%d)"

# Launch training (8x A100)
uv run torchrun --nproc_per_node=8 train_round2_gradual.py

# Or 2x A100
uv run torchrun --nproc_per_node=2 train_round2_gradual.py
```

**Expected runtime:**
- **With Knesset (5,050h):** 8-12 hours on 8x A100, 30-40 hours on 2x A100
- **Without Knesset (350h):** 2-3 hours on 8x A100, 8-10 hours on 2x A100

**Recommendation:** Start without Knesset to validate methods work, then add Knesset for full training.

---

## Testing Without Knesset First (Quick Validation)

To test the implementation without downloading 4,700h of Knesset data:

```python
# In TrainingConfig.__post_init__()
if self.datasets is None:
    self.datasets = [
        # "ivrit-ai/knesset-plenums-whisper-training",  # Comment out for testing
        "ivrit-ai/crowd-transcribe-v5",
        "ivrit-ai/crowd-recital-whisper-training"
    ]

if self.dataset_sampling_probs is None:
    self.dataset_sampling_probs = {
        # "ivrit-ai/knesset-plenums-whisper-training": 0.50,
        "ivrit-ai/crowd-transcribe-v5": 0.60,           # 60% (boosted)
        "ivrit-ai/crowd-recital-whisper-training": 0.40, # 40% (boosted)
    }
```

**Expected WER (without Knesset):** 7.5-9.0% (still better than Round 1's 12.3%)

---

## File Changes Summary

**Modified:** `train_hebrew_asr_enhanced.py`

1. **Imports:** Added `torchaudio`, `interleave_datasets`, `random`
2. **TrainingConfig:** Added 10+ new fields for augmentation, timestamps, sampling
3. **HebrewTextNormalizer:** Conditional timestamp preservation
4. **AudioDataPreprocessor:**
   - `apply_audio_augmentation()` - speed, pitch, noise
   - `preprocess_function()` - previous context injection
5. **DataLoader:** `load_datasets()` with interleaving + sampling
6. **Post-training:** `average_model_checkpoints()` function
7. **Logging:** Enhanced W&B tracking, updated print statements
8. **Config defaults:** Balanced sampling, Knesset added

**Created:**
- `ROUND2.5_IMPLEMENTATION.md` (this document)
- `IVRIT_AI_DATASET_INTEGRATION.md` (detailed dataset guide)
- `BEATING_SOTA_ANALYSIS.md` (gap analysis + roadmap)

---

## Next Steps

### **Option A: Quick Validation (2-3 hours)**
1. Test without Knesset (350h data)
2. Verify all methods work (timestamps, context, augmentation, averaging)
3. Benchmark on eval-d1
4. Expected: 7.5-9.0% WER

### **Option B: Full Training (8-12 hours)**
1. Include Knesset data (5,050h total)
2. Full ivrit.ai SOTA replication with balanced sampling
3. Benchmark on all 6 leaderboard datasets
4. Expected: 6.0-7.5% WER

### **Option C: Iterative (Recommended)**
1. Start with Option A (quick validation)
2. If methods work, proceed to Option B (full training)
3. Compare at each step

---

## Troubleshooting

### **If audio augmentation fails:**
- Check torchaudio installation: `uv add torchaudio`
- Check sox_effects availability (for speed perturbation)
- Fallback: disable with `use_audio_augmentation=False`

### **If Knesset dataset fails to load:**
- Check HuggingFace access (may need authentication)
- Check storage (150GB compressed, ~300GB uncompressed)
- Fallback: train without Knesset first

### **If model averaging fails:**
- Check checkpoint format (pytorch_model.bin vs safetensors)
- Check at least 3 checkpoints exist
- Logs will warn if <3 checkpoints, will skip averaging

---

## Success Criteria

**Round 2.5 is successful if:**
1. ✅ Training completes without errors
2. ✅ All augmentation methods apply (check logs)
3. ✅ Model averaging produces `-averaged` variant
4. ✅ WER < 9.0% on eval-d1 (without Knesset)
5. ✅ WER < 7.5% on eval-d1 (with Knesset)
6. ✅ No degradation vs Round 1 (12.3% baseline)

**Beat SOTA if:**
- WER < 5.1% on eval-d1 (current ivrit.ai SOTA)
- Competitive across all 6 test sets (WhatsApp, CommonVoice, etc.)

---

## Monitoring During Training

**Watch for:**
1. **Dataset loading:** Should show balanced sampling (50-30-20)
2. **Augmentation logs:** Should see "Applying synthetic augmentation" messages
3. **Loss convergence:** Should break through 11.3 plateau from Round 1
4. **WER metrics:** Should work now (no 100.0% errors)
5. **Checkpoint count:** Need 3+ for model averaging

**W&B dashboard:** https://wandb.ai/OzLabs/qwen3-asr-hebrew

---

## Summary

**Round 2.5 implements 5 major ivrit.ai SOTA methods:**
1. ✅ Timestamp preservation (40%)
2. ✅ Previous context (50%)
3. ✅ Synthetic audio augmentation (speed, pitch, noise)
4. ✅ Model averaging (3 best checkpoints)
5. ✅ Balanced dataset sampling (50-30-20)

**Plus all Round 2 optimizations:**
- SpecAugment
- Cosine LR schedule
- Fixed batch size
- Layer-wise discriminative LR
- BF16-safe evaluation
- Auto HF upload

**Expected result:** 6.0-7.5% WER (vs SOTA 5.1%, vs Round 1 12.3%)

**Status:** ✅ **Ready to train!**
