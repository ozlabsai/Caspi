# Evaluation Guide: Benchmarking Against ivrit.ai SOTA

**Purpose:** Evaluate your trained models on all 6 ivrit.ai test sets to compare against SOTA (5.1% WER on eval-d1)

---

## Overview

The ivrit.ai leaderboard uses 6 test sets covering different domains:

| Test Set | Description | SOTA WER | Model |
|----------|-------------|----------|-------|
| **eval-d1** | Clean, high-quality recordings | **5.1%** | ivrit-ai/whisper-large-v3-ct2-20250513 |
| **whatsapp** | Noisy, real-world voice messages | **7.2%** | ivrit-ai/whisper-large-v3-ct2-20250513 |
| **saspeech** | Spontaneous Hebrew speech | **6.4%** | ivrit-ai/whisper-large-v3-ct2-20250513 |
| **fleurs** | Formal reading (multilingual) | **17.4%** | ivrit-ai/whisper-large-v3-ct2-20250513 |
| **commonvoice** | Diverse accents & conditions | **14.9%** | ivrit-ai/whisper-large-v3-ct2-20250513 |
| **kan** | Israeli public broadcasting | **8.1%** | ivrit-ai/whisper-large-v3-ct2-20250513 |

**Source:** https://huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard

---

## Quick Evaluation (Development)

Use `quick_eval.py` for rapid iteration during development:

```bash
# Evaluate on eval-d1 (default) with 100 samples
uv run python scripts/quick_eval.py \
    --model ./qwen3-asr-hebrew-round2.5

# Evaluate on WhatsApp test set
uv run python scripts/quick_eval.py \
    --model ./qwen3-asr-hebrew-round2.5 \
    --test-set whatsapp \
    --max-samples 50

# Compare Round 1 vs Round 2.5
uv run python scripts/quick_eval.py \
    --model ./qwen3-asr-hebrew-round1 \
    --model ./qwen3-asr-hebrew-round2.5 \
    --test-set eval-d1 \
    --max-samples 100
```

**Output:**
```
QUICK EVALUATION
======================================================================
Test set: eval-d1
Max samples: 100
Models: 2
======================================================================

Loading model: ./qwen3-asr-hebrew-round1
✓ Model loaded on cuda
  Total params: 1,993,456,128

======================================================================
Evaluating: eval-d1
======================================================================
Dataset: ivrit-ai/eval-d1
Description: Clean, high-quality recordings
Samples: 100

  Results:
    WER: 12.35%
    CER: 6.21%
    vs SOTA: +7.2% (SOTA: 5.1%)
    Valid samples: 98/100

Loading model: ./qwen3-asr-hebrew-round2.5
✓ Model loaded on cuda
  Total params: 1,993,456,128

======================================================================
Evaluating: eval-d1
======================================================================
...
  Results:
    WER: 7.15%
    CER: 3.42%
    vs SOTA: +2.1% (SOTA: 5.1%)
    Valid samples: 100/100

======================================================================
COMPARISON
======================================================================

qwen3-asr-hebrew-round1              : 12.35% WER (vs SOTA: +7.2%)
qwen3-asr-hebrew-round2.5            :  7.15% WER (vs SOTA: +2.1%)

✓ Best: qwen3-asr-hebrew-round2.5
```

---

## Full Benchmark (Official Evaluation)

Use `evaluate_ivrit_benchmarks.py` for complete evaluation on all 6 test sets:

```bash
# Evaluate single model on all 6 test sets
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model ./qwen3-asr-hebrew-round2.5 \
    --output results_round2.5.json

# Compare multiple models
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model OzLabs/Qwen3-ASR-Hebrew-1.7B \
    --model ./qwen3-asr-hebrew-round2.5 \
    --model ./qwen3-asr-hebrew-round2.5-averaged \
    --compare \
    --output comparison.json

# Quick test with limited samples (fast)
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model ./qwen3-asr-hebrew-round2.5 \
    --max-samples 50 \
    --output quick_test.json
```

**Output:**
```
======================================================================
SUMMARY
======================================================================
Model: ./qwen3-asr-hebrew-round2.5

Per-dataset results:
  eval-d1        :   7.15% WER (vs SOTA: +2.0%)
  whatsapp       :   9.32% WER (vs SOTA: +2.1%)
  saspeech       :   8.10% WER (vs SOTA: +1.7%)
  fleurs         :  19.45% WER (vs SOTA: +2.1%)
  commonvoice    :  16.23% WER (vs SOTA: +1.3%)
  kan            :   9.87% WER (vs SOTA: +1.8%)

  Average WER: 11.69%
  SOTA Average: 9.80%
======================================================================

✓ Results saved to: results_round2.5.json
```

**Output JSON format:**
```json
{
  "models": [
    {
      "model_path": "./qwen3-asr-hebrew-round2.5",
      "results": {
        "eval-d1": {
          "wer": 0.0715,
          "cer": 0.0342,
          "samples_total": 150,
          "samples_valid": 150,
          "sota_wer": 0.051,
          "vs_sota": "+2.0%"
        },
        "whatsapp": { ... },
        ...
      },
      "average_wer": 0.1169,
      "average_cer": 0.0589
    }
  ],
  "sota_baseline": {
    "eval-d1": 0.051,
    "whatsapp": 0.072,
    ...
  }
}
```

---

## Evaluation Workflow

### **After Training (Recommended):**

```bash
# 1. Quick sanity check (5 minutes)
uv run python scripts/quick_eval.py \
    --model ./qwen3-asr-hebrew-round2.5 \
    --max-samples 50

# 2. Full benchmark (30-60 minutes on GPU, 2-4 hours on CPU)
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model ./qwen3-asr-hebrew-round2.5 \
    --model ./qwen3-asr-hebrew-round2.5-averaged \
    --compare \
    --output round2.5_benchmark.json

# 3. Compare with baseline
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model OzLabs/Qwen3-ASR-Hebrew-1.7B \
    --model ./qwen3-asr-hebrew-round2.5-averaged \
    --compare \
    --output round1_vs_round2.5.json
```

---

## Command-Line Options

### **evaluate_ivrit_benchmarks.py**

```bash
--model PATH             Model path or HF ID (can use multiple times)
--output FILE            Output JSON file (default: evaluation_results.json)
--batch-size N           Batch size for inference (default: 8)
--max-samples N          Max samples per test set (for quick testing)
--device {auto,cuda,cpu} Device to run on (default: auto)
--compare                Show comparison table (requires multiple --model)
```

### **quick_eval.py**

```bash
--model PATH             Model path (can use multiple times)
--test-set NAME          Test set: eval-d1, whatsapp, saspeech, fleurs, commonvoice, kan
--max-samples N          Max samples (default: 100)
--batch-size N           Batch size (default: 8)
```

---

## Performance Tips

### **GPU Evaluation (Recommended):**
- **8x A100:** ~30-45 minutes for all 6 test sets
- **1x A100:** ~1-2 hours for all 6 test sets
- Use `--batch-size 16` or `--batch-size 32` for faster evaluation

### **CPU Evaluation:**
- **M1 Mac (16GB):** ~3-6 hours for all 6 test sets
- **CPU Server:** ~2-4 hours for all 6 test sets
- Use `--batch-size 4` to avoid OOM
- Consider `--max-samples 100` for quicker results

### **Quick Testing:**
```bash
# Test on 50 samples per set (5-10 minutes on GPU)
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model ./qwen3-asr-hebrew-round2.5 \
    --max-samples 50
```

---

## Interpreting Results

### **WER (Word Error Rate):**
- **< 5%:** Excellent (SOTA level)
- **5-8%:** Very good (competitive)
- **8-12%:** Good (usable)
- **> 12%:** Needs improvement

### **vs SOTA Metric:**
- **Negative (e.g., -1.5%):** Better than SOTA ✅
- **+0 to +2%:** Very close to SOTA 👍
- **+2 to +5%:** Competitive 👌
- **> +5%:** Behind SOTA, needs work

### **Domain Analysis:**

**If model performs well on:**
- **eval-d1 + kan:** Good at clean, formal speech (Knesset training helped)
- **whatsapp + commonvoice:** Good at noisy, informal speech (crowd data helped)
- **saspeech:** Good at spontaneous speech (balanced training helped)
- **fleurs:** Good at formal reading (may need more formal data)

**If model struggles on:**
- **whatsapp:** Add more noise augmentation
- **commonvoice:** Add more diverse accents
- **fleurs:** Add more formal reading data
- **saspeech:** Add more spontaneous speech data

---

## Submitting to ivrit.ai Leaderboard

Once you're happy with your results, you can submit to the official leaderboard:

**Prerequisites:**
1. Model must be publicly accessible on HuggingFace Hub
2. Must evaluate on all 6 test sets
3. Must report WER for each test set

**Steps:**
1. Upload model to HuggingFace Hub (automatic after training with `HF_REPO_ID` set)
2. Run full evaluation:
   ```bash
   uv run python scripts/evaluate_ivrit_benchmarks.py \
       --model OzLabs/Qwen3-ASR-Hebrew-Round2.5 \
       --output leaderboard_submission.json
   ```
3. Follow submission instructions at: https://huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard

---

## Example: Complete Evaluation Pipeline

```bash
#!/bin/bash
# evaluate_after_training.sh

MODEL_PATH="./qwen3-asr-hebrew-round2.5-averaged"
OUTPUT_DIR="./evaluation_results"

mkdir -p $OUTPUT_DIR

echo "Starting evaluation pipeline..."

# 1. Quick sanity check
echo "Step 1: Quick sanity check (50 samples on eval-d1)..."
uv run python scripts/quick_eval.py \
    --model $MODEL_PATH \
    --max-samples 50

# 2. Full benchmark
echo "Step 2: Full benchmark on all 6 test sets..."
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model $MODEL_PATH \
    --output $OUTPUT_DIR/full_benchmark.json

# 3. Compare with baseline
echo "Step 3: Comparing with Round 1 baseline..."
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model OzLabs/Qwen3-ASR-Hebrew-1.7B \
    --model $MODEL_PATH \
    --compare \
    --output $OUTPUT_DIR/comparison.json

echo "✓ Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/"
```

---

## Troubleshooting

### **Issue: Dataset download fails**
```bash
# Pre-download datasets manually
python3 -c "
from datasets import load_dataset
load_dataset('ivrit-ai/eval-d1', split='test')
load_dataset('ivrit-ai/whatsapp-test', split='test')
# ... etc
"
```

### **Issue: OOM during evaluation**
```bash
# Reduce batch size
--batch-size 1

# Or evaluate on CPU
--device cpu
```

### **Issue: Model generates empty transcriptions**
- Check model was trained correctly (not just random init)
- Check BF16/FP32 dtype compatibility
- Try `--device cpu` to rule out GPU issues

### **Issue: Very slow on CPU**
- Use `--max-samples 50` for quick results
- Or rent a small GPU instance (1x T4 ~$0.30/hr)

---

## Next Steps After Evaluation

**If WER < 7% on eval-d1:**
- ✅ You're competitive with SOTA!
- Submit to ivrit.ai leaderboard
- Deploy for production use

**If WER 7-10% on eval-d1:**
- 👍 Good results, but room for improvement
- Try Round 3 optimizations:
  - Larger LoRA rank (or keep full fine-tuning)
  - More training epochs
  - Better data augmentation
  - Hyperparameter tuning

**If WER > 10% on eval-d1:**
- 🤔 Something might be wrong
- Check training logs for issues
- Verify data quality
- Try smaller learning rate
- Check if model converged

---

## Summary

**Quick evaluation:**
```bash
uv run python scripts/quick_eval.py --model ./model --max-samples 100
```

**Full benchmark:**
```bash
uv run python scripts/evaluate_ivrit_benchmarks.py --model ./model
```

**Compare models:**
```bash
uv run python scripts/evaluate_ivrit_benchmarks.py \
    --model ./model1 --model ./model2 --compare
```

**Target:** Beat SOTA (5.1% WER on eval-d1) or get within 2% (< 7.1% WER)
