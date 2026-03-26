# Getting Started with Hebrew ASR Fine-tuning

## Quick Start

You're all set up! Here's what to do next:

### Step 1: Verify Setup

```bash
uv run python test_setup.py
```

Expected results:
- ✓ All packages installed
- ✓ HF authentication working
- ✗ No local GPU (this is fine - we'll use cloud)
- ⚠ Model/dataset access (may need approval)

### Step 2: Launch Training

Since you don't have a local GPU, use Hugging Face Jobs:

```bash
# Simple launch
uv run python launch_training.py

# Launch and monitor logs
uv run python launch_training.py --monitor
```

This will:
1. Submit your training job to HF Jobs infrastructure
2. Use an A100 GPU (or A10G for lower cost)
3. Train for ~6-8 hours
4. Save the model to your HF account

### Step 3: Monitor Progress

```bash
# Check job status
hf jobs ps

# View logs
hf jobs logs qwen3-asr-hebrew-training --follow

# Cancel if needed
hf jobs cancel qwen3-asr-hebrew-training
```

## What We've Built

### Project Structure

```
caspi/
├── train_hebrew_asr.py          # Main training script (LoRA-based)
├── train_with_qwen_asr.py       # Alternative using qwen-asr package
├── launch_training.py            # HF Jobs launcher
├── test_setup.py                 # Environment verification
├── config.yaml                   # Training configuration
├── pyproject.toml               # Dependencies
├── README.md                     # Full documentation
├── training-suggestion.md        # Advanced training strategies
└── GETTING_STARTED.md           # This file
```

### What's Configured

1. **Model**: Qwen3-ASR-1.7B (1.7B parameters)
2. **Datasets**:
   - ivrit-ai/crowd-transcribe-v5
   - ivrit-ai/crowd-recital-whisper-training
3. **Training Method**: LoRA (Low-Rank Adaptation)
   - Only trains ~1.5% of parameters
   - Reduces GPU memory from 40GB → 12GB
   - Faster training, similar quality
4. **Hardware**: Cloud GPU (A100 40GB recommended)
5. **Cost**: ~$9-12 per training run

## Training Approaches

### Option A: Standard LoRA Fine-tuning (Recommended)

This is what `train_hebrew_asr.py` implements:

```bash
uv run python launch_training.py
```

**Pros**:
- Proven approach
- Lower GPU memory requirements
- Faster iterations
- Easy to merge back to full model

**Configuration**:
- LoRA rank: 16
- Learning rate: 2e-4
- Batch size: 8 (effective: 32 with gradient accumulation)
- Epochs: 3

### Option B: Using Qwen3-ASR Training Framework

For the most official/supported path:

```bash
# Clone official repo
git clone https://github.com/QwenLM/Qwen3-ASR
cd Qwen3-ASR

# Check finetuning/ directory for official scripts
ls finetuning/
```

Use this if you want the exact training setup used by the Qwen team.

## Important Decisions to Make

### 1. Audio Length Filtering Strategy

**Current Status**: ⚠️ Needs your input

The training script has placeholder for audio filtering. See `training-suggestion.md` for options:

- **Hard filtering**: Simple, but loses ~20-40% of data
- **Chunking with overlap** (recommended): Keeps all data, more complex
- **Dynamic bucketing**: Most memory-efficient, most complex

**Action Required**:
Decide on a strategy and implement in `train_hebrew_asr.py:103-107`

### 2. Output Format Normalization

From `training-suggestion.md`:

- Numbers: digits vs spelled out?
- Punctuation: heavy or minimal?
- Hebrew-specific: niqqud/diacritics?
- Mixed Hebrew/English handling?

**Recommendation**: Start with "clean Hebrew text" (no niqqud, minimal punctuation)

### 3. GPU Selection

| GPU Type | Memory | Speed | Cost/hour | Total Cost (6h) |
|----------|--------|-------|-----------|-----------------|
| A10G     | 24GB   | 1.0x  | $1.50     | ~$9             |
| A100     | 40GB   | 1.5x  | $2.50     | ~$12.50         |

**Recommendation**: Start with A100 for faster iteration, switch to A10G once everything works

## Troubleshooting

### Dataset Access Denied

If you get permission errors:

1. Check you've accepted licenses:
   - [ivrit-ai/crowd-transcribe-v5](https://huggingface.co/datasets/ivrit-ai/crowd-transcribe-v5)
   - [ivrit-ai/crowd-recital-whisper-training](https://huggingface.co/datasets/ivrit-ai/crowd-recital-whisper-training)

2. Verify authentication:
   ```bash
   huggingface-cli whoami
   ```

3. Re-login if needed:
   ```bash
   huggingface-cli login
   ```

### Model Architecture Not Found

The error about `qwen3_asr` architecture is expected. Solutions:

1. **Recommended**: Use HF Jobs (the job environment has everything pre-configured)
2. **Alternative**: Install qwen-asr package:
   ```bash
   uv add qwen-asr
   ```

### Out of Memory During Training

1. Reduce batch size in `config.yaml`:
   ```yaml
   training:
     batch_size: 4  # was 8
     gradient_accumulation_steps: 8  # was 4
   ```

2. Use A100 instead of A10G

3. Enable gradient checkpointing (already enabled)

### Training Fails to Start

Check logs:
```bash
hf jobs logs qwen3-asr-hebrew-training
```

Common issues:
- Dataset access not granted yet
- HF token not set correctly
- Syntax error in training script

## Next Steps After Training

### 1. Evaluate the Model

```python
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "./qwen3-asr-hebrew",  # or your HF repo
    dtype=torch.bfloat16,
)

results = model.transcribe("test_hebrew_audio.wav")
print(results[0].text)
```

### 2. Calculate WER (Word Error Rate)

```python
import evaluate

wer_metric = evaluate.load("wer")

# Compare predictions to ground truth
predictions = ["predicted text"]
references = ["actual text"]

wer = wer_metric.compute(predictions=predictions, references=references)
print(f"WER: {wer * 100:.2f}%")
```

### 3. Deploy

Options:
- Hugging Face Inference API
- Local server with qwen-asr
- vLLM for high-throughput serving

### 4. Iterate

Based on results:
1. Adjust hyperparameters in `config.yaml`
2. Add more data augmentation
3. Experiment with different LoRA ranks
4. Try full fine-tuning if results plateau

## Cost Breakdown

**Per Training Run**:
- A100 (6 hours): ~$12.50
- A10G (8 hours): ~$12.00

**Recommended Budget for Initial Experiments**:
- 3-5 training runs: $40-60
- This allows testing different configurations

**Cost Optimization**:
1. Start with smaller dataset subset
2. Use A10G after confirming setup works
3. Use gradient checkpointing (already enabled)
4. Share GPU across experiments

## Getting Help

### Documentation
- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Hugging Face Jobs Guide](https://huggingface.co/docs/hub/jobs)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [ivrit.ai](https://www.ivrit.ai/)

### Quick Questions
Check `README.md` for detailed troubleshooting and usage examples.

### Issues
- Model issues: [Qwen3-ASR Issues](https://github.com/QwenLM/Qwen3-ASR/issues)
- Dataset issues: Contact ivrit.ai
- Training issues: Check HF Jobs logs first

## Summary

You're ready to start training! The recommended path is:

1. **Run**: `uv run python launch_training.py`
2. **Monitor**: `hf jobs logs qwen3-asr-hebrew-training --follow`
3. **Wait**: ~6-8 hours
4. **Evaluate**: Check WER on validation set
5. **Iterate**: Adjust and re-run if needed

The training infrastructure is set up to push your model to Hugging Face Hub automatically, so you can use it immediately after training completes.

**Ready to launch?** 🚀

```bash
uv run python launch_training.py
```
