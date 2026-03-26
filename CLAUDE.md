# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tunes Qwen3-ASR-1.7B for Hebrew ASR using LoRA. Trains on ivrit.ai Hebrew datasets, deploys to GPU instances (Lambda Labs H100, HuggingFace Jobs). Published model: `OzLabs/Qwen3-ASR-Hebrew-1.7B`.

## Common Commands

```bash
# Install dependencies
uv sync

# Prepare training data (creates JSONL + WAV files in ./qwen3_asr_data/)
uv run python training/prepare_qwen_data.py

# Run local training (requires GPU with 16GB+ VRAM)
uv run python training/train_hebrew_asr_enhanced.py

# Launch cloud training on HuggingFace Jobs
uv run python training/launch_training.py
uv run python training/launch_training.py --monitor

# Monitor HF Jobs
hf jobs ps
hf jobs logs qwen3-asr-hebrew-training --follow

# Run FastAPI inference server (from qwen3-asr-hebrew-model/)
cd qwen3-asr-hebrew-model && uv run fastapi serve_asr.py --host 0.0.0.0 --port 8000

# Evaluate on all 6 ivrit.ai benchmark test sets
uv run python scripts/eval/evaluate_ivrit_benchmarks.py --model ./qwen3-asr-hebrew --output results.json

# Compare two model rounds
uv run python scripts/eval/evaluate_ivrit_benchmarks.py \
    --model ./qwen3-asr-hebrew-round1 --model ./qwen3-asr-hebrew-round2.5 \
    --compare --output comparison.json

# Round 2 Training Commands
source scripts/infra/setup_wandb.sh                    # Set up experiment tracking
uv run python scripts/data/phase0_align_audit.py       # Phase 0 data quality audit
uv run python training/train_round2_gradual.py         # Round 2 gradual unfreezing
uv run python scripts/eval/eval_round2.py \            # Round 2 vs Round 1 comparison
    --round1-model OzLabs/Qwen3-ASR-Hebrew-1.7B \
    --round2-model ./qwen3-asr-hebrew-round2
```

## Experiment Tracking

Uses Weights & Biases (`wandb`). Setup: `wandb login` then `source scripts/infra/setup_wandb.sh`.

Env vars: `WANDB_PROJECT="qwen3-asr-hebrew"`, `WANDB_RUN_NAME="experiment-name"`, `WANDB_PHASE0_LOGGING="true"` (for Phase 0 audit).

Tracks: loss, WER, CER, GPU metrics, LR schedules, strategy switches (B→A at epoch 3), checkpoints.

## Architecture

### Repo Layout

```
training/           — All training & data prep scripts
scripts/eval/       — Evaluation & benchmarks
scripts/data/       — Data preparation utilities
scripts/infra/      — Infrastructure & setup (Lambda, W&B, cache)
docs/               — Reference documentation
qwen3-asr-hebrew-model/  — Self-contained deployment package
_archive/           — Deprecated scripts (for reference)
```

### Training Pipeline

1. **Data Preparation** (`training/prepare_qwen_data.py`): Downloads ivrit-ai datasets, extracts audio bytes via PyArrow (bypasses torchcodec issues), normalizes Hebrew text (removes niqqud/vowel marks), outputs JSONL + WAV in Qwen3-ASR format.

2. **Training** (`training/train_hebrew_asr_enhanced.py`): Primary training script. LoRA (rank=16, alpha=32) targeting attention + FFN layers. Effective batch size = batch_size × grad_accumulation × num_gpus. Config in `config.yaml`.

3. **Round 2 Gradual Unfreezing** (`training/train_round2_gradual.py`): Imports from `train_hebrew_asr_enhanced.py`. Epochs 1-2 use Strategy B (projector + top 12 LLM layers), epochs 3-5 switch to Strategy A (+ top 8 audio layers).

4. **Cloud Training** (`training/launch_training.py`): Submits to HuggingFace Jobs with A100 flavor.

5. **Phase 0 Quality Gate** (`scripts/data/phase0_align_audit.py`): Samples 10% of training data stratified by domain, runs forced alignment. If >15% have coverage <0.6 → stop and filter first.

### Model Serving (`qwen3-asr-hebrew-model/`)

Self-contained deployment package with its own `pyproject.toml`:
- `serve_asr.py`: vLLM-backed FastAPI server, OpenAI-compatible API at `/v1/audio/transcriptions`
- `src/qwen_asr/audio.py`: AudioProcessor handles dict/bytes/torchcodec formats, `normalize_hebrew_text()` strips niqqud
- `src/qwen_asr/client.py`: VLLMClient wrapper for transcription requests
- `src/qwen_asr/datasets.py`: HEBREW_DATASETS config for 6 eval sets
- `src/benchmarks/evaluate.py`: Benchmark runner outputting WER/CER per dataset to CSV
- `frontend/`: Web UI for testing transcription

### Key Configuration

- `config.yaml`: LoRA params, training hyperparameters, dataset sources, hardware specs
- Audio constraints: 16kHz mono WAV, 0.5-30s duration
- Text normalization: Removes niqqud (U+0591-U+05C7), Whisper timestamps, duplicate punctuation
- Python >=3.11,<3.14 (see `pyproject.toml`)

## Critical Technical Details

**Qwen3-ASR Data Format**: Standard HuggingFace Audio feature fails due to custom tokenizer. Must use raw PyArrow + librosa to extract audio bytes, save as WAV files, and create JSONL with absolute paths.

**Memory Optimization**: For OOM issues, halve batch_size and double gradient_accumulation to maintain effective batch size. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Platform Requirements**: Audio + GPU training works on native Linux (Lambda Labs). Mac M-series and HF Jobs containers have torchcodec/FFmpeg limitations.

**Evaluation Benchmark**: The 6 ivrit.ai test sets match the official Hebrew transcription leaderboard at `huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard`. Primary metric is WER (Word Error Rate).
