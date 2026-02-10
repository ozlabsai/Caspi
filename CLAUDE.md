# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project fine-tunes Qwen3-ASR-1.7B for Hebrew automatic speech recognition (ASR) using LoRA. It trains on ivrit.ai Hebrew datasets and can deploy to GPU instances (Lambda Labs H100, HuggingFace Jobs).

## Common Commands

```bash
# Install dependencies
uv sync

# Prepare training data (creates JSONL + WAV files in ./qwen3_asr_data/)
uv run python prepare_qwen_data.py

# Run local training (requires GPU with 16GB+ VRAM)
uv run python train_hebrew_asr.py

# Launch cloud training on HuggingFace Jobs
uv run python launch_training.py
uv run python launch_training.py --monitor  # with log monitoring

# Monitor HF Jobs
hf jobs ps
hf jobs logs qwen3-asr-hebrew-training --follow

# Run FastAPI inference server (from qwen3-asr-hebrew-model/)
uv run fastapi serve_asr.py --host 0.0.0.0 --port 8000

# Run benchmarks against vLLM server
uv run python scripts/benchmark.py --server http://localhost:8000/v1 --max-samples 100
```

## Architecture

### Training Pipeline

1. **Data Preparation** (`prepare_qwen_data.py`): Downloads ivrit-ai datasets, extracts audio bytes via PyArrow (bypasses torchcodec issues), normalizes Hebrew text (removes niqqud/vowel marks), outputs JSONL + WAV files in Qwen3-ASR format: `{"audio": "path.wav", "text": "language Hebrew<asr_text>TRANSCRIPTION"}`

2. **Training** (`train_hebrew_asr.py`): Uses LoRA (rank=16, alpha=32) targeting attention + FFN layers. Effective batch size = 256 (batch_size × grad_accumulation × num_gpus).

3. **Cloud Training** (`launch_training.py`): Submits to HuggingFace Jobs with A100 flavor.

### Model Serving

The `qwen3-asr-hebrew-model/` subdirectory contains:
- `serve_asr.py`: FastAPI server with `/transcribe` (base64) and `/transcribe/file` (upload) endpoints
- `src/qwen_asr/`: Audio processing, vLLM client, dataset configs
- `src/benchmarks/`: WER evaluation on eval-d1, eval-whatsapp, Common Voice Hebrew

### Key Configuration

- `config.yaml`: LoRA params, training hyperparameters, dataset sources
- Audio: 16kHz mono WAV, 0.5-30s duration
- Text normalization: Removes niqqud (U+0591-U+05C7), Whisper timestamps, duplicate punctuation

## Critical Technical Details

**Qwen3-ASR Data Format**: Standard HuggingFace Audio feature fails due to custom tokenizer. Must use raw PyArrow + librosa to extract audio bytes, save as WAV files, and create JSONL with absolute paths.

**Memory Optimization**: For OOM issues, halve batch_size and double gradient_accumulation to maintain effective batch size. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Platform Requirements**: Audio + GPU training works on native Linux (Lambda Labs). Mac M-series and HF Jobs containers have torchcodec/FFmpeg limitations.
