# Qwen3-ASR Hebrew

Fine-tuned Qwen3-ASR model for Hebrew speech recognition, optimized for performance on H100 GPUs using vLLM.

## Model

**HuggingFace Hub:** [OzLabs/Qwen3-ASR-Hebrew-1.7B](https://huggingface.co/OzLabs/Qwen3-ASR-Hebrew-1.7B)

Fine-tuned on Hebrew datasets from ivrit.ai:
- `ivrit-ai/whisper-training` (100,000 samples)
- `ivrit-ai/eval-d1` (evaluation)
- `ivrit-ai/eval-whatsapp` (evaluation)

Trained on Lambda Labs 8x A100 (80GB) for ~3 hours.

## Project Structure

```
qwen3-asr-hebrew-model/
├── src/
│   ├── qwen_asr/          # Core utilities
│   │   ├── audio.py       # Audio processing (decoding, normalization)
│   │   ├── client.py      # vLLM client wrapper
│   │   └── datasets.py    # Hebrew dataset configuration
│   └── benchmarks/        # Evaluation utilities
│       ├── evaluate.py    # Benchmark runner
│       └── metrics.py     # WER calculation
├── scripts/
│   ├── deploy.sh          # Deploy to Lambda Labs
│   └── benchmark.py       # Run benchmarks
├── pyproject.toml         # Project dependencies
└── README.md
```

## Quick Start

### 1. Deploy to Lambda Labs H100

```bash
# Deploy and start vLLM server
./scripts/deploy.sh [LAMBDA_IP]

# The script will:
# - Install UV and dependencies
# - Download model from HuggingFace Hub (~3.8GB)
# - Start vLLM server on port 8000
```

### 2. Run Benchmarks

```bash
# SSH tunnel to access server locally
ssh -L 8000:localhost:8000 ubuntu@[LAMBDA_IP]

# Run benchmarks on all Hebrew datasets
uv run python scripts/benchmark.py \
    --server http://localhost:8000/v1 \
    --max-samples 100 \
    --output benchmark_results.csv
```

### 3. Use in Code

```python
from src.qwen_asr import VLLMClient

# Connect to vLLM server
client = VLLMClient(base_url="http://localhost:8000/v1")

# Transcribe audio
text = client.transcribe("audio.wav", language="he")
print(text)
```

## Evaluation Datasets

The model is evaluated on three Hebrew ASR datasets:

| Dataset | Description | Samples |
|---------|-------------|---------|
| `ivrit-ai/eval-d1` | General Hebrew speech | Test split |
| `ivrit-ai/eval-whatsapp` | WhatsApp voice messages | Test split |
| `mozilla-foundation/common_voice_17_0` (he) | Common Voice Hebrew | Test split |

## Installation

### Local Development

```bash
# Create environment with UV
uv venv --python 3.10
source .venv/bin/activate

# Install core dependencies
uv pip install -e .

# Install audio processing support
uv pip install -e ".[audio]"

# Install development tools
uv pip install -e ".[dev]"
```

### Lambda Labs Deployment

The deployment script (`scripts/deploy.sh`) automatically handles:
- UV installation
- Python 3.10 environment
- vLLM with audio support
- All benchmark dependencies
- Model download from HuggingFace Hub

## Usage

### Command Line Benchmark

```bash
# Run on all datasets
uv run python scripts/benchmark.py

# Run on specific datasets
uv run python scripts/benchmark.py --datasets eval-d1 eval-whatsapp

# Limit samples per dataset
uv run python scripts/benchmark.py --max-samples 50
```

### Python API

```python
from src.benchmarks import run_benchmark
from src.qwen_asr import VLLMClient

client = VLLMClient(base_url="http://localhost:8000/v1")

# Run benchmarks
results_df = run_benchmark(
    client=client,
    dataset_keys=["eval-d1", "eval-whatsapp"],
    max_samples=100,
    output_file="results.csv"
)

print(results_df)
```

### Audio Processing

```python
from src.qwen_asr.audio import AudioProcessor, normalize_hebrew_text

# Decode audio from various formats (dict, tuple, torchcodec)
audio_array, sample_rate = AudioProcessor.decode_audio(audio_data)

# Ensure mono audio
audio_mono = AudioProcessor.ensure_mono(audio_array)

# Save to WAV
wav_path = AudioProcessor.save_to_wav(audio_data)

# Normalize Hebrew text (remove niqqud)
normalized = normalize_hebrew_text("שָׁלוֹם")  # -> "שלום"
```

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Code Formatting

```bash
# Format with black
uv run black src/ scripts/

# Lint with ruff
uv run ruff check src/ scripts/
```

## Deployment Details

### Lambda Labs H100 Instance

- **GPU:** 1x H100 SXM5 (80GB HBM3)
- **Cost:** ~$2.49/hour
- **vLLM Server:** OpenAI-compatible API on port 8000

### Model Serving

The model is served using vLLM with the following configuration:
- Host: `0.0.0.0` (accessible externally)
- Port: `8000`
- API: OpenAI-compatible `/v1/audio/transcriptions`

### SSH Tunnel

To access the vLLM server from your local machine:

```bash
ssh -L 8000:localhost:8000 ubuntu@[LAMBDA_IP]

# Now you can access http://localhost:8000/v1
```

## License

See the [OzLabs/Qwen3-ASR-Hebrew-1.7B](https://huggingface.co/OzLabs/Qwen3-ASR-Hebrew-1.7B) model card for license information.

## Acknowledgments

- Based on [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- Hebrew datasets from [ivrit.ai](https://ivrit.ai/)
- Trained on [Lambda Labs](https://lambdalabs.com/)
