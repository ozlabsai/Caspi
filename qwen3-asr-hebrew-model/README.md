# Qwen3-ASR Hebrew

Fine-tuned Qwen3-ASR model for Hebrew speech recognition.

## Model Information

- **Base Model**: Qwen/Qwen3-ASR-1.7B
- **Training Data**: 203,743 Hebrew speech samples from ivrit.ai datasets
- **Training Time**: 3 hours on 8x A100 GPUs
- **Final Loss**: 9.02 (96.6% reduction from 264.81)
- **Model Size**: 3.8GB

## Quick Start

### Local Testing (CPU)

```bash
# Install dependencies
uv pip install -e .

# Start server
uv run fastapi serve_asr.py

# Test transcription
python client_example.py your_audio.wav
```

### Production Deployment (GPU)

```bash
# 1. Upload model to GPU instance (Lambda Labs, RunPod, etc.)
rsync -avz --progress qwen3-asr-hebrew-model/ user@gpu-host:~/qwen3-asr-hebrew-model/

# 2. SSH and deploy
ssh user@gpu-host
cd ~/qwen3-asr-hebrew-model
chmod +x deploy_to_gpu.sh
./deploy_to_gpu.sh
```

Server will be available at `http://gpu-host:8000`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_name": "NVIDIA RTX A6000"
}
```

### Transcribe Audio File

```bash
curl -X POST http://localhost:8000/transcribe/file \
  -F "file=@audio.wav" \
  -F "language=Hebrew"
```

Response:
```json
{
  "text": "שלום, זה בדיקה של מודל קוון לתמלול עברית",
  "language": "Hebrew",
  "model": "qwen3-asr-hebrew",
  "filename": "audio.wav",
  "duration_seconds": 5.2
}
```

### Transcribe Base64 Audio

```bash
# Encode audio
AUDIO_B64=$(base64 -i audio.wav)

# Send request
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\":\"$AUDIO_B64\",\"language\":\"Hebrew\"}"
```

### Python Client

```python
import requests

# Upload file
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe/file",
        files={"file": ("audio.wav", f, "audio/wav")},
        params={"language": "Hebrew"}
    )

result = response.json()
print(result['text'])
```

Or use the provided client:
```bash
python client_example.py audio.wav --server http://gpu-host:8000
```

## Files

```
qwen3-asr-hebrew-model/
├── model.safetensors          # Model weights (3.8GB)
├── config.json                # Model configuration
├── tokenizer.json             # Tokenizer vocabulary
├── tokenizer_config.json      # Tokenizer settings
├── preprocessor_config.json   # Audio preprocessing config
├── chat_template.json         # Chat-based ASR template
│
├── serve_asr.py              # FastAPI production server (alternative)
├── client_example.py         # Example client for FastAPI
├── benchmark_hebrew_asr.py   # Evaluation script
├── deploy_to_gpu.sh          # FastAPI deployment script
│
├── DEPLOYMENT_SUMMARY.md     # Quick start guide ⭐ START HERE
├── DEPLOY_WITH_VLLM.md       # vLLM deployment guide (recommended)
├── DEPLOYMENT.md             # All deployment options comparison
├── WHY_NOT_VLLM.md          # vLLM vs FastAPI comparison
├── FINETUNING_GUIDE.md      # Training documentation
├── pyproject.toml           # Python dependencies
└── README.md                # This file
```

## Benchmarking

Evaluate on Hebrew ASR leaderboard datasets:

```bash
python benchmark_hebrew_asr.py \
  --model-path ./qwen3-asr-hebrew-model \
  --max-samples 100 \
  --output benchmark_results.csv
```

This evaluates on:
- ivrit-ai/eval-d1
- ivrit-ai/eval-whatsapp
- ivrit-ai/saspeech
- google/fleurs (Hebrew)
- mozilla-foundation/common_voice_17_0 (Hebrew)
- imvladikon/hebrew_speech_kan

## Performance

| Hardware | Latency (30s audio) | Cost |
|----------|---------------------|------|
| CPU (Mac M1) | ~30-60s | Free |
| GPU (RTX A6000) | ~3-5s | $0.50-0.70/hr |
| GPU (RTX 4090) | ~2-3s | $0.69/hr |

## Deployment Options

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for detailed deployment guides:

1. **Lambda Labs** (recommended for cost) - $0.50-0.70/hr
2. **RunPod Serverless** (recommended for variable load) - Pay per use
3. **HuggingFace Inference Endpoints** (managed) - $0.60-1.50/hr
4. **Self-hosted with Systemd** (for on-prem)
5. **Local development** (CPU, for testing)

## Deployment with vLLM (Recommended)

vLLM **officially supports** Qwen3-ASR with high-performance inference:

```bash
# Install vLLM with audio support
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]"

# Serve your fine-tuned model
vllm serve ./qwen3-asr-hebrew-model --host 0.0.0.0 --port 8000

# Use OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
result = client.audio.transcriptions.create(
    model="qwen3-asr-hebrew",
    file=open("audio.wav", "rb"),
    language="he"
)
print(result.text)
```

**Benefits**:
- 2-3x higher throughput (continuous batching)
- OpenAI-compatible API
- Built-in monitoring (Prometheus)
- Official Qwen3-ASR support

See [`DEPLOY_WITH_VLLM.md`](DEPLOY_WITH_VLLM.md) for complete vLLM deployment guide.

**When to use what**:
- **vLLM**: Production with multiple concurrent users (recommended)
- **FastAPI + qwen-asr**: Development, custom workflows, single user

## Training

The model was fine-tuned on Hebrew datasets using:
- Base: Qwen/Qwen3-ASR-1.7B
- Infrastructure: Lambda Labs 8x A100 (40GB)
- Training time: ~3 hours
- Final loss: 9.02 (96.6% reduction)

See [`FINETUNING_GUIDE.md`](FINETUNING_GUIDE.md) for complete training details.

## License

Apache 2.0 (inherited from Qwen3-ASR)

## Citation

If you use this model, please cite:

```bibtex
@software{qwen3_asr_hebrew,
  title={Qwen3-ASR Hebrew: Fine-tuned Hebrew Speech Recognition},
  author={OzLabs},
  year={2024},
  url={https://huggingface.co/ozlabs/qwen3-asr-hebrew}
}

@article{qwen3_asr,
  title={Qwen3-ASR: Multilingual Speech Recognition},
  author={Qwen Team},
  year={2024},
  url={https://huggingface.co/Qwen/Qwen3-ASR-1.7B}
}
```

## Acknowledgments

- **Qwen Team** for the base Qwen3-ASR model
- **ivrit.ai** for Hebrew speech datasets
- **Lambda Labs** for GPU infrastructure

## Support

For issues or questions:
- Check [`DEPLOYMENT.md`](DEPLOYMENT.md) for deployment troubleshooting
- Review logs: `sudo journalctl -u qwen-asr -f`
- Check GPU: `nvidia-smi`
- Test API: `curl http://localhost:8000/health`

## TODO

- [ ] Upload model to HuggingFace Hub (pending org permissions)
- [ ] Complete benchmark evaluation on GPU
- [ ] Add batch inference endpoint
- [ ] Add Prometheus metrics
- [ ] Set up load testing
- [ ] Add authentication and rate limiting
