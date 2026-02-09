# Deploy Qwen3-ASR Hebrew with vLLM

vLLM officially supports Qwen3-ASR with optimized inference, continuous batching, and OpenAI-compatible API.

## Why vLLM for Qwen3-ASR?

✅ **Official Support**: Day-0 model support from vLLM team
✅ **High Performance**: Continuous batching, PagedAttention optimization
✅ **OpenAI API Compatible**: Drop-in replacement for OpenAI transcription API
✅ **Easy Deployment**: Simple CLI command to start serving
✅ **Production Ready**: Built-in monitoring, logging, and error handling

## Quick Start

### Installation

```bash
# Create environment with UV
uv venv
source .venv/bin/activate

# Install vLLM nightly with audio support
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

# Install audio dependencies
uv pip install "vllm[audio]"
```

### Serve Your Fine-Tuned Model

```bash
# Serve from local directory
vllm serve ./qwen3-asr-hebrew-model --host 0.0.0.0 --port 8000

# Or serve from HuggingFace Hub (once uploaded)
vllm serve ozlabs/qwen3-asr-hebrew --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000` with:
- Chat API: `http://localhost:8000/v1/chat/completions`
- Transcription API: `http://localhost:8000/v1/audio/transcriptions`
- Docs: `http://localhost:8000/docs`

---

## Usage Examples

### 1. OpenAI Transcription API (Recommended)

```python
import httpx
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Transcribe local file
with open("audio.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="./qwen3-asr-hebrew-model",  # Or "ozlabs/qwen3-asr-hebrew"
        file=f,
        language="he"  # Hebrew language code
    )

print(transcription.text)
```

### 2. OpenAI Chat API with Audio

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Encode audio to base64
with open("audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    audio_data_uri = f"data:audio/wav;base64,{audio_base64}"

# Create chat completion
response = client.chat.completions.create(
    model="./qwen3-asr-hebrew-model",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_data_uri}
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### 3. cURL (Simple Testing)

```bash
# Transcription API
curl http://localhost:8000/v1/audio/transcriptions \
    -H "Content-Type: multipart/form-data" \
    -F file="@audio.wav" \
    -F model="./qwen3-asr-hebrew-model" \
    -F language="he"
```

### 4. Python Requests (No OpenAI SDK)

```python
import requests

# Transcription API
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        files={"file": ("audio.wav", f, "audio/wav")},
        data={
            "model": "./qwen3-asr-hebrew-model",
            "language": "he"
        }
    )

print(response.json()["text"])
```

### 5. Offline Inference (Batch Processing)

```python
from vllm import LLM, SamplingParams
import librosa

# Initialize LLM
llm = LLM(model="./qwen3-asr-hebrew-model")

# Load audio
audio, sr = librosa.load("audio.wav", sr=16000)

# Prepare conversation
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": "data:audio/wav;base64,..."}  # base64 audio
            }
        ]
    }
]

# Run inference
sampling_params = SamplingParams(temperature=0.01, max_tokens=256)
outputs = llm.chat(conversation, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

---

## Production Deployment

### Option 1: Lambda Labs with vLLM

```bash
# 1. SSH to Lambda instance
ssh ubuntu@<lambda-ip>

# 2. Install vLLM
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

uv venv
source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]"

# 3. Upload model
# (from local machine)
rsync -avz --progress qwen3-asr-hebrew-model/ ubuntu@<lambda-ip>:~/qwen3-asr-hebrew-model/

# 4. Start vLLM server
nohup vllm serve ~/qwen3-asr-hebrew-model \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    > vllm.log 2>&1 &

# 5. Check logs
tail -f vllm.log
```

### Option 2: Docker (Recommended for Production)

```bash
# Pull official image
docker pull qwenllm/qwen3-asr:latest

# Run container with your fine-tuned model
LOCAL_WORKDIR=$(pwd)  # Current directory with model files
HOST_PORT=8000
CONTAINER_PORT=80

docker run --gpus all --name qwen3-asr-hebrew \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p $HOST_PORT:$CONTAINER_PORT \
    --mount type=bind,source=$LOCAL_WORKDIR,target=/data/shared/Qwen3-ASR \
    --shm-size=4gb \
    -it qwenllm/qwen3-asr:latest

# Inside container, start vLLM
vllm serve /data/shared/Qwen3-ASR/qwen3-asr-hebrew-model \
    --host 0.0.0.0 \
    --port 80
```

Access at `http://localhost:8000/v1/audio/transcriptions`

### Option 3: Systemd Service

```bash
# Create systemd service
sudo tee /etc/systemd/system/qwen-asr-vllm.service > /dev/null <<EOF
[Unit]
Description=Qwen3-ASR Hebrew vLLM Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="PATH=/home/ubuntu/.cargo/bin:/home/ubuntu/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/.venv/bin/vllm serve /home/ubuntu/qwen3-asr-hebrew-model --host 0.0.0.0 --port 8000 --trust-remote-code
Restart=always
RestartSec=10
StandardOutput=append:/var/log/qwen-asr-vllm.log
StandardError=append:/var/log/qwen-asr-vllm-error.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable qwen-asr-vllm
sudo systemctl start qwen-asr-vllm

# Check status
sudo systemctl status qwen-asr-vllm
sudo journalctl -u qwen-asr-vllm -f
```

---

## Advanced Configuration

### GPU Memory Optimization

```bash
# Limit GPU memory usage
vllm serve ./qwen3-asr-hebrew-model \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192

# Multi-GPU deployment
vllm serve ./qwen3-asr-hebrew-model \
    --tensor-parallel-size 2  # Use 2 GPUs
```

### Performance Tuning

```bash
# Increase throughput with larger batch size
vllm serve ./qwen3-asr-hebrew-model \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192

# Reduce latency (smaller batches)
vllm serve ./qwen3-asr-hebrew-model \
    --max-num-seqs 4 \
    --max-num-batched-tokens 2048
```

### Enable Authentication

```bash
# Set API key
vllm serve ./qwen3-asr-hebrew-model \
    --api-key "your-secret-key"

# Use in client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"
)
```

---

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Prometheus Metrics

vLLM exposes Prometheus metrics at `http://localhost:8000/metrics`:

```bash
curl http://localhost:8000/metrics | grep vllm
```

Key metrics:
- `vllm:num_requests_running` - Active requests
- `vllm:num_requests_waiting` - Queued requests
- `vllm:gpu_cache_usage_perc` - GPU cache utilization
- `vllm:avg_prompt_throughput_toks_per_s` - Throughput

### Grafana Dashboard

1. Set up Prometheus to scrape vLLM metrics
2. Import vLLM Grafana dashboard (available in vLLM repo)
3. Monitor:
   - Request latency (p50, p95, p99)
   - GPU utilization
   - Throughput (requests/second)
   - Queue depth

---

## Performance Comparison

| Metric | FastAPI + qwen-asr | vLLM |
|--------|-------------------|------|
| **Latency (30s audio)** | 3-5s | 2-4s |
| **Throughput** | ~10-15 req/min | ~30-50 req/min |
| **Memory Efficiency** | Standard | Optimized (PagedAttention) |
| **Concurrent Requests** | Limited by GPU memory | Continuous batching |
| **API Compatibility** | Custom | OpenAI-compatible |
| **Setup Complexity** | Low | Low |

**Recommendation**: Use vLLM for production workloads with multiple concurrent users.

---

## Client Libraries

### Python Client

```python
# client_vllm.py
from openai import OpenAI

class HebrewASRClient:
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

    def transcribe(self, audio_path: str, language: str = "he") -> str:
        """Transcribe audio file to Hebrew text."""
        with open(audio_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                model="qwen3-asr-hebrew",
                file=f,
                language=language
            )
        return transcription.text

    def transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        """Transcribe multiple audio files."""
        return [self.transcribe(path) for path in audio_paths]

# Usage
client = HebrewASRClient(base_url="http://gpu-host:8000/v1")
text = client.transcribe("audio.wav")
print(text)
```

### JavaScript/TypeScript Client

```typescript
import OpenAI from "openai";
import fs from "fs";

const client = new OpenAI({
    baseURL: "http://localhost:8000/v1",
    apiKey: "EMPTY"
});

async function transcribe(audioPath: string): Promise<string> {
    const audioFile = fs.createReadStream(audioPath);

    const transcription = await client.audio.transcriptions.create({
        model: "qwen3-asr-hebrew",
        file: audioFile,
        language: "he"
    });

    return transcription.text;
}

// Usage
const text = await transcribe("audio.wav");
console.log(text);
```

---

## Troubleshooting

### Issue: "Model not found"

```bash
# Ensure model path is correct
ls -la qwen3-asr-hebrew-model/

# Required files:
# - model.safetensors
# - config.json
# - tokenizer files
# - preprocessor_config.json
# - chat_template.json
```

### Issue: Out of memory (OOM)

```bash
# Reduce memory usage
vllm serve ./qwen3-asr-hebrew-model \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 8
```

### Issue: Slow inference

```bash
# Check GPU utilization
nvidia-smi

# Enable tensor parallelism (if you have multiple GPUs)
vllm serve ./qwen3-asr-hebrew-model \
    --tensor-parallel-size 2
```

### Issue: Connection refused

```bash
# Ensure server is listening on 0.0.0.0 (not 127.0.0.1)
vllm serve ./qwen3-asr-hebrew-model --host 0.0.0.0 --port 8000

# Check firewall
sudo ufw status
sudo ufw allow 8000
```

---

## Cost Estimate

### Lambda Labs (RTX A6000)
- **Cost**: $0.70/hour
- **Throughput**: ~30-50 transcriptions/minute (with vLLM continuous batching)
- **Monthly (24/7)**: $511/month
- **Per transcription**: ~$0.0003-0.0005

### Cost Optimization
- **Use spot instances**: 50-70% cheaper (but can be interrupted)
- **Auto-scaling**: Scale up/down based on demand
- **Serverless**: RunPod serverless charges per second of GPU usage

---

## Next Steps

1. ✅ Install vLLM with audio support
2. ✅ Start vLLM server with your fine-tuned model
3. ✅ Test with OpenAI transcription API
4. ⏭️ Deploy to production (Lambda/RunPod/Docker)
5. ⏭️ Set up monitoring (Prometheus + Grafana)
6. ⏭️ Add authentication and rate limiting
7. ⏭️ Load test to determine optimal instance size

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3-ASR Model Card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/audio)
- [vLLM Audio Support](https://docs.vllm.ai/en/latest/models/supported_models.html#qwen3-asr)
