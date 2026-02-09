# Deployment Guide: Qwen3-ASR Hebrew Model

This guide covers production deployment options for the fine-tuned Hebrew ASR model.

## Quick Start (Local Testing)

```bash
# Install dependencies
uv add fastapi uvicorn soundfile librosa qwen-asr

# Start server (uses model in current directory)
uv run fastapi serve_asr.py --host 0.0.0.0 --port 8000

# Test transcription
python client_example.py test_audio.wav
```

---

## Deployment Options Comparison

| Option | Cost | Setup Complexity | Scalability | Throughput | Best For |
|--------|------|-----------------|-------------|------------|----------|
| **vLLM (Lambda/RunPod)** | $0.50-0.70/hr | Low | Manual | High (30-50/min) | Production (recommended) |
| **vLLM (Docker)** | $0.50-0.70/hr | Medium | Kubernetes | High (30-50/min) | Container orchestration |
| **FastAPI + qwen-asr** | $0.50-0.70/hr | Low | Manual | Medium (10-15/min) | Development, custom |
| **HF Inference Endpoints** | $0.60-1.50/hr | Low | Auto-scaling | Medium | Managed deployment |
| **RunPod Serverless** | Pay-per-second | Medium | Auto-scaling | Medium-High | Variable workloads |
| **AWS/GCP/Azure** | $1-3/hr | High | Full control | Variable | Enterprise |

---

## Option 0: vLLM Deployment (Recommended for Production)

**Cost**: ~$0.50-0.70/hr for RTX A6000 (48GB)
**Throughput**: ~30-50 transcriptions/minute (2-3x better than FastAPI)

### Why vLLM?

✅ **Official Qwen3-ASR support** - Day-0 model support
✅ **High throughput** - Continuous batching for concurrent requests
✅ **OpenAI-compatible** - Drop-in replacement for OpenAI Whisper API
✅ **Production-ready** - Built-in monitoring, health checks, metrics

See **[`DEPLOY_WITH_VLLM.md`](DEPLOY_WITH_VLLM.md)** for complete vLLM deployment guide.

### Quick Start:

```bash
# Install vLLM with audio support
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]"

# Serve model
vllm serve ./qwen3-asr-hebrew-model --host 0.0.0.0 --port 8000

# Test with OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
result = client.audio.transcriptions.create(
    model="qwen3-asr-hebrew",
    file=open("audio.wav", "rb"),
    language="he"
)
print(result.text)
```

**For full vLLM deployment guide, see [`DEPLOY_WITH_VLLM.md`](DEPLOY_WITH_VLLM.md)**

---

## Option 1: Lambda Labs with FastAPI

**Cost**: ~$0.50-0.70/hr for RTX A6000 (48GB)

### Setup Steps:

1. **Launch Instance:**
   ```bash
   # From Lambda Labs dashboard:
   # - Select: 1x RTX A6000 (48GB)
   # - OS: Ubuntu 22.04
   # - Add SSH key
   ```

2. **Deploy on Lambda:**
   ```bash
   # SSH into instance
   ssh ubuntu@<lambda-ip>

   # Install UV
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.cargo/env

   # Clone/upload your model
   mkdir -p ~/qwen3-asr-hebrew-model
   # Use scp or rsync to upload model files

   # Install dependencies
   cd ~/qwen3-asr-hebrew-model
   uv venv
   source .venv/bin/activate
   uv pip install fastapi uvicorn soundfile librosa qwen-asr torch

   # Start server (with nohup for persistence)
   nohup uv run fastapi serve_asr.py --host 0.0.0.0 --port 8000 > server.log 2>&1 &
   ```

3. **Test:**
   ```bash
   # From your local machine:
   curl http://<lambda-ip>:8000/health
   ```

4. **Production Setup:**
   - Set up nginx as reverse proxy with SSL
   - Use systemd for auto-restart
   - Configure firewall rules

---

## Option 2: RunPod Serverless (Recommended for Variable Load)

**Cost**: Pay only when running (per-second billing)

### Setup Steps:

1. **Create Docker Image:**
   ```dockerfile
   # Dockerfile
   FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

   # Install Python and UV
   RUN apt-get update && apt-get install -y \
       python3.10 python3-pip curl && \
       curl -LsSf https://astral.sh/uv/install.sh | sh

   ENV PATH="/root/.cargo/bin:${PATH}"

   # Copy model files
   WORKDIR /app
   COPY . /app/

   # Install dependencies
   RUN uv pip install --system fastapi uvicorn soundfile librosa qwen-asr torch

   # Expose port
   EXPOSE 8000

   # Run server
   CMD ["python3", "-m", "uvicorn", "serve_asr:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and Push:**
   ```bash
   docker build -t your-dockerhub/qwen3-asr-hebrew:latest .
   docker push your-dockerhub/qwen3-asr-hebrew:latest
   ```

3. **Deploy on RunPod:**
   - Go to RunPod dashboard → Serverless
   - Create endpoint with your Docker image
   - Configure: GPU (RTX 3090/4090), workers, scaling
   - Get API endpoint

4. **Usage:**
   ```python
   import requests

   response = requests.post(
       "https://api.runpod.ai/v2/<endpoint-id>/runsync",
       json={"input": {"audio_base64": "...", "language": "Hebrew"}},
       headers={"Authorization": "Bearer <runpod-api-key>"}
   )
   ```

---

## Option 3: HuggingFace Inference Endpoints

**Cost**: $0.60/hr (T4) - $1.50/hr (A10G)

### Setup Steps:

1. **Upload Model to Hub:**
   ```bash
   # Fix organization permissions first, then:
   cd qwen3-asr-hebrew-model

   # Create model card
   cat > README.md << 'EOF'
   ---
   language: he
   license: apache-2.0
   tags:
   - audio
   - automatic-speech-recognition
   - qwen3
   - hebrew
   ---

   # Qwen3-ASR Hebrew

   Fine-tuned Qwen3-ASR for Hebrew speech recognition.

   ## Usage

   \`\`\`python
   from qwen_asr import Qwen3ASRModel

   model = Qwen3ASRModel.from_pretrained("ozlabs/qwen3-asr-hebrew")
   result = model.transcribe(audio_array, language="Hebrew")
   print(result['text'])
   \`\`\`
   EOF

   # Upload
   huggingface-cli upload ozlabs/qwen3-asr-hebrew . --repo-type model
   ```

2. **Create Endpoint:**
   - Go to https://ui.endpoints.huggingface.co/
   - New Endpoint → Select your model
   - Choose GPU: T4 (16GB) or A10G (24GB)
   - Deploy

3. **Usage:**
   ```python
   import requests
   import base64

   # Read audio
   with open("audio.wav", "rb") as f:
       audio_b64 = base64.b64encode(f.read()).decode()

   # Transcribe
   response = requests.post(
       "https://<endpoint-id>.aws.endpoints.huggingface.cloud",
       headers={"Authorization": "Bearer hf_..."},
       json={"inputs": {"audio": audio_b64, "language": "Hebrew"}}
   )

   print(response.json()['text'])
   ```

---

## Option 4: Self-Hosted with Systemd (Production)

For long-running production deployment on any Ubuntu/Linux server:

### Setup Steps:

1. **Create Systemd Service:**
   ```bash
   sudo tee /etc/systemd/system/qwen-asr.service > /dev/null <<EOF
   [Unit]
   Description=Qwen3-ASR Hebrew API
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/qwen3-asr-hebrew-model
   Environment="PATH=/home/ubuntu/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
   ExecStart=/home/ubuntu/.cargo/bin/uv run fastapi serve_asr.py --host 0.0.0.0 --port 8000
   Restart=always
   RestartSec=10
   StandardOutput=append:/var/log/qwen-asr.log
   StandardError=append:/var/log/qwen-asr-error.log

   [Install]
   WantedBy=multi-user.target
   EOF
   ```

2. **Enable and Start:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable qwen-asr
   sudo systemctl start qwen-asr

   # Check status
   sudo systemctl status qwen-asr

   # View logs
   sudo journalctl -u qwen-asr -f
   ```

3. **Nginx Reverse Proxy (Optional):**
   ```nginx
   # /etc/nginx/sites-available/qwen-asr
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           client_max_body_size 100M;  # Allow large audio files
       }
   }
   ```

---

## Option 5: Local Development/Testing

For local testing on Mac/Windows:

```bash
# Your current setup
cd /Users/guynachshon/Documents/ozlabs/labs/caspi/qwen3-asr-hebrew-model

# Install dependencies (CPU only)
uv pip install fastapi uvicorn soundfile librosa qwen-asr torch

# Start server (CPU mode)
uv run fastapi serve_asr.py

# Test
python client_example.py test_audio.wav
```

**Note**: CPU inference is slow (~30-60s per audio file). GPU recommended for production.

---

## Performance Optimization

### GPU Memory Optimization

If you encounter OOM errors:

```python
# In serve_asr.py, modify model loading:
MODEL = Qwen3ASRModel.from_pretrained(
    str(MODEL_PATH),
    device_map="auto",  # Auto device placement
    torch_dtype=torch.float16,  # Use FP16
    max_memory={0: "20GB"}  # Limit GPU memory
)
```

### Batching for Higher Throughput

For processing multiple files:

```python
@app.post("/transcribe/batch")
async def transcribe_batch(files: list[UploadFile]):
    """Batch transcription (more efficient)."""
    audio_arrays = []

    for file in files:
        audio_bytes = await file.read()
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        # Resample and normalize...
        audio_arrays.append(audio_array)

    # Batch inference (if supported by qwen-asr)
    results = MODEL.transcribe_batch(audio_arrays, language="Hebrew")
    return results
```

---

## Monitoring and Logging

### Add Prometheus Metrics:

```bash
uv pip install prometheus-fastapi-instrumentator
```

```python
# In serve_asr.py
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(...)

# Add metrics endpoint
Instrumentator().instrument(app).expose(app)
```

Access metrics at `http://localhost:8000/metrics`

---

## Security Considerations

1. **Authentication**: Add API key authentication
   ```python
   from fastapi import Security, HTTPException
   from fastapi.security import APIKeyHeader

   api_key_header = APIKeyHeader(name="X-API-Key")

   def verify_api_key(api_key: str = Security(api_key_header)):
       if api_key != "your-secret-key":
           raise HTTPException(status_code=403, detail="Invalid API key")
       return api_key

   @app.post("/transcribe", dependencies=[Depends(verify_api_key)])
   async def transcribe(...):
       ...
   ```

2. **Rate Limiting**: Add rate limits
   ```bash
   uv pip install slowapi
   ```

3. **Input Validation**:
   - Limit audio file size (e.g., max 100MB)
   - Validate audio format
   - Sanitize inputs

---

## Cost Estimate (Monthly)

Assuming 24/7 operation:

| Provider | GPU | $/hour | $/month (730hrs) |
|----------|-----|--------|------------------|
| Lambda Labs | RTX A6000 | $0.70 | $511 |
| RunPod (dedicated) | RTX 4090 | $0.69 | $504 |
| HF Endpoints | T4 | $0.60 | $438 |
| AWS | g5.xlarge (A10G) | $1.01 | $737 |

**Recommendation**:
- **Development/Testing**: Local CPU or spot instances
- **Production (steady load)**: Lambda Labs or RunPod dedicated
- **Production (variable load)**: RunPod Serverless (pay per use)
- **Enterprise**: HuggingFace Endpoints or AWS with auto-scaling

---

## Next Steps

1. **Upload model to HuggingFace** (once org permissions are fixed)
2. **Choose deployment option** based on your workload
3. **Set up monitoring** (Prometheus + Grafana)
4. **Load testing** to determine optimal instance size
5. **Add authentication** and rate limiting
6. **Set up CI/CD** for model updates

## Support

For issues:
- Check logs: `sudo journalctl -u qwen-asr -f`
- GPU memory: `nvidia-smi`
- API health: `curl http://localhost:8000/health`
