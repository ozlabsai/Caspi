# vLLM vs FastAPI for Qwen3-ASR

## TL;DR - **UPDATE: vLLM IS SUPPORTED!**

**vLLM officially supports Qwen3-ASR** with day-0 support! ✅

- ✅ Official audio model support
- ✅ OpenAI-compatible API
- ✅ Continuous batching for higher throughput
- ✅ Production-ready with monitoring

**Recommendation**: Use vLLM for production deployments (especially with concurrent users). Use FastAPI + qwen-asr for simple/custom use cases.

See [`DEPLOY_WITH_VLLM.md`](DEPLOY_WITH_VLLM.md) for vLLM deployment guide.

---

## What Changed?

**Previous Understanding (Incorrect)**:
- vLLM only supports text-to-text models
- No audio preprocessing in vLLM
- Requires custom integration

**Current Reality (Correct)**:
- ✅ vLLM added official Qwen3-ASR support
- ✅ Built-in audio preprocessing pipeline
- ✅ OpenAI-compatible transcription API
- ✅ Docker images available

---

## What is vLLM?

vLLM is a high-performance inference engine optimized for:
- **Text generation** with autoregressive decoding
- **Multimodal models** (vision, audio, etc.)
- **Continuous batching** for multiple concurrent requests
- **PagedAttention** for efficient memory management
- **OpenAI-compatible APIs** for easy integration

### vLLM Now Excels At:
- Text-to-text models (GPT, Llama, Mistral, Qwen, etc.)
- Vision models (LLaVA, Qwen-VL, etc.)
- **Audio models** (Qwen3-ASR) ← **NEW!**
- High-throughput multi-modal inference

---

## Why vLLM IS Great for Qwen3-ASR

### 1. Built-in Audio Preprocessing ✅

vLLM now handles **audio feature extraction** automatically:

```python
# vLLM handles this internally!
audio_array (waveform)
  ↓ (vLLM's audio processor)
mel_spectrogram (128 x T features)
  ↓ (Qwen3-ASR encoder)
text tokens
```

You just pass audio files/URLs, vLLM handles the rest.

### 2. Continuous Batching ✅

**vLLM's key benefit for Qwen3-ASR**:
- Batch multiple audio transcription requests together
- Efficient GPU utilization with multiple concurrent users
- Dynamic batching (requests come and go)

**Without vLLM (FastAPI)**:
```
Request 1: [====== 4s ======] (GPU idle between requests)
Request 2:                     [====== 4s ======]
Request 3:                                       [====== 4s ======]
Total: 12 seconds
```

**With vLLM**:
```
Request 1: [====== 4s ======]
Request 2: [====== 4s ======]  ← Batched together!
Request 3: [====== 4s ======]
Total: ~6 seconds (2x throughput)
```

### 3. OpenAI-Compatible API ✅

vLLM provides **standard OpenAI transcription API**:

```python
# Standard OpenAI SDK works!
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")
transcription = client.audio.transcriptions.create(
    model="qwen3-asr-hebrew",
    file=audio_file,
    language="he"
)
```

Easy migration if you're already using OpenAI Whisper API.

### 4. Production Features ✅

vLLM includes:
- **Prometheus metrics** for monitoring
- **Health check endpoints**
- **Structured logging**
- **Graceful shutdown**
- **API authentication** support

---

## vLLM Official Support

vLLM supports Qwen3-ASR as a **first-class audio model**:

✅ **Listed in supported models**: [vLLM docs](https://docs.vllm.ai/en/latest/models/supported_models.html#qwen3-asr)
✅ **Official Docker images**: `qwenllm/qwen3-asr:latest`
✅ **OpenAI transcription API**: Full compatibility
✅ **Continuous batching**: Optimized for concurrent requests

---

## Deployment Options Comparison

### 1. **vLLM** (Recommended for Production)
✅ **Official support, high performance**

**Pros**:
- Official Qwen3-ASR support
- Continuous batching (2-3x throughput vs single requests)
- OpenAI-compatible API
- Built-in monitoring (Prometheus)
- Simple deployment (`vllm serve model`)

**Cons**:
- Slightly more complex setup than FastAPI
- Nightly builds required for latest features

**Performance**:
- GPU (RTX A6000): ~2-4 seconds per 30-second audio
- Throughput: ~30-50 transcriptions/minute with batching
- **Best for**: Multiple concurrent users

**Deployment**:
```bash
vllm serve ./qwen3-asr-hebrew-model --host 0.0.0.0 --port 8000
```

---

### 2. **FastAPI + Official qwen-asr Package**
✅ **Simple, custom control**

**Pros**:
- Uses official `qwen-asr` package
- Simple, maintainable codebase
- Full control over audio preprocessing
- Easy to customize

**Cons**:
- No continuous batching
- Manual monitoring setup
- Lower throughput than vLLM

**Performance**:
- GPU (RTX A6000): ~3-5 seconds per 30-second audio
- Throughput: ~10-15 transcriptions/minute
- **Best for**: Single user, custom workflows, prototyping

**Deployment**:
```bash
uv run fastapi serve_asr.py --host 0.0.0.0 --port 8000
```

---

### 3. **HuggingFace Inference Endpoints**
✅ **Managed, auto-scaling**

**Pros**:
- Fully managed infrastructure
- Auto-scaling based on load
- Built-in monitoring
- No ops required

**Cons**:
- Must upload to HF Hub
- Higher cost than self-hosted
- Less control

**Cost**: $0.60-1.50/hour depending on GPU

**Best for**: Managed deployments, variable load

---

### 4. **Docker with vLLM**
✅ **Containerized deployment**

**Pros**:
- Official Docker images from Qwen team
- Reproducible deployments
- Easy to scale with Kubernetes

**Cons**:
- Requires Docker/container knowledge

**Deployment**:
```bash
docker run --gpus all -p 8000:80 qwenllm/qwen3-asr:latest
```

---

## Performance Comparison

| Solution | Latency (30s audio) | Throughput (concurrent) | Setup Complexity | Cost (Lambda) |
|----------|---------------------|------------------------|------------------|---------------|
| **vLLM** ✅ | ~2-4s | ~30-50 req/min | Low | $0.50-0.70/hr |
| **FastAPI + qwen-asr** | ~3-5s | ~10-15 req/min | Low | $0.50-0.70/hr |
| **Docker + vLLM** | ~2-4s | ~30-50 req/min | Medium | $0.50-0.70/hr |
| **HF Endpoints** | ~3-5s | Auto-scales | Low (managed) | $0.60-1.50/hr |

**Winner**: vLLM for production (best performance + ease of use)

---

## When to Use What?

### Use vLLM When:
- ✅ Production deployment
- ✅ Multiple concurrent users
- ✅ Need high throughput (>15 req/min)
- ✅ Want OpenAI-compatible API
- ✅ Need built-in monitoring

**Deployment**:
```bash
vllm serve ./qwen3-asr-hebrew-model
```

### Use FastAPI + qwen-asr When:
- ✅ Prototyping / development
- ✅ Single user / low concurrency
- ✅ Need custom preprocessing
- ✅ Want full control over pipeline
- ✅ Testing new features

**Deployment**:
```bash
uv run fastapi serve_asr.py
```

### Use Docker + vLLM When:
- ✅ Kubernetes / container orchestration
- ✅ Reproducible deployments
- ✅ CI/CD pipelines
- ✅ Multi-environment (dev/staging/prod)

**Deployment**:
```bash
docker run --gpus all qwenllm/qwen3-asr:latest
```

---

## Recommendation for Production

For your Hebrew ASR model:

### Production (Multiple Users):
```bash
# vLLM on Lambda Labs RTX A6000
vllm serve ./qwen3-asr-hebrew-model --host 0.0.0.0 --port 8000

# Cost: ~$0.50/hr
# Latency: 2-4s per 30s audio
# Throughput: 30-50 req/min
# Setup: 5 minutes
```

**Why vLLM**:
- 2-3x better throughput than FastAPI
- OpenAI-compatible (easy client integration)
- Built-in monitoring and health checks
- Official Qwen support

### Development / Testing:
```bash
# FastAPI for quick prototyping
uv run fastapi serve_asr.py

# Cost: Free (local) or $0.50/hr (GPU)
# Latency: 3-5s per 30s audio
# Setup: 2 minutes
```

### Migration Path:
1. **Start**: FastAPI (local development)
2. **Test**: vLLM (GPU instance, small scale)
3. **Scale**: vLLM + Load Balancer (production)
4. **Enterprise**: HF Endpoints (fully managed)

---

## Summary

| Aspect | vLLM | FastAPI + qwen-asr |
|--------|------|-------------------|
| **Compatibility** | ✅ Official support | ✅ Official package |
| **Audio preprocessing** | ✅ Built-in | ✅ Built-in |
| **Setup complexity** | 🟢 Low (5 min) | 🟢 Low (2 min) |
| **Maintenance** | 🟢 Official support | 🟢 Official package |
| **Performance (single)** | ✅ 2-4s | ✅ 3-5s |
| **Performance (concurrent)** | 🟢 30-50 req/min | 🟡 10-15 req/min |
| **API standard** | 🟢 OpenAI-compatible | 🟡 Custom |
| **Monitoring** | 🟢 Prometheus built-in | 🟡 Manual setup |
| **Cost** | 🟢 $0.50-0.70/hr | 🟢 $0.50-0.70/hr |

**Bottom line**:
- **Production with multiple users**: Use vLLM (better throughput, OpenAI API)
- **Development / single user**: Use FastAPI (simpler, more control)
- **Enterprise**: Use HuggingFace Endpoints (fully managed)

---

## Next Steps

### For Production Deployment:

1. ✅ **Install vLLM**:
   ```bash
   uv pip install -U vllm --pre \
       --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
       --index-strategy unsafe-best-match
   uv pip install "vllm[audio]"
   ```

2. ✅ **Start vLLM server**:
   ```bash
   vllm serve ./qwen3-asr-hebrew-model --host 0.0.0.0 --port 8000
   ```

3. ✅ **Test with OpenAI SDK**:
   ```python
   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
   with open("audio.wav", "rb") as f:
       result = client.audio.transcriptions.create(
           model="qwen3-asr-hebrew", file=f, language="he"
       )
   print(result.text)
   ```

4. ✅ **Deploy to GPU instance** (Lambda Labs / RunPod)
5. ✅ **Set up monitoring** (Prometheus metrics at `/metrics`)
6. ⏭️  **Load test** and optimize batch size
7. ⏭️  **Add authentication** (API keys)

### For Development:

1. ✅ Use FastAPI server (`serve_asr.py`)
2. ✅ Test locally on CPU or small GPU
3. ✅ Migrate to vLLM when ready for production

## Questions?

- **"Should I use vLLM or FastAPI?"** → vLLM for production (better performance), FastAPI for development
- **"Will vLLM work with my fine-tuned model?"** → Yes! Just point to your model directory
- **"Is the OpenAI API the same?"** → Yes, drop-in replacement for OpenAI Whisper API
- **"What's the throughput difference?"** → vLLM: ~30-50 req/min, FastAPI: ~10-15 req/min (same GPU)
- **"Do I need to modify my model?"** → No, vLLM works with your existing model files
