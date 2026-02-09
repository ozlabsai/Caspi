# Lambda Labs H100 Deployment Guide

**Instance**: 192.222.55.73
**GPU**: 1x NVIDIA H100 SXM5 (80GB HBM3)
**Cost**: ~$2.49/hour

---

## Step-by-Step Deployment

### Step 1: Wait for Instance to Initialize (2-3 minutes)

Check if the instance is ready:

```bash
# Test SSH connection
ssh ubuntu@192.222.55.73

# If connection refused, wait 1-2 minutes and try again
# Lambda instances take a few minutes to initialize after creation
```

### Step 2: Upload Model to Lambda

From your local machine:

```bash
cd /Users/guynachshon/Documents/ozlabs/labs/caspi

# Upload model files (3.8GB, takes 2-5 minutes depending on internet speed)
rsync -avz --progress \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='test_output' \
    qwen3-asr-hebrew-model/ \
    ubuntu@192.222.55.73:~/qwen3-asr-hebrew-model/
```

### Step 3: SSH to Lambda Instance

```bash
ssh ubuntu@192.222.55.73
```

### Step 4: Install UV Package Manager

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
source $HOME/.cargo/env

# Verify installation
uv --version
```

### Step 5: Set Up Python Environment

```bash
# Navigate to model directory
cd ~/qwen3-asr-hebrew-model

# Create virtual environment
uv venv --python 3.10

# Activate environment
source .venv/bin/activate
```

### Step 6: Install vLLM with Audio Support

```bash
# Install vLLM nightly with CUDA 12.9 support
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

# Install audio dependencies
uv pip install "vllm[audio]"

# This will take 3-5 minutes
```

### Step 7: Verify GPU

```bash
# Check GPU info
nvidia-smi

# Should show:
# - GPU 0: NVIDIA H100 SXM5
# - Memory: ~80GB
# - CUDA Version: 12.x
```

### Step 8: Test Model Loading

```bash
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 12.x
GPU: NVIDIA H100 SXM5
GPU memory: 80.0 GB
```

### Step 9: Start vLLM Server

```bash
# Start vLLM in background
nohup vllm serve . --host 0.0.0.0 --port 8000 --trust-remote-code > vllm.log 2>&1 &

# Check process started
ps aux | grep vllm

# Monitor logs (Ctrl+C to exit)
tail -f vllm.log
```

Wait for these log messages:
```
INFO: Started server process
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

This takes 1-2 minutes to load the model into GPU memory.

### Step 10: Test API (From Local Machine)

Open a new terminal on your local machine:

```bash
# Test health endpoint
curl http://192.222.55.73:8000/health

# Should return:
# {"status": "healthy"}
```

### Step 11: Test Transcription

Create a test script on your local machine:

```python
# test_vllm_hebrew.py
from openai import OpenAI

client = OpenAI(
    base_url="http://192.222.55.73:8000/v1",
    api_key="EMPTY"
)

# Test with a local audio file
with open("test_audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="qwen3-asr-hebrew",
        file=f,
        language="he"
    )

print("Transcription:", result.text)
```

Run it:
```bash
python test_vllm_hebrew.py
```

---

## Performance Expectations

### H100 SXM5 Specifications:
- **Memory**: 80GB HBM3
- **Memory Bandwidth**: 3.35 TB/s
- **FP16 Performance**: 1,979 TFLOPS
- **Power**: ~700W

### Expected Performance for Qwen3-ASR:
- **Single request**: ~1-2 seconds per 30s audio (faster than A6000)
- **Concurrent throughput**: ~50-80 transcriptions/minute
- **Model fits easily**: 3.8GB model + 16GB context fits in 80GB with room to spare

### Optimization Tips:
```bash
# For maximum throughput
vllm serve . --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.9

# For minimum latency
vllm serve . --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.8
```

---

## Monitoring

### Check vLLM Logs
```bash
ssh ubuntu@192.222.55.73
tail -f ~/qwen3-asr-hebrew-model/vllm.log
```

### Monitor GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or with more details
nvidia-smi dmon -i 0 -s pucvmet
```

### Prometheus Metrics
```bash
# vLLM exposes metrics at /metrics
curl http://192.222.55.73:8000/metrics
```

Key metrics to watch:
- `vllm:num_requests_running` - Active requests
- `vllm:gpu_cache_usage_perc` - GPU memory utilization
- `vllm:avg_generation_throughput_toks_per_s` - Throughput

---

## Production Setup

### 1. Set Up Systemd Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/qwen-asr-vllm.service > /dev/null <<EOF
[Unit]
Description=Qwen3-ASR Hebrew vLLM Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/qwen3-asr-hebrew-model
Environment="PATH=/home/ubuntu/.cargo/bin:/home/ubuntu/qwen3-asr-hebrew-model/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/qwen3-asr-hebrew-model/.venv/bin/vllm serve . --host 0.0.0.0 --port 8000 --trust-remote-code
Restart=always
RestartSec=10
StandardOutput=append:/var/log/qwen-asr-vllm.log
StandardError=append:/var/log/qwen-asr-vllm-error.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable qwen-asr-vllm
sudo systemctl start qwen-asr-vllm

# Check status
sudo systemctl status qwen-asr-vllm

# View logs
sudo journalctl -u qwen-asr-vllm -f
```

### 2. Set Up Nginx Reverse Proxy (Optional)

```bash
# Install nginx
sudo apt update
sudo apt install -y nginx

# Configure reverse proxy
sudo tee /etc/nginx/sites-available/qwen-asr > /dev/null <<EOF
server {
    listen 80;
    server_name 192.222.55.73;

    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/qwen-asr /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 3. Add Authentication

```bash
# Start vLLM with API key
vllm serve . --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --api-key "your-secret-api-key"
```

Client usage:
```python
client = OpenAI(
    base_url="http://192.222.55.73:8000/v1",
    api_key="your-secret-api-key"
)
```

---

## Cost Management

### Lambda Labs H100 Pricing:
- **On-Demand**: $2.49/hour
- **Per day**: $59.76
- **Per month**: ~$1,800

### Cost Optimization:
1. **Use only when needed** - Shut down when not in use
2. **Auto-shutdown script**:
   ```bash
   # Auto-shutdown after 2 hours of inactivity
   sudo apt install -y at
   echo "sudo shutdown -h now" | at now + 2 hours
   ```
3. **Monitor usage** - Track requests per hour
4. **Switch to cheaper GPU** if throughput allows:
   - RTX A6000 (48GB): $0.70/hr (3.5x cheaper)
   - RTX 4090 (24GB): $0.69/hr

---

## Troubleshooting

### Issue: SSH Connection Refused
**Cause**: Instance still initializing
**Solution**: Wait 2-3 minutes after creation, then try again

### Issue: Out of Memory (OOM)
**Cause**: Shouldn't happen with H100's 80GB
**Solution**: Reduce batch size
```bash
vllm serve . --max-num-seqs 8 --gpu-memory-utilization 0.7
```

### Issue: Slow Uploads
**Cause**: Large model file (3.8GB)
**Solution**: Use compression
```bash
rsync -avz --compress qwen3-asr-hebrew-model/ ubuntu@192.222.55.73:~/
```

### Issue: vLLM Import Error
**Cause**: Missing dependencies
**Solution**: Reinstall with audio support
```bash
uv pip install --force-reinstall "vllm[audio]"
```

### Issue: CUDA Version Mismatch
**Cause**: Wrong PyTorch/CUDA version
**Solution**: Use correct wheel index
```bash
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129
```

---

## Next Steps

1. ✅ Wait for instance to initialize (2-3 min)
2. ✅ Upload model with rsync
3. ✅ SSH and install vLLM
4. ✅ Start vLLM server
5. ✅ Test transcription
6. ⏭️ Set up systemd service for production
7. ⏭️ Add monitoring (Prometheus + Grafana)
8. ⏭️ Load test to measure max throughput

---

## Quick Command Reference

```bash
# Upload model
rsync -avz qwen3-asr-hebrew-model/ ubuntu@192.222.55.73:~/qwen3-asr-hebrew-model/

# SSH to instance
ssh ubuntu@192.222.55.73

# Start vLLM
cd ~/qwen3-asr-hebrew-model
source .venv/bin/activate
nohup vllm serve . --host 0.0.0.0 --port 8000 > vllm.log 2>&1 &

# Check logs
tail -f vllm.log

# Monitor GPU
nvidia-smi

# Test API
curl http://192.222.55.73:8000/health

# Stop vLLM
pkill -f "vllm serve"

# Shutdown instance (to save cost)
sudo shutdown -h now
```

---

## Support

- Instance not starting? Wait 3-5 minutes
- Connection issues? Check Lambda Labs dashboard
- Performance issues? Check `nvidia-smi` and vLLM logs
- Questions? See [`DEPLOY_WITH_VLLM.md`](DEPLOY_WITH_VLLM.md)
