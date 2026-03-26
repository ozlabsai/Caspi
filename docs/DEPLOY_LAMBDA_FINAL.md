# Qwen3-ASR Hebrew Training - Lambda Labs Deployment Guide

## What We Learned

The training script requires **FFmpeg system libraries** because the HuggingFace `datasets` library automatically uses `torchcodec` for audio decoding. This means:

- ✗ **Cannot test locally** (your Mac lacks FFmpeg)
- ✗ **Cannot use HuggingFace Jobs** (their Docker lacks FFmpeg)
- ✓ **Lambda Labs works** (Lambda Stack has FFmpeg pre-installed)

## Files Ready for Deployment

1. **train_qwen3_asr_fixed.py** - Main training script (fixed to work with torchcodec)
2. **train_on_lambda.sh** - Setup script for Lambda instance

## Step-by-Step Deployment

### 1. Launch Lambda Labs Instance

Go to: https://cloud.lambdalabs.com/instances

**Configuration:**
- Instance type: **8x A100 (80 GB SXM4)** - $14.32/hr
- Base image: **Lambda Stack 22.04**
- Filesystem: **Don't attach a filesystem**
- Firewall: **Use default (Global firewall rules)**

Click **Confirm** and wait ~1-2 minutes for instance to become active.

### 2. Copy Instance IP

Once active, copy the instance IP address from the dashboard.

### 3. Upload Files

```bash
cd /Users/guynachshon/Documents/ozlabs/labs/caspi

# Set instance IP
INSTANCE_IP="<your_instance_ip>"

# Upload training files
scp -i ~/.ssh/id_ed25519 train_qwen3_asr_fixed.py ubuntu@$INSTANCE_IP:~/
scp -i ~/.ssh/id_ed25519 train_on_lambda.sh ubuntu@$INSTANCE_IP:~/
```

### 4. SSH Into Instance

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@$INSTANCE_IP
```

### 5. Start Training

```bash
# Start tmux session (so training continues if you disconnect)
tmux new -s training

# Set your HuggingFace token
export HF_TOKEN="<your_hf_token>"

# Install dependencies and run training
chmod +x train_on_lambda.sh
./train_on_lambda.sh
```

**Alternative:** Run the Python script directly:

```bash
# Install dependencies manually
pip install -q torch transformers datasets[audio] accelerate peft librosa soundfile evaluate jiwer scipy

# Run training
export HF_TOKEN="<your_hf_token>"
python3 train_qwen3_asr_fixed.py
```

### 6. Detach from tmux

Press: **Ctrl+B**, then **D**

You can now close SSH. Training will continue running.

### 7. Monitor Progress

Reconnect anytime:

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@$INSTANCE_IP
tmux attach -t training
```

To view just the logs:

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@$INSTANCE_IP
tail -f ~/training.log  # if you used the shell script
```

## Expected Timeline

1. **Dataset Loading** (~5 minutes)
   - Downloads ~17GB of parquet files
   - Loads 210k examples

2. **Audio Processing** (~30-45 minutes)
   - Converts audio to 16kHz WAV files
   - Creates JSONL format for Qwen3-ASR
   - Saves ~200k audio files

3. **Model Training** (~1-2 hours with 8x A100)
   - Downloads Qwen3-ASR-1.7B model
   - Trains for 3 epochs
   - Saves checkpoints every 1000 steps

**Total time: ~2-2.5 hours**

## Cost Estimate

- Instance: $14.32/hour
- Training time: ~2.5 hours
- **Total cost: ~$35-40**

## After Training Completes

### Download the Trained Model

```bash
# From your local machine
scp -r -i ~/.ssh/id_ed25519 ubuntu@$INSTANCE_IP:~/qwen3-asr-hebrew ./
```

### Terminate Instance

**CRITICAL:** Go to https://cloud.lambdalabs.com/instances and **terminate the instance** to stop billing!

## Troubleshooting

### If Training Fails with "torchcodec" Error

This means FFmpeg is missing. Lambda Stack should have it, but if not:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
```

### If Out of Disk Space

The audio processing creates ~200k WAV files. If disk fills up:

```bash
# Check disk space
df -h

# Clean up if needed
rm -rf ~/.cache/huggingface
```

### If Training is Too Slow

The 8x A100 setup should complete in ~2 hours. If slower, check GPU usage:

```bash
nvidia-smi
```

All 8 GPUs should show activity during training.

## What Gets Trained

- **Base model:** Qwen/Qwen3-ASR-1.7B
- **Training data:** 199,866 Hebrew ASR examples
- **Validation data:** 10,520 examples
- **Format:** Audio + Hebrew transcripts
- **Output:** Fine-tuned model in `./qwen3-asr-hebrew/`

## Using the Trained Model

After downloading, you can use it for Hebrew ASR inference:

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model = AutoModelForSpeechSeq2Seq.from_pretrained("./qwen3-asr-hebrew")
processor = AutoProcessor.from_pretrained("./qwen3-asr-hebrew")

# Use for Hebrew speech recognition
# (add inference code here)
```

## Questions?

- Dataset info: https://huggingface.co/datasets/ivrit-ai/crowd-transcribe-v5
- Qwen3-ASR docs: https://github.com/QwenLM/Qwen3-ASR
- Lambda Labs docs: https://docs.lambda.ai
