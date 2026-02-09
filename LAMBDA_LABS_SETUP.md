# Lambda Labs Training Setup Guide

## Step 1: Launch Lambda Labs Instance

### Option A: Using Lambda Labs Web UI (Easiest)

1. Go to https://cloud.lambdalabs.com/instances
2. Click "Launch Instance"
3. Select GPU type:
   - **Recommended**: 1x A100 (40GB) - ~$1.10/hour
   - **Budget**: 1x A10 (24GB) - ~$0.60/hour
4. Select region with availability
5. Add your SSH key (or create one)
6. Click "Launch"

### Option B: Using Lambda Labs CLI

```bash
# Install Lambda Labs CLI
pip install lambda-cloud

# Configure with your API key
lambda cloud config

# List available instance types
lambda cloud instance-types

# Launch instance (A100)
lambda cloud instance launch \
  --instance-type gpu_1x_a100 \
  --ssh-key-name YOUR_KEY_NAME \
  --name qwen3-asr-training
```

## Step 2: Connect to Instance

Once the instance is running, you'll get an SSH command:

```bash
ssh ubuntu@<INSTANCE_IP>
```

## Step 3: Upload Training Files

From your local machine:

```bash
# Upload training script
scp train_qwen3_asr_official.py ubuntu@<INSTANCE_IP>:~/

# Upload setup script
scp train_on_lambda.sh ubuntu@<INSTANCE_IP>:~/

# Make setup script executable
ssh ubuntu@<INSTANCE_IP> "chmod +x train_on_lambda.sh"
```

## Step 4: Run Training

SSH into the instance and run:

```bash
# Set your Hugging Face token
export HF_TOKEN="your_hf_token_here"

# Run the training
./train_on_lambda.sh
```

### Alternative: Run in tmux/screen (Recommended)

To keep training running even if SSH disconnects:

```bash
# Start tmux session
tmux new -s training

# Set HF token and run training
export HF_TOKEN="your_hf_token_here"
./train_on_lambda.sh

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t training
```

## Step 5: Monitor Training

The training will show progress in the console. Key metrics to watch:
- Training loss (should decrease)
- Validation WER (Word Error Rate - should decrease)
- GPU utilization (should be near 100%)

## Step 6: Retrieve Trained Model

After training completes, the model will be saved to `./qwen3-asr-hebrew/`

Download it to your local machine:

```bash
# From your local machine
scp -r ubuntu@<INSTANCE_IP>:~/qwen3-asr-hebrew ./
```

Or push directly to Hugging Face Hub (recommended):

```bash
# On the Lambda instance, after training
huggingface-cli upload your-username/qwen3-asr-hebrew ./qwen3-asr-hebrew
```

## Step 7: Terminate Instance

**IMPORTANT**: Don't forget to terminate your instance to stop billing!

```bash
# Via CLI
lambda cloud instance terminate <INSTANCE_ID>

# Or via web UI at https://cloud.lambdalabs.com/instances
```

## Cost Estimate

For ~200k examples with A100:
- Data preparation: ~30-45 minutes
- Training (3 epochs): ~6-8 hours
- **Total cost**: ~$8-10

## Troubleshooting

### SSH Connection Issues
```bash
# Check instance status
lambda cloud instance list

# Ensure your SSH key is added
lambda cloud ssh-keys list
```

### Out of Memory
If you get CUDA OOM errors, reduce batch size in the training script:
- Change `--batch_size 8` to `--batch_size 4`
- Or `--batch_size 2` for smaller GPUs

### Training Interrupted
If training stops unexpectedly:
```bash
# Check last checkpoint
ls -lth qwen3-asr-hebrew/

# Resume from checkpoint (modify train script to add --resume_from_checkpoint)
```

## Quick Start Commands

```bash
# 1. Launch instance (Lambda UI recommended)

# 2. Upload files
scp train_qwen3_asr_official.py train_on_lambda.sh ubuntu@<IP>:~/

# 3. SSH and run
ssh ubuntu@<IP>
tmux new -s training
export HF_TOKEN="<your_token>"
chmod +x train_on_lambda.sh
./train_on_lambda.sh

# 4. Detach from tmux: Ctrl+B, D

# 5. Check progress later
ssh ubuntu@<IP>
tmux attach -t training

# 6. After completion, download model
scp -r ubuntu@<IP>:~/qwen3-asr-hebrew ./

# 7. Terminate instance!
```
