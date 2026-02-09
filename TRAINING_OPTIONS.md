# Training Options for Hebrew ASR Fine-tuning

**Note**: Hugging Face Jobs requires a Pro subscription ($9/month). Here are your alternatives:

## Option 1: Google Colab (Free/Pro)

### Free Tier
- **GPU**: T4 (16GB)
- **Cost**: Free
- **Limits**: ~12 hours max, may disconnect
- **Best for**: Testing, small experiments

### Colab Pro ($10/month)
- **GPU**: T4, V100, or A100 (24-40GB)
- **Cost**: $10/month
- **Limits**: ~24 hours max
- **Best for**: Full training runs

### Setup:

1. Upload training files to Google Drive
2. Create new Colab notebook
3. Mount Drive and run training:

```python
# Colab notebook cells:

# Install dependencies
!pip install torch transformers datasets peft accelerate librosa soundfile evaluate jiwer

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your project
%cd /content/drive/MyDrive/caspi

# Run training
!python train_hebrew_asr_enhanced.py
```

**Pros**: Free option available, easy setup
**Cons**: May disconnect, slower than dedicated GPU

## Option 2: Kaggle Notebooks (Free)

- **GPU**: T4 (16GB) or P100 (16GB)
- **Cost**: Free
- **Limits**: 30 hours/week GPU time
- **Best for**: Small to medium runs

### Setup:

1. Create Kaggle account
2. Upload dataset or connect to HF
3. Create new notebook with GPU
4. Upload training script and run

**Pros**: More stable than Colab free, better GPU quota
**Cons**: Limited to 9 hours per session

## Option 3: Hugging Face Spaces with GPU (Cheaper Alternative)

- **GPU**: T4 or A10G
- **Cost**: $0.60/hour (T4) or $1.50/hour (A10G)
- **Limits**: None
- **Best for**: Pay-per-use without subscription

### Setup:

1. Create HF Space with GPU
2. Upload training code
3. Run as background process

**Pros**: No subscription, pay only for usage
**Cons**: Requires Space setup, more complex

## Option 4: Local GPU (If Available)

If you have access to a machine with GPU:

### Requirements:
- GPU: ≥16GB VRAM (RTX 4090, A4000, A100, etc.)
- RAM: ≥32GB
- Storage: 50GB free

### Setup:

```bash
# Install dependencies
uv sync

# Run training
uv run python train_hebrew_asr_enhanced.py
```

**Pros**: No cost, full control
**Cons**: Requires hardware, slower training

## Option 5: Cloud GPU Providers

### Lambda Labs
- **GPU**: A100 (40GB)
- **Cost**: ~$1.10/hour
- **Setup**: SSH access, manual setup

### RunPod
- **GPU**: A100, H100, etc.
- **Cost**: $0.79-2.09/hour depending on GPU
- **Setup**: Jupyter or SSH access

### Vast.ai (Cheapest)
- **GPU**: Various (A100, 4090, etc.)
- **Cost**: $0.20-0.80/hour
- **Setup**: SSH or Jupyter

### Example Setup (Lambda/RunPod):

```bash
# SSH into instance
ssh user@instance

# Clone your repo
git clone https://github.com/your-username/caspi
cd caspi

# Install dependencies
pip install torch transformers datasets peft accelerate librosa soundfile evaluate jiwer

# Run training
python train_hebrew_asr_enhanced.py
```

## Recommended Option Based on Budget

### $0 (Free)
→ **Kaggle** (best free option)
- 30 hours GPU/week
- T4 or P100
- Stable sessions

### $10-20 total
→ **Lambda Labs** or **RunPod**
- Rent A100 for 6-8 hours
- ~$7-16 total cost
- Best performance/price

### $10/month ongoing
→ **Google Colab Pro**
- Unlimited sessions
- Access to A100
- Good for experimentation

### Want to avoid subscriptions?
→ **HF Spaces GPU** (pay-per-use)
- $0.60/hour (T4) = ~$5 total
- $1.50/hour (A10G) = ~$12 total
- No subscription needed

## Step-by-Step: Using Google Colab (Recommended for Easy Start)

1. **Prepare Files**:
   ```bash
   # Create a zip of your project
   zip -r caspi.zip train_hebrew_asr_enhanced.py config.yaml
   ```

2. **Upload to Google Drive**:
   - Upload `caspi.zip` to Drive

3. **Create Colab Notebook**:
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Runtime → Change runtime type → T4 GPU

4. **Run This in Colab**:
   ```python
   # Install HF CLI and login
   !pip install -U huggingface_hub[cli]
   from huggingface_hub import notebook_login
   notebook_login()

   # Mount Drive
   from google.colab import drive
   drive.mount('/content/drive')

   # Extract and navigate
   !unzip /content/drive/MyDrive/caspi.zip -d /content/
   %cd /content/caspi

   # Install dependencies
   !pip install torch transformers datasets peft accelerate librosa soundfile evaluate jiwer

   # Run training
   !python train_hebrew_asr_enhanced.py
   ```

5. **Monitor Progress**:
   - Watch output in notebook
   - TensorBoard logs saved to `./qwen3-asr-hebrew/runs/`

## Cost Comparison (for 8-hour training)

| Option | GPU | Cost | Pros |
|--------|-----|------|------|
| Kaggle | T4 | **Free** | Easy, stable |
| Colab Free | T4 | **Free** | May disconnect |
| Colab Pro | A100 | **$10/mo** | Fast, reliable |
| HF Spaces | A10G | **$12** | No subscription |
| Lambda Labs | A100 | **$9** | Best price/perf |
| RunPod | A100 | **$14** | Flexible |
| Vast.ai | A100 | **$5-8** | Cheapest |

## My Recommendation

**For your first run**, I recommend:

1. **Kaggle** (Free, 30hr/week quota)
   - Best free option
   - More stable than Colab
   - Easy setup

2. If you need more power: **Lambda Labs** ($1.10/hr)
   - Rent A100 for one 8-hour session
   - ~$9 total
   - Professional-grade GPU

## Next Steps

Choose your option and I can help you:
1. Set up Kaggle notebook
2. Create Colab training script
3. Configure Lambda Labs instance
4. Or any other option you prefer

Which would you like to proceed with?
