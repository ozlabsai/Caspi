#!/bin/bash
set -e

echo "========================================="
echo "Fix Knesset Processing on GCP VM"
echo "========================================="
echo ""
echo "This will:"
echo "1. Delete empty Knesset JSONL and audio folder"
echo "2. Process Knesset using NON-streaming mode (uses cached data)"
echo "3. Takes ~2-4 hours but guarantees success"
echo ""

cd ~/Caspi

# Delete empty/broken Knesset files
echo "Removing empty Knesset files..."
rm -f qwen3_asr_data/train_ivrit-ai_knesset-plenums-whisper-training.jsonl
rm -rf qwen3_asr_data/audio/train/ivrit-ai_knesset-plenums-whisper-training/

echo ""
echo "Processing Knesset WITHOUT streaming mode..."
echo "Using cached data from ~/.cache/huggingface/"
echo "This will take 2-4 hours but should succeed."
echo ""

# Process ONLY Knesset, non-streaming (uses cache)
uv run python training/prepare_qwen_data.py \
    --workers 8 \
    --datasets knesset

echo ""
echo "========================================="
echo "✓ Knesset processing complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Verify: ls -lh qwen3_asr_data/train_ivrit-ai_knesset-plenums-whisper-training.jsonl"
echo "2. Verify: du -sh qwen3_asr_data/audio/train/ivrit-ai_knesset-plenums-whisper-training/"
echo "3. Upload to Lambda S3: ./scripts/upload_to_lambda_s3.sh"
echo ""
