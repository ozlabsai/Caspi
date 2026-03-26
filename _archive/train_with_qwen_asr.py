#!/usr/bin/env python3
"""
Fine-tune Qwen3-ASR-1.7B on Hebrew ASR datasets using the official qwen-asr training API.

This script uses the Qwen3-ASR training framework directly, which handles
the model architecture specifics automatically.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model
import evaluate


def load_hebrew_datasets(target_sampling_rate=16000):
    """Load and combine Hebrew ASR datasets."""
    print("=" * 60)
    print("Loading Hebrew ASR Datasets")
    print("=" * 60)

    dataset_names = [
        "ivrit-ai/crowd-transcribe-v5",
        "ivrit-ai/crowd-recital-whisper-training"
    ]

    datasets = []
    for dataset_name in dataset_names:
        try:
            print(f"\nLoading {dataset_name}...")
            ds = load_dataset(dataset_name, split="train")
            print(f"  ✓ Loaded: {len(ds)} examples")
            print(f"  ✓ Columns: {ds.column_names}")

            # Standardize column names
            if "transcript" in ds.column_names:
                # crowd-recital-whisper-training format
                # Remove Whisper-style timestamp tokens if present
                def clean_transcript(example):
                    text = example["transcript"]
                    # Remove <|0.00|> style tokens
                    import re
                    text = re.sub(r'<\|[\d.]+\|>', '', text)
                    example["text"] = text.strip()
                    return example

                ds = ds.map(clean_transcript, desc="Cleaning transcripts")
                ds = ds.remove_columns([col for col in ds.column_names if col not in ["audio", "text"]])

            elif "transcription" in ds.column_names:
                # Rename to standardize
                ds = ds.rename_column("transcription", "text")
                ds = ds.remove_columns([col for col in ds.column_names if col not in ["audio", "text"]])

            # Cast audio to target sampling rate
            ds = ds.cast_column("audio", Audio(sampling_rate=target_sampling_rate))

            datasets.append(ds)

        except Exception as e:
            print(f"  ✗ Error loading {dataset_name}: {e}")
            continue

    if not datasets:
        raise ValueError("No datasets were successfully loaded! Check your access permissions.")

    # Combine all datasets
    combined = concatenate_datasets(datasets)
    print(f"\n✓ Combined dataset: {len(combined)} examples")

    # Remove very short/long samples
    def filter_duration(example):
        audio_array = example["audio"]["array"]
        duration = len(audio_array) / example["audio"]["sampling_rate"]
        return 0.5 <= duration <= 30.0

    combined = combined.filter(filter_duration, desc="Filtering by duration")
    print(f"✓ After filtering: {len(combined)} examples")

    # Split into train/validation
    split = combined.train_test_split(test_size=0.05, seed=42)

    print(f"\n✓ Train set: {len(split['train'])} examples")
    print(f"✓ Validation set: {len(split['test'])} examples")

    return split["train"], split["test"]


def prepare_model_for_training(model_name="Qwen/Qwen3-ASR-1.7B"):
    """
    Load and prepare Qwen3-ASR model with LoRA for training.

    NOTE: This requires the qwen-asr package to be installed.
    The Qwen3-ASR architecture is not in standard Transformers yet.
    """
    print("\n" + "=" * 60)
    print("Preparing Model for Training")
    print("=" * 60)

    try:
        # Try using qwen-asr package
        from qwen_asr import Qwen3ASRModel

        print("Loading model with qwen-asr package...")
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            max_inference_batch_size=-1,  # Unlimited for training
        )

        # The qwen-asr package wraps the model
        # We need to access the underlying model for training
        base_model = model.model

        print(f"✓ Model loaded: {model_name}")

    except ImportError:
        print("qwen-asr package not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qwen-asr"])

        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        base_model = model.model

    # Apply LoRA
    print("\nApplying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",  # Qwen3-ASR uses decoder architecture
    )

    model_with_lora = get_peft_model(base_model, lora_config)

    trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_with_lora.parameters())
    print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model_with_lora, model.processor


def main():
    """Main training orchestration."""
    print("=" * 60)
    print("Qwen3-ASR Hebrew Fine-tuning")
    print("=" * 60)

    # Check for HF token
    if not os.environ.get("HF_TOKEN"):
        print("\n⚠ Warning: HF_TOKEN not set. Authentication may be required.")
        print("  Set it with: export HF_TOKEN=your_token")

    # Load datasets
    try:
        train_dataset, eval_dataset = load_hebrew_datasets()
    except Exception as e:
        print(f"\n✗ Error loading datasets: {e}")
        print("\nMake sure you have:")
        print("  1. Accepted the ivrit.ai license for both datasets")
        print("  2. Authenticated with: huggingface-cli login")
        return 1

    # IMPORTANT NOTE:
    # The qwen-asr package uses a custom training pipeline.
    # For fine-tuning, you should use the official Qwen3-ASR training scripts
    # from: https://github.com/QwenLM/Qwen3-ASR

    print("\n" + "=" * 60)
    print("IMPORTANT: Next Steps for Training")
    print("=" * 60)
    print("""
The Qwen3-ASR model uses a custom architecture that requires
special handling for training.

Recommended approach:

1. Use the official Qwen3-ASR training framework:
   git clone https://github.com/QwenLM/Qwen3-ASR
   cd Qwen3-ASR/finetuning

2. Or, submit a training job on Hugging Face Jobs:
   uv run python launch_training.py

3. The training script will handle:
   - Proper audio preprocessing for Qwen3-ASR
   - Custom data collation for speech-to-text
   - Model-specific optimization settings

For now, your datasets are ready:
  - Train: {len(train_dataset)} samples
  - Validation: {len(eval_dataset)} samples
  - Format: 16kHz audio + Hebrew text
""")

    # Save dataset info for later use
    dataset_info = {
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "columns": train_dataset.column_names,
        "features": str(train_dataset.features),
    }

    import json
    with open("dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("✓ Dataset info saved to dataset_info.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
