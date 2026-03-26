#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "transformers>=4.45.0",
#   "datasets>=2.14.0",
#   "accelerate>=0.20.0",
#   "soundfile>=0.12.0",
#   "librosa>=0.10.0",
#   "evaluate>=0.4.0",
#   "jiwer>=3.0.0",
# ]
# ///
"""
Qwen3-ASR Hebrew Fine-tuning for HuggingFace Jobs

Uses vanilla transformers.Trainer without qwen-asr wrapper to avoid dependency conflicts.
"""

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import evaluate
from huggingface_hub import login

# Authenticate with HF Hub
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print(f"Authenticating with HF_TOKEN...")
    login(token=hf_token, add_to_git_credential=False)
    print("✓ Authenticated")
else:
    print("Warning: HF_TOKEN not found in environment")


def normalize_hebrew_text(text: str) -> str:
    """Normalize Hebrew text for ASR training."""
    if not text:
        return ""

    # Remove Whisper timestamp tokens
    text = re.sub(r'<\|[\d.]+\|>', '', text)

    # Remove Hebrew niqqud
    text = re.sub(r'[\u0591-\u05C7]', '', text)

    # Clean excessive punctuation
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)

    # Clean whitespace
    text = ' '.join(text.split())

    return text.strip()


def prepare_hebrew_dataset():
    """Load Hebrew ASR datasets."""
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

            # Standardize column names
            if "transcript" in ds.column_names:
                ds = ds.rename_column("transcript", "text")
            elif "sentence" in ds.column_names:
                ds = ds.rename_column("sentence", "text")
            elif "transcription" in ds.column_names:
                ds = ds.rename_column("transcription", "text")

            # Keep only audio and text
            cols_to_keep = ["audio", "text"]
            cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
            if cols_to_remove:
                ds = ds.remove_columns(cols_to_remove)

            # Resample audio to 16kHz
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))

            datasets.append(ds)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    if not datasets:
        raise ValueError("No datasets loaded!")

    # Combine datasets
    combined = concatenate_datasets(datasets)
    print(f"\n✓ Combined: {len(combined)} examples")

    # Normalize text
    print("\nNormalizing Hebrew text...")
    def normalize_example(example):
        example["text"] = normalize_hebrew_text(example["text"])
        return example

    combined = combined.map(normalize_example, desc="Text normalization")

    # Filter out empty text and invalid audio
    def is_valid(example):
        if not example["text"] or len(example["text"].strip()) == 0:
            return False
        if example["audio"]["array"] is None or len(example["audio"]["array"]) == 0:
            return False
        # Check duration (0.5-30 seconds)
        duration = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
        return 0.5 <= duration <= 30.0

    print("\nFiltering invalid examples...")
    combined = combined.filter(is_valid, desc="Filtering")
    print(f"✓ After filtering: {len(combined)} examples")

    # Split train/val
    split = combined.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    print(f"\n✓ Train: {len(train_ds)} examples")
    print(f"✓ Validation: {len(val_ds)} examples")

    return train_ds, val_ds


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text models."""

    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract audio arrays
        audio_features = [{"array": f["audio"]["array"], "sampling_rate": f["audio"]["sampling_rate"]} for f in features]

        # Process audio
        batch = self.processor.feature_extractor(
            [f["array"] for f in audio_features],
            sampling_rate=audio_features[0]["sampling_rate"],
            return_tensors="pt",
            padding=True
        )

        # Process text labels
        labels = [f["text"] for f in features]
        label_features = self.processor.tokenizer(
            labels,
            padding=True,
            return_tensors="pt"
        )

        batch["labels"] = label_features["input_ids"]

        return batch


def compute_metrics(pred):
    """Compute WER metric."""
    wer_metric = evaluate.load("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main():
    """Main training orchestration."""
    print("=" * 60)
    print("Qwen3-ASR Hebrew Fine-tuning (HuggingFace Jobs)")
    print("=" * 60)

    # Prepare datasets
    train_ds, val_ds = prepare_hebrew_dataset()

    # Load model and processor
    print("\n" + "=" * 60)
    print("Loading Qwen3-ASR Model")
    print("=" * 60)

    model_name = "Qwen/Qwen3-ASR-1.7B"

    print(f"\nLoading processor from {model_name}...")
    global processor
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Loading model from {model_name}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("✓ Model loaded")

    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training arguments
    output_dir = "./qwen3-asr-hebrew"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        save_total_limit=3,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
        report_to=["tensorboard"],
    )

    # Create trainer
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
