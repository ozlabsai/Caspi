#!/usr/bin/env python3
"""
Fine-tune Qwen3-ASR-1.7B on Hebrew ASR datasets using LoRA.

This script combines ivrit-ai/crowd-transcribe-v5 and ivrit-ai/crowd-recital-whisper-training
datasets to train a Hebrew-specialized ASR model.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate


@dataclass
class TrainingConfig:
    """Configuration for Hebrew ASR fine-tuning."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-ASR-1.7B"
    output_dir: str = "./qwen3-asr-hebrew"

    # Dataset configuration
    datasets: List[str] = None
    max_audio_length_seconds: float = 30.0
    min_audio_length_seconds: float = 0.5

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None

    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 500

    # Mixed precision and optimization
    fp16: bool = False
    bf16: bool = True  # Better for modern GPUs
    gradient_checkpointing: bool = True

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = [
                "ivrit-ai/crowd-transcribe-v5",
                "ivrit-ai/crowd-recital-whisper-training"
            ]
        if self.lora_target_modules is None:
            # Target attention and feed-forward layers in the model
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class HebrewASRDataPreprocessor:
    """Handles data loading and preprocessing for Hebrew ASR datasets."""

    def __init__(self, config: TrainingConfig, processor):
        self.config = config
        self.processor = processor
        self.target_sampling_rate = processor.feature_extractor.sampling_rate

    def load_datasets(self) -> tuple:
        """Load and combine Hebrew ASR datasets."""
        print("Loading datasets...")

        datasets = []
        for dataset_name in self.config.datasets:
            try:
                # Load dataset without trust_remote_code (these are Parquet datasets)
                ds = load_dataset(dataset_name, split="train")
                print(f"Loaded {dataset_name}: {len(ds)} examples")
                print(f"  Columns: {ds.column_names}")

                # Standardize column names if needed
                # crowd-recital-whisper-training uses 'transcript'
                # crowd-transcribe-v5 might use different names
                if "transcript" not in ds.column_names and "text" in ds.column_names:
                    ds = ds.rename_column("text", "transcript")

                datasets.append(ds)
            except Exception as e:
                print(f"Warning: Could not load {dataset_name}: {e}")
                continue

        if not datasets:
            raise ValueError("No datasets were successfully loaded!")

        # Combine all datasets
        combined = concatenate_datasets(datasets)
        print(f"Combined dataset: {len(combined)} examples")

        # Cast audio to target sampling rate
        combined = combined.cast_column("audio", Audio(sampling_rate=self.target_sampling_rate))

        # Split into train/validation
        split = combined.train_test_split(test_size=0.05, seed=42)

        return split["train"], split["test"]

    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess audio and text for training.

        This function:
        1. Extracts audio arrays and sampling rates
        2. Processes audio through the feature extractor
        3. Tokenizes transcriptions
        4. Filters by audio length
        """
        # Extract audio data
        audio_arrays = [audio["array"] for audio in examples["audio"]]
        sampling_rates = [audio["sampling_rate"] for audio in examples["audio"]]

        # Get transcripts (handle different field names)
        if "transcript" in examples:
            transcripts = examples["transcript"]
        elif "transcription" in examples:
            transcripts = examples["transcription"]
        elif "text" in examples:
            transcripts = examples["text"]
        else:
            raise ValueError("Could not find transcript field in dataset")

        # Process audio features
        inputs = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        # Tokenize transcripts
        labels = self.processor.tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 for loss computation
        labels["input_ids"] = [
            [(l if l != self.processor.tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

        return {
            "input_features": inputs.input_features,
            "labels": labels.input_ids,
        }


class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for speech-to-text tasks.
    Handles dynamic padding for both audio features and text labels.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        # Pad labels
        label_features = [{"input_ids": label} for label in labels]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def compute_metrics(pred, metric):
    """Compute Word Error Rate (WER) for evaluation."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = metric.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = metric.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = metric.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def setup_lora_model(model, config: TrainingConfig):
    """Configure and apply LoRA to the model."""
    print("Setting up LoRA configuration...")

    # Prepare model for k-bit training (optimization)
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="SEQ_2_SEQ_LM",  # Sequence-to-sequence task
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model


def main():
    """Main training function."""

    # Configuration
    config = TrainingConfig()

    print("=" * 60)
    print("Hebrew ASR Fine-tuning with Qwen3-ASR-1.7B")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Output directory: {config.output_dir}")
    print(f"Batch size: {config.batch_size} (effective: {config.batch_size * config.gradient_accumulation_steps})")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print("=" * 60)

    # Load processor and model
    print("\nLoading model and processor...")
    processor = AutoProcessor.from_pretrained(config.model_name)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Enable gradient checkpointing to save memory
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Apply LoRA
    model = setup_lora_model(model, config)

    # Load and preprocess datasets
    preprocessor = HebrewASRDataPreprocessor(config, processor)
    train_dataset, eval_dataset = preprocessor.load_datasets()

    print("\nPreprocessing datasets...")
    train_dataset = train_dataset.map(
        preprocessor.preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing train dataset",
    )

    eval_dataset = eval_dataset.map(
        preprocessor.preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Preprocessing eval dataset",
    )

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    # Evaluation metric
    wer_metric = evaluate.load("wer")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_epochs,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        push_to_hub=True,  # Push to Hugging Face Hub
        hub_model_id=f"{config.output_dir.replace('./', '')}",
        report_to=["tensorboard"],
        predict_with_generate=True,
        generation_max_length=225,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda pred: compute_metrics(pred, wer_metric),
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)

    # Evaluate
    print("\nFinal evaluation...")
    metrics = trainer.evaluate()
    print(f"Final WER: {metrics['eval_wer']:.2f}%")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
