#!/usr/bin/env python3
# Version: 2.0 - Optimized filtering (no audio decoding)
# /// script
# dependencies = [
#   "torch>=2.0.0",
#   "transformers>=4.48.0,<4.49.0",
#   "datasets>=3.6.0,<4.0.0",
#   "accelerate>=0.20.0",
#   "soundfile>=0.12.0",
#   "librosa>=0.10.0",
#   "evaluate>=0.4.0",
#   "jiwer>=3.0.0",
#   "trackio",
#   "peft>=0.7.0"
# ]
# ///
"""
Qwen3-ASR Hebrew Fine-tuning

Trains Qwen3-ASR on Hebrew speech recognition datasets.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
import evaluate


def normalize_hebrew_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<\|[\d.]+\|>', '', text)
    text = re.sub(r'[\u0591-\u05C7]', '', text)
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    text = ' '.join(text.split())
    return text.strip()


def prepare_hebrew_dataset():
    print("=" * 60)
    print("Loading Hebrew ASR Datasets")
    print("=" * 60)

    dataset_names = [
        "ivrit-ai/crowd-transcribe-v5",
        "ivrit-ai/crowd-recital-whisper-training"
    ]

    datasets = []
    for dataset_name in dataset_names:
        print(f"\nLoading {dataset_name}...")
        ds = load_dataset(dataset_name, split="train")
        print(f"  Loaded: {len(ds)} examples")

        if "transcript" in ds.column_names:
            ds = ds.rename_column("transcript", "text")
        elif "sentence" in ds.column_names:
            ds = ds.rename_column("sentence", "text")
        elif "transcription" in ds.column_names:
            ds = ds.rename_column("transcription", "text")

        cols_to_keep = ["audio", "text"]
        cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)

        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    print(f"\nCombined: {len(combined)} examples")

    print("\nNormalizing Hebrew text...")
    combined = combined.map(
        lambda x: {"text": normalize_hebrew_text(x["text"])},
        desc="Text normalization"
    )

    # Filter only empty text (audio validation triggers slow decoding - let Trainer handle invalid audio)
    def is_valid(example):
        return example["text"] and len(example["text"].strip()) > 0

    print("\nFiltering empty text examples...")
    combined = combined.filter(is_valid, desc="Filtering")
    print(f"After filtering: {len(combined)} examples")

    split = combined.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    print(f"\nTrain: {len(train_ds)} examples")
    print(f"Validation: {len(val_ds)} examples")

    return train_ds, val_ds


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        audio_features = [{"array": f["audio"]["array"], "sampling_rate": f["audio"]["sampling_rate"]} for f in features]

        batch = self.processor.feature_extractor(
            [f["array"] for f in audio_features],
            sampling_rate=audio_features[0]["sampling_rate"],
            return_tensors="pt",
            padding=True
        )

        labels = [f["text"] for f in features]
        label_features = self.processor.tokenizer(
            labels,
            padding=True,
            return_tensors="pt"
        )

        batch["labels"] = label_features["input_ids"]
        return batch


def compute_metrics(pred, processor):
    wer_metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    print("=" * 60)
    print("Qwen3-ASR Hebrew Fine-tuning")
    print("=" * 60)

    train_ds, val_ds = prepare_hebrew_dataset()

    print("\n" + "=" * 60)
    print("Loading Qwen3-ASR Model")
    print("=" * 60)

    model_name = "Qwen/Qwen3-ASR-1.7B"
    print(f"\nLoading processor from {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"Loading model from {model_name}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    output_dir = "./qwen3-asr-hebrew"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=True,
        hub_model_id="guychuk/qwen3-asr-hebrew",
        hub_strategy="every_save",
        report_to="trackio",
        run_name="qwen3-hebrew-asr",
    )

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, processor),
        tokenizer=processor.feature_extractor,
    )

    print("\nStarting training...")
    trainer.train()

    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    trainer.push_to_hub()

    print(f"\nTraining complete!")
    print(f"Model saved to Hub: guychuk/qwen3-asr-hebrew")
    print("=" * 60)


if __name__ == "__main__":
    main()
