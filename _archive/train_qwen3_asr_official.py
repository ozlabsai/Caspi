#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0.0",
#   "qwen-asr>=0.0.6",
#   "datasets[audio]>=2.14.0",
#   "transformers>=4.45.0",
#   "accelerate>=0.20.0",
#   "soundfile>=0.12.0",
#   "librosa>=0.10.0",
#   "scipy>=1.10.0",
# ]
# ///
"""
Qwen3-ASR Hebrew Fine-tuning using Official Script

Prepares Hebrew datasets in required format and trains using
the official Qwen3-ASR finetuning method.
"""

import json
import os
import tempfile
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Audio
import re
import subprocess
import sys


def normalize_hebrew_text(text: str) -> str:
    """
    Normalize Hebrew text for ASR training.
    - Remove Whisper timestamp tokens
    - Remove niqqud (diacritics)
    - Clean punctuation
    """
    # Handle None or empty text
    if not text:
        return ""

    # Remove Whisper timestamp tokens
    text = re.sub(r'<\|[\d.]+\|>', '', text)

    # Remove Hebrew niqqud
    text = re.sub(r'[\u0591-\u05C7]', '', text)

    # Clean excessive punctuation
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)

    # Standardize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Clean whitespace
    text = ' '.join(text.split())

    return text.strip()


def prepare_hebrew_dataset():
    """
    Load Hebrew ASR datasets and convert to Qwen3-ASR format.

    Required format (JSONL):
    {"audio": "/path/to/audio.wav", "text": "language Hebrew<asr_text>TRANSCRIPT"}
    """
    print("=" * 60)
    print("Preparing Hebrew ASR Datasets for Qwen3-ASR")
    print("=" * 60)

    dataset_names = [
        "ivrit-ai/crowd-transcribe-v5",
        "ivrit-ai/crowd-recital-whisper-training"
    ]

    # Load datasets
    datasets = []
    for dataset_name in dataset_names:
        try:
            print(f"\nLoading {dataset_name}...")
            ds = load_dataset(dataset_name, split="train")
            print(f"  ✓ Loaded: {len(ds)} examples")

            # Standardize column names
            if "transcript" in ds.column_names:
                ds = ds.rename_column("transcript", "text")
            elif "transcription" in ds.column_names:
                ds = ds.rename_column("transcription", "text")

            # Keep only audio and text
            cols_to_keep = ["audio", "text"]
            cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
            if cols_to_remove:
                ds = ds.remove_columns(cols_to_remove)

            # Align audio features by casting to None sampling rate first
            # This ensures concatenation will work across datasets with different audio configs
            ds = ds.cast_column("audio", Audio(sampling_rate=None))

            datasets.append(ds)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    if not datasets:
        raise ValueError("No datasets loaded! Check access permissions.")

    # Combine datasets
    combined = concatenate_datasets(datasets)
    print(f"\n✓ Combined: {len(combined)} examples")

    # Ensure all audio is at 16kHz after concatenation
    print("\nResampling audio to 16kHz...")
    combined = combined.cast_column("audio", Audio(sampling_rate=16000))

    # Normalize text
    print("\nNormalizing Hebrew text...")
    def normalize_example(example):
        example["text"] = normalize_hebrew_text(example["text"])
        return example

    combined = combined.map(normalize_example, desc="Text normalization")

    # Split train/val
    # Note: Text and audio validation happens during JSONL creation to avoid decoding issues
    split = combined.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    print(f"\n✓ Train: {len(train_ds)} examples")
    print(f"✓ Validation: {len(val_ds)} examples")

    return train_ds, val_ds


def save_audio_and_create_jsonl(dataset, output_dir, split_name):
    """
    Save audio files and create JSONL in Qwen3-ASR format.

    Format: {"audio": "/path/to/audio.wav", "text": "language Hebrew<asr_text>TRANSCRIPT"}
    """
    audio_dir = Path(output_dir) / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = Path(output_dir) / f"{split_name}.jsonl"

    print(f"\nCreating {split_name} JSONL...")

    import soundfile as sf
    import librosa
    valid_count = 0
    skipped_count = 0

    # Disable audio decoding by removing the Audio feature temporarily
    # This prevents datasets from trying to use torchcodec
    dataset_no_audio = dataset.remove_columns(["audio"])

    # Access original dataset to get raw audio paths (before decoding)
    # Use the internal arrow table to get raw paths without triggering decoding
    audio_paths = []
    for idx in range(len(dataset)):
        try:
            # Access the raw audio dict without decoding
            # This gets the internal structure before datasets tries to decode
            audio_feature = dataset.features["audio"]
            raw_example = dataset._data.slice(idx, idx + 1).to_pydict()
            audio_dict = raw_example["audio"][0]

            # Extract the file path from the raw audio dict
            if isinstance(audio_dict, dict) and "path" in audio_dict:
                audio_paths.append(audio_dict["path"])
            elif isinstance(audio_dict, str):
                # Sometimes it's just a path string
                audio_paths.append(audio_dict)
            else:
                audio_paths.append(None)
        except Exception as e:
            audio_paths.append(None)

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for idx in range(len(dataset)):
            try:
                # Get text from the dataset (no audio decoding)
                example = dataset_no_audio[idx]
                text = example["text"]

                # Check text is not empty
                if not text or len(str(text).strip()) == 0:
                    skipped_count += 1
                    continue

                # Get audio path
                audio_path_original = audio_paths[idx]
                if not audio_path_original:
                    skipped_count += 1
                    continue

                # Load audio using librosa (completely bypasses datasets library)
                audio_array, sr = librosa.load(audio_path_original, sr=16000, mono=True)

                # Check if audio array is valid
                if audio_array is None or len(audio_array) == 0:
                    skipped_count += 1
                    continue

                # Check duration (0.5-30 seconds)
                duration = len(audio_array) / sr
                if not (0.5 <= duration <= 30.0):
                    skipped_count += 1
                    continue

                # Save audio file
                audio_path = audio_dir / f"audio_{valid_count:06d}.wav"

                # Convert audio array to WAV
                sf.write(
                    str(audio_path),
                    audio_array,
                    sr
                )

                # Create JSONL entry in Qwen3-ASR format
                entry = {
                    "audio": str(audio_path.absolute()),
                    "text": f"language Hebrew<asr_text>{text}"
                }

                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                valid_count += 1

                if valid_count % 1000 == 0:
                    print(f"  Processed {valid_count} valid examples (skipped {skipped_count})...")

            except Exception as e:
                print(f"  Warning: Skipping example {idx} due to error: {e}")
                skipped_count += 1
                continue

    print(f"✓ Created {jsonl_path}")
    print(f"  Valid examples: {valid_count}")
    print(f"  Skipped examples: {skipped_count}")
    return jsonl_path


def main():
    """Main training orchestration."""
    print("=" * 60)
    print("Qwen3-ASR Hebrew Fine-tuning (Official Method)")
    print("=" * 60)

    # Prepare datasets
    train_ds, val_ds = prepare_hebrew_dataset()

    # Create output directory
    output_dir = Path("./qwen3_asr_data")
    output_dir.mkdir(exist_ok=True)

    # Save datasets in Qwen3-ASR format
    print("\n" + "=" * 60)
    print("Converting to Qwen3-ASR Format")
    print("=" * 60)

    train_jsonl = save_audio_and_create_jsonl(train_ds, output_dir, "train")
    val_jsonl = save_audio_and_create_jsonl(val_ds, output_dir, "val")

    print("\n" + "=" * 60)
    print("Starting Qwen3-ASR Training")
    print("=" * 60)

    # Download official training script
    script_url = "https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/finetuning/qwen3_asr_sft.py"
    script_path = output_dir / "qwen3_asr_sft.py"

    print(f"\nDownloading official training script...")
    import urllib.request
    urllib.request.urlretrieve(script_url, script_path)
    print(f"✓ Downloaded to {script_path}")

    # Run official training script
    # Using the official Qwen3-ASR argument names (not standard Transformers)
    train_cmd = [
        sys.executable,
        str(script_path),
        "--model_path", "Qwen/Qwen3-ASR-1.7B",
        "--train_file", str(train_jsonl),
        "--eval_file", str(val_jsonl),
        "--output_dir", "./qwen3-asr-hebrew",
        "--epochs", "3",
        "--batch_size", "8",
        "--grad_acc", "4",
        "--lr", "2e-4",
        "--warmup_ratio", "0.1",
        "--save_steps", "1000",
        "--log_steps", "50",
        "--save_total_limit", "3",
        "--num_workers", "4",
    ]

    print(f"\nRunning official Qwen3-ASR training...")
    print(f"Command: {' '.join(train_cmd)}\n")

    # Run training
    subprocess.run(train_cmd, check=True)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: ./qwen3-asr-hebrew")
    print("=" * 60)


if __name__ == "__main__":
    main()
