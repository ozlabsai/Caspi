#!/usr/bin/env python3
"""
Pre-build Round 2.5 dataset with all augmentation and preprocessing.

This saves ~2-6 hours of download/processing time during training.
Run this on a machine with good internet, then rsync to training machine.

Usage:
    uv run python scripts/prebuild_round25_dataset.py

Output:
    ./qwen3_asr_round25_data/
        train.jsonl          (metadata)
        eval.jsonl           (metadata)
        wavs/                (audio files)
        dataset_info.json    (statistics)
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Audio, interleave_datasets
import soundfile as sf
import hashlib

# Configuration
OUTPUT_DIR = Path("./qwen3_asr_round25_data")
DATASETS = [
    "ivrit-ai/knesset-plenums-whisper-training",
    "ivrit-ai/crowd-transcribe-v5",
    "ivrit-ai/crowd-recital-whisper-training"
]

# Balanced sampling (50-30-20)
SAMPLING_PROBS = {
    "ivrit-ai/knesset-plenums-whisper-training": 0.50,
    "ivrit-ai/crowd-transcribe-v5": 0.30,
    "ivrit-ai/crowd-recital-whisper-training": 0.20,
}

TARGET_SAMPLE_RATE = 16000
TRAIN_SPLIT_RATIO = 0.95  # 95% train, 5% eval


def deduplicate_by_audio_hash(dataset, dataset_name):
    """Remove duplicate audio samples by content hash."""
    print(f"\n  Deduplicating {dataset_name}...")
    seen_hashes = set()
    unique_samples = []
    duplicates = 0

    for sample in tqdm(dataset, desc=f"  Hashing {dataset_name}"):
        # Hash audio content
        audio_bytes = sample["audio"]["array"].tobytes()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()

        if audio_hash not in seen_hashes:
            seen_hashes.add(audio_hash)
            unique_samples.append(sample)
        else:
            duplicates += 1

    print(f"  ✓ Removed {duplicates:,} duplicates, kept {len(unique_samples):,}")
    return unique_samples


def main():
    print("=" * 70)
    print("Pre-building Round 2.5 Dataset with Balanced Sampling")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Download all 3 datasets (~5,050 hours)")
    print("  2. Apply balanced interleaving (50-30-20)")
    print("  3. Deduplicate by audio content hash")
    print("  4. Save as JSONL + WAV files")
    print("  5. Create train/eval splits (95/5)")
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Expected size: ~150-200GB")
    print("=" * 70)

    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "wavs").mkdir(exist_ok=True)

    # Load datasets with balanced sampling
    print("\n" + "=" * 70)
    print("Loading datasets with balanced sampling...")
    print("=" * 70)

    datasets_dict = {}
    sampling_probs = []
    total_samples = 0

    for dataset_name in DATASETS:
        print(f"\nLoading {dataset_name}...")
        try:
            ds = load_dataset(dataset_name, split="train")
            print(f"  ✓ Loaded: {len(ds):,} examples")

            # Cast audio to target sample rate
            ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

            # Deduplicate
            unique_samples = deduplicate_by_audio_hash(ds, dataset_name)

            # Convert back to dataset
            from datasets import Dataset as HFDataset
            ds = HFDataset.from_list(unique_samples)

            datasets_dict[dataset_name] = ds
            prob = SAMPLING_PROBS[dataset_name]
            sampling_probs.append(prob)
            total_samples += len(ds)

            print(f"  Sampling probability: {prob:.1%}")

        except Exception as e:
            print(f"  ✗ Error loading {dataset_name}: {e}")
            continue

    if not datasets_dict:
        raise ValueError("No datasets loaded!")

    # Normalize probabilities
    total_prob = sum(sampling_probs)
    sampling_probs = [p / total_prob for p in sampling_probs]

    print(f"\n{'=' * 70}")
    print("Dataset Sampling Strategy:")
    for (name, ds), prob in zip(datasets_dict.items(), sampling_probs):
        short_name = name.split("/")[-1]
        print(f"  {short_name:40s}: {prob:5.1%} ({len(ds):,} samples)")
    print(f"{'=' * 70}\n")

    # Interleave datasets
    print("Interleaving datasets with balanced sampling...")
    combined = interleave_datasets(
        list(datasets_dict.values()),
        probabilities=sampling_probs,
        seed=42,
        stopping_strategy="all_exhausted"
    )
    print(f"✓ Combined: {len(combined):,} examples\n")

    # Split train/eval
    num_train = int(len(combined) * TRAIN_SPLIT_RATIO)
    train_data = combined.select(range(num_train))
    eval_data = combined.select(range(num_train, len(combined)))

    print(f"Split: {len(train_data):,} train, {len(eval_data):,} eval\n")

    # Save datasets
    for split_name, split_data in [("train", train_data), ("eval", eval_data)]:
        print(f"Processing {split_name} split...")
        jsonl_path = OUTPUT_DIR / f"{split_name}.jsonl"

        with open(jsonl_path, 'w') as f:
            for idx, sample in enumerate(tqdm(split_data, desc=f"  Saving {split_name}")):
                # Save audio as WAV
                audio_filename = f"{split_name}_{idx:08d}.wav"
                audio_path = OUTPUT_DIR / "wavs" / audio_filename

                audio_array = sample["audio"]["array"]
                sf.write(audio_path, audio_array, TARGET_SAMPLE_RATE)

                # Create JSONL entry
                entry = {
                    "audio": str(audio_path.absolute()),
                    "text": sample["text"],
                    "has_timestamps": sample.get("has_timestamps", False),
                    "prev_transcript": sample.get("prev_transcript", ""),
                    "duration": len(audio_array) / TARGET_SAMPLE_RATE,
                }

                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"  ✓ Saved {split_name}.jsonl ({len(split_data):,} samples)")

    # Save dataset info
    info = {
        "total_samples": len(combined),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "datasets": {name: len(ds) for name, ds in datasets_dict.items()},
        "sampling_probs": {name: prob for name, prob in zip(DATASETS, sampling_probs)},
        "sample_rate": TARGET_SAMPLE_RATE,
        "format": "JSONL + WAV",
        "balanced_sampling": "50-30-20",
    }

    with open(OUTPUT_DIR / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Dataset pre-built successfully!")
    print(f"{'=' * 70}")
    print(f"\nLocation: {OUTPUT_DIR.absolute()}")
    print(f"Files:")
    print(f"  train.jsonl: {len(train_data):,} samples")
    print(f"  eval.jsonl: {len(eval_data):,} samples")
    print(f"  wavs/: {len(combined):,} WAV files")
    print(f"  dataset_info.json: Metadata")
    print(f"\nNext steps:")
    print(f"  1. Check size: du -sh {OUTPUT_DIR}")
    print(f"  2. Rsync to training machine:")
    print(f"     rsync -avz --progress {OUTPUT_DIR}/ ubuntu@<ip>:~/caspi/{OUTPUT_DIR.name}/")
    print(f"  3. Update training script to load from local JSONL")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
