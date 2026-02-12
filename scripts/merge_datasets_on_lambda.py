#!/usr/bin/env python3
"""
Merge Knesset with Transcribe + Recital datasets on Lambda GPU instance.
Run this ON THE LAMBDA GPU INSTANCE after mounting filesystem.

This script:
1. Reads existing qwen3_asr_data (Transcribe + Recital)
2. Reads knesset data
3. Merges into final combined dataset with 95/5 train/eval split
"""

import json
import random
from pathlib import Path


def main():
    print("=" * 70)
    print("Merging Knesset + Transcribe + Recital Datasets")
    print("=" * 70)

    # Paths on Lambda filesystem
    base_path = Path("/lambda/nfs/persistent-storage/datasets")

    # Existing data (Transcribe + Recital)
    existing_train = base_path / "qwen3_asr_data/train.jsonl"
    existing_eval = base_path / "qwen3_asr_data/eval.jsonl"

    # Knesset data
    knesset_jsonl = base_path / "knesset/knesset.jsonl"

    # Output (final merged dataset)
    output_dir = base_path / "qwen3_asr_data_full"
    output_dir.mkdir(exist_ok=True)

    print(f"\nInput sources:")
    print(f"  1. Existing train: {existing_train}")
    print(f"  2. Existing eval: {existing_eval}")
    print(f"  3. Knesset: {knesset_jsonl}")
    print(f"\nOutput: {output_dir}/")

    # Check files exist
    if not existing_train.exists():
        print(f"\nERROR: {existing_train} not found!")
        print("Make sure filesystem is mounted and data uploaded")
        return False

    if not knesset_jsonl.exists():
        print(f"\nERROR: {knesset_jsonl} not found!")
        print("Run: python scripts/prep_knesset_only.py --streaming --upload")
        return False

    # Read all data
    print("\nReading existing data...")
    with open(existing_train) as f:
        existing_train_lines = f.readlines()
    with open(existing_eval) as f:
        existing_eval_lines = f.readlines()

    print(f"  Existing train: {len(existing_train_lines):,} examples")
    print(f"  Existing eval: {len(existing_eval_lines):,} examples")

    print("\nReading Knesset data...")
    with open(knesset_jsonl) as f:
        knesset_lines = f.readlines()

    print(f"  Knesset: {len(knesset_lines):,} examples")

    # Combine all data
    print("\nCombining all datasets...")
    all_lines = existing_train_lines + existing_eval_lines + knesset_lines
    total = len(all_lines)

    print(f"  Total combined: {total:,} examples")

    # Shuffle and split 95/5
    print("\nShuffling and creating 95/5 train/eval split...")
    random.seed(42)
    random.shuffle(all_lines)

    eval_size = int(total * 0.05)
    train_lines = all_lines[eval_size:]
    eval_lines = all_lines[:eval_size]

    # Write output
    train_out = output_dir / "train.jsonl"
    eval_out = output_dir / "eval.jsonl"

    print(f"\nWriting final dataset...")
    with open(train_out, 'w') as f:
        f.writelines(train_lines)

    with open(eval_out, 'w') as f:
        f.writelines(eval_lines)

    print(f"  ✓ Train: {len(train_lines):,} examples")
    print(f"  ✓ Eval: {len(eval_lines):,} examples")

    # Create symlink for easy access
    symlink = Path("./qwen3_asr_data")
    if symlink.exists():
        symlink.unlink()

    symlink.symlink_to(output_dir)

    print("\n" + "=" * 70)
    print("✓ Datasets merged successfully!")
    print("=" * 70)
    print(f"\nFinal dataset: {output_dir}/")
    print(f"  - train.jsonl: {len(train_lines):,} examples")
    print(f"  - eval.jsonl: {len(eval_lines):,} examples")
    print(f"\nSymlink created: ./qwen3_asr_data -> {output_dir}")
    print("\nAudio files referenced in JSONL are in:")
    print(f"  - {base_path}/qwen3_asr_data/audio/")
    print(f"  - {base_path}/knesset/audio/")
    print("\nReady to train!")
    print("  uv run python train_hebrew_asr_enhanced.py")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
