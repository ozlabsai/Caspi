#!/usr/bin/env python3
"""
Convert cached Arrow dataset to Qwen3-ASR JSONL format with WAV files.

Input: ./preprocessed_dataset_cache/ (HuggingFace Arrow with audio arrays)
Output:
  - ./qwen3_asr_data/train/*.wav
  - ./qwen3_asr_data/eval/*.wav
  - ./qwen3_asr_data/train.jsonl
  - ./qwen3_asr_data/eval.jsonl
"""

import json
import os
from pathlib import Path

import soundfile as sf
from datasets import load_from_disk
from tqdm import tqdm


def convert_split(dataset, output_dir: Path, split_name: str):
    """Convert a dataset split to WAV files + JSONL."""
    wav_dir = output_dir / split_name
    wav_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{split_name}.jsonl"

    print(f"\n[{split_name}] Converting {len(dataset):,} examples...")
    print(f"  WAV files → {wav_dir}/")
    print(f"  JSONL → {jsonl_path}")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, example in enumerate(tqdm(dataset, desc=f"Converting {split_name}")):
            # Extract audio array and sampling rate
            audio_dict = example["audio"]
            audio_array = audio_dict["array"]
            sampling_rate = int(audio_dict["sampling_rate"])

            # Save as WAV file
            wav_filename = f"{split_name}_{idx:08d}.wav"
            wav_path = wav_dir / wav_filename
            sf.write(str(wav_path), audio_array, sampling_rate)

            # Get text and ensure it has the language prefix
            text = example["text"]
            if not text.startswith("language Hebrew"):
                text = f"language Hebrew<asr_text>{text}"

            # Write JSONL entry with absolute path
            jsonl_entry = {
                "audio": str(wav_path.absolute()),
                "text": text
            }
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")

    print(f"✓ {split_name}: {len(dataset):,} examples → {jsonl_path}")


def main():
    cache_dir = Path("./preprocessed_dataset_cache")
    output_dir = Path("./qwen3_asr_data")

    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache not found: {cache_dir}")

    print(f"Loading cached dataset from {cache_dir}...")

    # Load cached splits
    train_ds = load_from_disk(str(cache_dir / "train"))
    eval_ds = load_from_disk(str(cache_dir / "eval"))

    print(f"\nLoaded:")
    print(f"  Train: {len(train_ds):,} examples")
    print(f"  Eval: {len(eval_ds):,} examples")

    # Check disk space (need ~500GB for 3.3M WAV files at ~30s avg)
    import shutil
    stat = shutil.disk_usage(".")
    available_gb = stat.free / (1024**3)
    print(f"\nDisk space available: {available_gb:.1f} GB")

    if available_gb < 500:
        print(f"⚠ WARNING: Low disk space. Conversion may need ~500GB for WAV files.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Convert both splits
    convert_split(train_ds, output_dir, "train")
    convert_split(eval_ds, output_dir, "eval")

    print("\n✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"  1. Verify JSONL files: head -n 1 {output_dir}/train.jsonl")
    print(f"  2. Launch training:")
    print(f"     torchrun --nproc_per_node=8 qwen3_asr_sft_official.py \\")
    print(f"       --model_path Qwen/Qwen3-ASR-1.7B \\")
    print(f"       --train_file {output_dir}/train.jsonl \\")
    print(f"       --eval_file {output_dir}/eval.jsonl \\")
    print(f"       --output_dir ./qwen3-asr-hebrew-round25 \\")
    print(f"       --batch_size 4 \\")
    print(f"       --grad_acc 8 \\")
    print(f"       --epochs 3")


if __name__ == "__main__":
    main()
