#!/usr/bin/env python3
"""
Qwen3-ASR Hebrew Training - Manual Audio Processing

Bypasses datasets Audio feature (which requires torchcodec/FFmpeg)
by manually extracting and decoding audio bytes with librosa.
"""

import json
import io
import random
import re
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
import pyarrow as pa
import librosa
import soundfile as sf


def normalize_text(text):
    """Clean Hebrew text."""
    if not text:
        return ""
    text = re.sub(r'<\|[\d.]+\|>', '', text)  # Whisper timestamps
    text = re.sub(r'[\u0591-\u05C7]', '', text)  # Niqqud
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)  # Duplicate punctuation
    return ' '.join(text.split()).strip()


def decode_audio(audio_bytes, target_sr=16000):
    """Decode audio bytes with librosa (no FFmpeg system libraries needed)."""
    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr)
        return audio, sr
    except:
        return None, None


def process_dataset(name, output_dir, split):
    """Process dataset using raw PyArrow table (bypasses torchcodec)."""
    print(f"\nProcessing {name}...")

    # Load dataset
    ds = load_dataset(name, split="train")

    # Access raw PyArrow table directly (bypasses torchcodec!)
    arrow_table = ds.data

    print(f"  {len(arrow_table)} examples")
    print(f"  Columns: {arrow_table.column_names}")

    # Find text column
    text_columns = ['sentence', 'transcript', 'text', 'transcription']
    text_col = next((c for c in text_columns if c in arrow_table.column_names), None)

    if not text_col:
        raise ValueError(f"No text column found in {name}")

    print(f"  Text column: {text_col}")

    audio_dir = Path(output_dir) / "audio" / split / name.replace("/", "_")
    audio_dir.mkdir(parents=True, exist_ok=True)

    jsonl = Path(output_dir) / f"{split}_{name.replace('/', '_')}.jsonl"

    valid = 0
    skipped = 0
    total_examples = len(arrow_table)

    print(f"  Processing {total_examples} audio files...")

    with open(jsonl, 'w', encoding='utf-8') as f:
        # Process in batches for efficiency
        batch_size = 1000
        for batch_start in range(0, total_examples, batch_size):
            batch_end = min(batch_start + batch_size, total_examples)
            batch_table = arrow_table.slice(batch_start, batch_end - batch_start)
            batch_dict = batch_table.to_pydict()

            batch_len = len(batch_dict[text_col])

            for i in range(batch_len):
                try:
                    # Extract text
                    text = normalize_text(str(batch_dict[text_col][i]))
                    if not text:
                        skipped += 1
                        continue

                    # Extract audio bytes from PyArrow struct
                    audio_struct = batch_dict['audio'][i]

                    if isinstance(audio_struct, dict) and 'bytes' in audio_struct:
                        audio_bytes = audio_struct['bytes']
                    elif isinstance(audio_struct, bytes):
                        audio_bytes = audio_struct
                    else:
                        audio_bytes = None

                    if not audio_bytes:
                        skipped += 1
                        continue

                    # Decode with librosa (pure Python, no FFmpeg)
                    audio, sr = decode_audio(audio_bytes, 16000)
                    if audio is None:
                        skipped += 1
                        continue

                    # Check duration
                    dur = len(audio) / sr
                    if not (0.5 <= dur <= 30.0):
                        skipped += 1
                        continue

                    # Save WAV
                    path = audio_dir / f"{valid:06d}.wav"
                    sf.write(str(path), audio, sr)

                    # JSONL entry in Qwen3-ASR format
                    entry = {
                        "audio": str(path.absolute()),
                        "text": f"language Hebrew<asr_text>{text}"
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    valid += 1

                except Exception as e:
                    skipped += 1
                    continue

            # Progress update after each batch
            progress = (batch_end / total_examples) * 100
            print(f"    Progress: {batch_end}/{total_examples} ({progress:.1f}%) | Valid: {valid}, Skipped: {skipped}")

    print(f"  ✓ {valid} valid examples, {skipped} skipped")
    return jsonl, valid


def main():
    print("=" * 60)
    print("Qwen3-ASR Hebrew Training (Manual Audio Processing)")
    print("=" * 60)

    out_dir = Path("./qwen3_asr_data")
    out_dir.mkdir(exist_ok=True)

    datasets = [
        "ivrit-ai/crowd-transcribe-v5",
        "ivrit-ai/crowd-recital-whisper-training"
    ]

    # Process both datasets
    jsonls = []
    total = 0
    for ds_name in datasets:
        try:
            jsonl_path, count = process_dataset(ds_name, out_dir, "train")
            jsonls.append(jsonl_path)
            total += count
        except Exception as e:
            print(f"  ✗ Error processing {ds_name}: {e}")

    if not jsonls:
        raise ValueError("No datasets processed successfully!")

    # Combine JSONL files
    print(f"\nCombining {len(jsonls)} files ({total} total examples)...")
    combined = out_dir / "combined.jsonl"
    with open(combined, 'w') as out:
        for jsonl_file in jsonls:
            with open(jsonl_file) as inf:
                out.write(inf.read())

    # Create train/val split (95/5)
    print("Creating train/val split (95/5)...")
    with open(combined) as f:
        lines = f.readlines()

    random.seed(42)
    random.shuffle(lines)

    val_size = int(len(lines) * 0.05)
    train_lines = lines[val_size:]
    val_lines = lines[:val_size]

    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"

    with open(train_file, 'w') as f:
        f.writelines(train_lines)
    with open(val_file, 'w') as f:
        f.writelines(val_lines)

    print(f"✓ Train: {len(train_lines)} examples")
    print(f"✓ Val: {len(val_lines)} examples")

    # Download Qwen3-ASR training script
    print("\n" + "=" * 60)
    print("Downloading Qwen3-ASR Training Script")
    print("=" * 60)

    script_url = "https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/finetuning/qwen3_asr_sft.py"
    script_path = out_dir / "qwen3_asr_sft.py"

    import urllib.request
    urllib.request.urlretrieve(script_url, script_path)
    print(f"✓ Downloaded to {script_path}")

    # Run official training
    print("\n" + "=" * 60)
    print("Starting Qwen3-ASR Training")
    print("=" * 60)

    cmd = [
        sys.executable, str(script_path),
        "--model_path", "Qwen/Qwen3-ASR-1.7B",
        "--train_file", str(train_file),
        "--eval_file", str(val_file),
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

    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: ./qwen3-asr-hebrew")
    print("=" * 60)


if __name__ == "__main__":
    main()
