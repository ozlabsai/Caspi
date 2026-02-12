#!/usr/bin/env python3
"""
Prepare data for Qwen3-ASR training (with multiprocessing)
Creates JSONL files in the format: {"audio": "path/to/file.wav", "text": "language Hebrew<asr_text>TRANSCRIPTION"}

Uses multiprocessing to speed up audio processing by 10-20x on multi-core machines.
"""

import json
import io
import random
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

from datasets import load_dataset
import librosa
import soundfile as sf
from tqdm import tqdm


def normalize_text(text):
    """Clean Hebrew text."""
    if not text:
        return ""
    text = re.sub(r'<\|[\d.]+\|>', '', text)  # Whisper timestamps
    text = re.sub(r'[\u0591-\u05C7]', '', text)  # Niqqud
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)  # Duplicate punctuation
    return ' '.join(text.split()).strip()


def decode_audio(audio_bytes, target_sr=16000):
    """Decode audio bytes with librosa."""
    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
        return audio, sr
    except:
        return None, None


def process_single_example(args):
    """
    Process a single audio example (for multiprocessing).

    Args:
        args: Tuple of (index, audio_struct, text, audio_dir)

    Returns:
        Tuple of (success, jsonl_entry or None, error_reason)
    """
    idx, audio_struct, text, audio_dir = args

    try:
        # Normalize text
        text = normalize_text(str(text))
        if not text:
            return (False, None, "empty_text")

        # Extract audio bytes
        if isinstance(audio_struct, dict) and 'bytes' in audio_struct:
            audio_bytes = audio_struct['bytes']
        elif isinstance(audio_struct, bytes):
            audio_bytes = audio_struct
        else:
            return (False, None, "no_audio_bytes")

        if not audio_bytes:
            return (False, None, "no_audio_bytes")

        # Decode with librosa
        audio, sr = decode_audio(audio_bytes, 16000)
        if audio is None:
            return (False, None, "decode_failed")

        # Check duration (0.5s to 30s)
        dur = len(audio) / sr
        if not (0.5 <= dur <= 30.0):
            return (False, None, f"duration_{dur:.1f}s")

        # Save WAV
        path = audio_dir / f"{idx:06d}.wav"
        sf.write(str(path), audio, sr)

        # JSONL entry in Qwen3-ASR format
        entry = {
            "audio": str(path.absolute()),
            "text": f"language Hebrew<asr_text>{text}"
        }

        return (True, entry, None)

    except Exception as e:
        return (False, None, f"exception_{str(e)[:50]}")


def process_dataset(name, output_dir, split, num_workers=None):
    """
    Process dataset using raw PyArrow table with multiprocessing.

    Args:
        name: Dataset name (e.g., "ivrit-ai/knesset-plenums-whisper-training")
        output_dir: Output directory for processed data
        split: Dataset split (usually "train")
        num_workers: Number of parallel workers (default: cpu_count() - 2)
    """
    print(f"\nProcessing {name}...")

    # Auto-detect optimal worker count
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # Leave 2 cores for system

    print(f"  Using {num_workers} parallel workers")

    # Load dataset
    print("  Loading dataset...")
    ds = load_dataset(name, split="train")

    # Access raw PyArrow table directly
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

    # Convert to dict for parallel processing
    print("  Converting to dict...")
    data_dict = arrow_table.to_pydict()
    total_examples = len(data_dict[text_col])

    # Prepare arguments for parallel processing
    args_list = [
        (i, data_dict['audio'][i], data_dict[text_col][i], audio_dir)
        for i in range(total_examples)
    ]

    # Process in parallel with progress bar
    print(f"  Processing {total_examples} audio files with {num_workers} workers...")

    valid = 0
    skipped = 0
    skip_reasons = {}

    with open(jsonl, 'w', encoding='utf-8') as f:
        with Pool(num_workers) as pool:
            # Use imap_unordered for better performance (order doesn't matter)
            for success, entry, error_reason in tqdm(
                pool.imap_unordered(process_single_example, args_list, chunksize=100),
                total=total_examples,
                desc=f"  {name.split('/')[-1]}",
                unit="files"
            ):
                if success:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    valid += 1
                else:
                    skipped += 1
                    skip_reasons[error_reason] = skip_reasons.get(error_reason, 0) + 1

    print(f"  ✓ {valid} valid examples, {skipped} skipped")
    if skip_reasons:
        print(f"  Skip reasons: {dict(sorted(skip_reasons.items(), key=lambda x: -x[1])[:5])}")

    return jsonl, valid


def main():
    print("=" * 70)
    print("Qwen3-ASR Data Preparation (Multiprocessing Edition)")
    print("=" * 70)

    # Detect CPU count
    num_cpus = cpu_count()
    num_workers = max(1, num_cpus - 2)
    print(f"System: {num_cpus} CPU cores detected, using {num_workers} workers")

    out_dir = Path("./qwen3_asr_data")
    out_dir.mkdir(exist_ok=True)

    # ALL datasets for Round 2.5 (including Knesset!)
    datasets = [
        "ivrit-ai/knesset-plenums-whisper-training",  # ~360 GB, 13,000 hours
        "ivrit-ai/crowd-transcribe-v5",               # ~2 GB, ~200 hours
        "ivrit-ai/crowd-recital-whisper-training"     # ~1 GB, ~100 hours
    ]

    print(f"\nDatasets to process:")
    print(f"  1. Knesset (~360 GB, ~13,000 hours)")
    print(f"  2. Transcribe (~2 GB, ~200 hours)")
    print(f"  3. Recital (~1 GB, ~100 hours)")
    print(f"  Total: ~363 GB raw, ~13,300 hours")
    print()

    # Process all datasets
    jsonls = []
    total = 0
    for ds_name in datasets:
        try:
            jsonl_path, count = process_dataset(ds_name, out_dir, "train", num_workers=num_workers)
            jsonls.append(jsonl_path)
            total += count
        except Exception as e:
            print(f"  ✗ Error processing {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    if not jsonls:
        raise ValueError("No datasets processed successfully!")

    # Combine JSONL files
    print(f"\nCombining {len(jsonls)} files ({total} total examples)...")
    combined = out_dir / "combined.jsonl"
    with open(combined, 'w') as out:
        for jsonl_file in jsonls:
            print(f"  Adding {jsonl_file.name}...")
            with open(jsonl_file) as inf:
                out.write(inf.read())

    # Create train/eval split (95/5)
    print("\nCreating train/eval split (95/5)...")
    with open(combined) as f:
        lines = f.readlines()

    random.seed(42)
    random.shuffle(lines)

    eval_size = int(len(lines) * 0.05)
    train_lines = lines[eval_size:]
    eval_lines = lines[:eval_size]

    train_file = out_dir / "train.jsonl"
    eval_file = out_dir / "eval.jsonl"

    with open(train_file, 'w') as f:
        f.writelines(train_lines)
    with open(eval_file, 'w') as f:
        f.writelines(eval_lines)

    print(f"✓ Train: {len(train_lines):,} examples")
    print(f"✓ Eval: {len(eval_lines):,} examples")

    # Print summary
    print("\n" + "=" * 70)
    print("✓ Data prepared successfully!")
    print("=" * 70)
    print(f"Output directory: {out_dir.absolute()}")
    print(f"  - train.jsonl: {len(train_lines):,} examples")
    print(f"  - eval.jsonl: {len(eval_lines):,} examples")
    print(f"  - audio/: WAV files")
    print()
    print("Next steps:")
    print("  1. Upload to Lambda storage: ./scripts/upload_to_lambda_storage.sh")
    print("  2. Launch GPU training: See ROUND2.5_LAUNCH_GUIDE.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
