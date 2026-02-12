#!/usr/bin/env python3
"""
Prepare data for Qwen3-ASR training (with multiprocessing + optional S3 upload)
Creates JSONL files in the format: {"audio": "path/to/file.wav", "text": "language Hebrew<asr_text>TRANSCRIPTION"}

Uses multiprocessing to speed up audio processing by 10-20x on multi-core machines.
Optionally uploads to Lambda Cloud Storage via S3 API using credentials from .env file.
"""

import json
import io
import random
import re
import os
import subprocess
import sys
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


def process_dataset_streaming(ds_iter, name, output_dir, split, num_workers):
    """
    Process dataset in streaming mode (no full download, processes chunks on-the-fly).

    Args:
        ds_iter: Streaming dataset iterator
        name: Dataset name
        output_dir: Output directory
        split: Dataset split
        num_workers: Number of parallel workers

    Returns:
        Tuple of (jsonl_path, valid_count)
    """
    audio_dir = Path(output_dir) / "audio" / split / name.replace("/", "_")
    audio_dir.mkdir(parents=True, exist_ok=True)

    jsonl = Path(output_dir) / f"{split}_{name.replace('/', '_')}.jsonl"

    # Find text column from first example
    first_example = next(iter(ds_iter))
    text_columns = ['sentence', 'transcript', 'text', 'transcription']
    text_col = next((c for c in text_columns if c in first_example.keys()), None)

    if not text_col:
        raise ValueError(f"No text column found in {name}")

    print(f"  Text column: {text_col}")
    print(f"  Processing streaming dataset in batches...")

    valid = 0
    skipped = 0
    batch_size = num_workers * 100  # Process in chunks

    # Recreate iterator (we consumed first example)
    ds_iter = load_dataset(name, split="train", streaming=True)

    with open(jsonl, 'w', encoding='utf-8') as f:
        batch_args = []

        for idx, example in enumerate(tqdm(ds_iter, desc=f"  {name.split('/')[-1]}", unit="files")):
            # Collect batch
            batch_args.append((
                valid + skipped,  # Use global counter as index
                example['audio'],
                example[text_col],
                audio_dir
            ))

            # Process batch when full
            if len(batch_args) >= batch_size:
                with Pool(num_workers) as pool:
                    for success, entry, error_reason in pool.imap_unordered(process_single_example, batch_args):
                        if success:
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            valid += 1
                        else:
                            skipped += 1

                batch_args = []  # Reset batch

        # Process remaining batch
        if batch_args:
            with Pool(num_workers) as pool:
                for success, entry, error_reason in pool.imap_unordered(process_single_example, batch_args):
                    if success:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        valid += 1
                    else:
                        skipped += 1

    print(f"  ✓ {valid} valid examples, {skipped} skipped")
    return jsonl, valid


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


def process_dataset(name, output_dir, split, num_workers=None, streaming=False):
    """
    Process dataset using raw PyArrow table with multiprocessing.

    Args:
        name: Dataset name (e.g., "ivrit-ai/knesset-plenums-whisper-training")
        output_dir: Output directory for processed data
        split: Dataset split (usually "train")
        num_workers: Number of parallel workers (default: cpu_count() - 2)
        streaming: If True, use streaming mode to avoid downloading entire dataset (saves disk space)
    """
    print(f"\nProcessing {name}...")

    # Auto-detect optimal worker count
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # Leave 2 cores for system

    print(f"  Using {num_workers} parallel workers")
    if streaming:
        print(f"  Using STREAMING mode (saves disk space)")

    # Load dataset (streaming mode if requested)
    print("  Loading dataset...")
    ds = load_dataset(name, split="train", streaming=streaming)

    # For streaming, we'll process in batches without loading all to memory
    if streaming:
        return process_dataset_streaming(ds, name, output_dir, split, num_workers)

    # Non-streaming: access raw PyArrow table directly
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


def upload_to_lambda_storage(data_dir: Path):
    """
    Upload prepared dataset to Lambda Cloud Storage using S3 API.
    Uses 'aws s3 sync' which automatically skips files already uploaded.

    Requires .env file with:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_REGION
    - S3_ENDPOINT_URL
    """
    print("\n" + "=" * 70)
    print("Uploading to Lambda Cloud Storage")
    print("=" * 70)

    # Load credentials from .env
    env_file = Path(".env")
    if not env_file.exists():
        print("ERROR: .env file not found!")
        print("Please create .env with Lambda storage credentials")
        return False

    # Load .env variables
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value.strip('"')

    # Verify required env vars
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'S3_ENDPOINT_URL']
    missing = [v for v in required_vars if v not in os.environ]
    if missing:
        print(f"ERROR: Missing environment variables: {missing}")
        return False

    # Use actual Lambda storage bucket UUID
    # Name: HEBREW-ASR-TRAIN
    # Bucket: 900a8c67-830b-40aa-9bc4-079f4c797735
    bucket_name = "900a8c67-830b-40aa-9bc4-079f4c797735"
    s3_key = "datasets/qwen3_asr_data/"

    print(f"\nTarget: s3://{bucket_name}/{s3_key}")
    print(f"Friendly name: HEBREW-ASR-TRAIN")
    print(f"Endpoint: {os.environ['S3_ENDPOINT_URL']}")
    print(f"Region: {os.environ['AWS_REGION']}")

    # Note: Skip the S3 check to avoid "too many open files" error
    # aws s3 sync will handle checking automatically
    print(f"\nNote: 'aws s3 sync' will automatically skip unchanged files")
    print("Only new or modified files will be uploaded")

    # Upload with aws s3 sync (automatically skips existing files!)
    print(f"\nSyncing {data_dir}/ to s3://{bucket_name}/{s3_key}")
    print("This may take 10-30 minutes for new files...")
    print("(Already uploaded files will be skipped automatically)")
    print()

    result = subprocess.run([
        "aws", "s3", "sync",
        str(data_dir),
        f"s3://{bucket_name}/{s3_key}",
        "--endpoint-url", os.environ['S3_ENDPOINT_URL'],
        "--region", os.environ['AWS_REGION']
    ])

    if result.returncode != 0:
        print("\n❌ Upload failed!")
        return False

    print("\n✅ Upload complete!")
    print(f"\nDataset location: s3://{bucket_name}/{s3_key}")
    print("\nNext steps:")
    print("  1. Launch Lambda GPU instance with HEBREW-ASR-TRAIN filesystem")
    print("  2. Data will be available at: /lambda/nfs/persistent-storage/datasets/qwen3_asr_data/")
    print("  3. Train: uv run python train_hebrew_asr_enhanced.py")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Qwen3-ASR training data")
    parser.add_argument("--upload", action="store_true", help="Upload to Lambda Cloud Storage after prep")
    parser.add_argument("--upload-only", action="store_true", help="Skip prep, only upload existing data")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (saves disk space, required for large datasets on limited disk)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: auto-detect)")
    args = parser.parse_args()

    out_dir = Path("./qwen3_asr_data")

    # Upload-only mode
    if args.upload_only:
        if not out_dir.exists():
            print(f"ERROR: {out_dir} does not exist!")
            print("Run without --upload-only to prepare data first")
            sys.exit(1)

        success = upload_to_lambda_storage(out_dir)
        sys.exit(0 if success else 1)

    # Normal prep workflow
    print("=" * 70)
    print("Qwen3-ASR Data Preparation (Multiprocessing Edition)")
    print("=" * 70)

    # Detect CPU count
    num_cpus = cpu_count()
    if args.workers:
        num_workers = args.workers
        print(f"System: {num_cpus} CPU cores detected, using {num_workers} workers (user-specified)")
    else:
        num_workers = max(1, min(num_cpus - 2, 8))  # Cap at 8 to avoid "too many open files"
        print(f"System: {num_cpus} CPU cores detected, using {num_workers} workers (capped at 8 to avoid file limit issues)")

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

    # Process all datasets (skip if already processed)
    jsonls = []
    total = 0
    for ds_name in datasets:
        # Check if this dataset was already processed
        expected_jsonl = out_dir / f"train_{ds_name.replace('/', '_')}.jsonl"

        if expected_jsonl.exists():
            print(f"\n✓ Skipping {ds_name} (already processed)")
            print(f"  Found: {expected_jsonl}")

            # Count existing samples
            with open(expected_jsonl) as f:
                count = sum(1 for _ in f)
            print(f"  {count:,} examples")

            jsonls.append(expected_jsonl)
            total += count
            continue

        try:
            jsonl_path, count = process_dataset(
                ds_name,
                out_dir,
                "train",
                num_workers=num_workers,
                streaming=args.streaming
            )
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

    # Upload to Lambda storage if requested
    if args.upload:
        upload_to_lambda_storage(out_dir)
    else:
        print("Next steps:")
        print("  1. Upload to Lambda storage: uv run python prepare_qwen_data.py --upload-only")
        print("     (Or use: ./scripts/upload_to_lambda_storage.sh)")
        print("  2. Launch GPU training: See ROUND2.5_LAUNCH_GUIDE.md")

    print("=" * 70)


if __name__ == "__main__":
    main()
