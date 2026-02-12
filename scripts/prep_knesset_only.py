#!/usr/bin/env python3
"""
Prepare ONLY Knesset dataset and upload to Lambda storage.
Use this to incrementally add Knesset to existing Transcribe + Recital data.

Usage:
  python scripts/prep_knesset_only.py --streaming --upload
"""

import sys
from pathlib import Path

# Add parent directory to path to import from prepare_qwen_data
sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_qwen_data import (
    process_dataset,
    upload_to_lambda_storage,
    cpu_count
)
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Prepare ONLY Knesset dataset")
    parser.add_argument("--upload", action="store_true", help="Upload to Lambda after prep")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (recommended)")
    args = parser.parse_args()

    print("=" * 70)
    print("Knesset-Only Dataset Preparation")
    print("=" * 70)
    print()
    print("This will ONLY process Knesset dataset (~360 GB, ~13,000 hours)")
    print("Output will be saved to: ./qwen3_asr_data_knesset/")
    print()

    # Detect CPU count
    num_cpus = cpu_count()
    num_workers = max(1, num_cpus - 2)
    print(f"System: {num_cpus} CPU cores detected, using {num_workers} workers")

    if args.streaming:
        print("Using STREAMING mode (saves disk space)")

    # Output to separate directory
    out_dir = Path("./qwen3_asr_data_knesset")
    out_dir.mkdir(exist_ok=True)

    # Process Knesset only
    knesset_dataset = "ivrit-ai/knesset-plenums-whisper-training"

    try:
        print()
        jsonl_path, count = process_dataset(
            knesset_dataset,
            out_dir,
            "train",
            num_workers=num_workers,
            streaming=args.streaming
        )
        print(f"\n✓ Knesset processed: {count:,} examples")

        # Rename to knesset-specific name for clarity
        knesset_jsonl = out_dir / "knesset.jsonl"
        jsonl_path.rename(knesset_jsonl)

        print()
        print("=" * 70)
        print("✓ Knesset data prepared successfully!")
        print("=" * 70)
        print(f"Output: {out_dir.absolute()}")
        print(f"  - knesset.jsonl: {count:,} examples")
        print(f"  - audio/train/ivrit-ai_knesset-plenums-whisper-training/: WAV files")
        print()

        # Upload if requested
        if args.upload:
            print("Uploading Knesset to Lambda storage...")
            print("This will add to existing data in s3://900a8c67-830b-40aa-9bc4-079f4c797735/datasets/")
            print()

            # Upload to knesset-specific path
            success = upload_knesset_to_lambda(out_dir)

            if success:
                print()
                print("=" * 70)
                print("✓ Upload complete!")
                print("=" * 70)
                print()
                print("Next steps:")
                print("1. Launch Lambda GPU with HEBREW-ASR-TRAIN filesystem attached")
                print("2. Merge datasets: python scripts/merge_datasets_on_lambda.py")
                print("3. Train: uv run python train_hebrew_asr_enhanced.py")
        else:
            print("To upload:")
            print("  python scripts/prep_knesset_only.py --upload-only")

        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error processing Knesset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def upload_knesset_to_lambda(data_dir: Path):
    """Upload Knesset data to Lambda storage (separate from existing data)."""
    import subprocess

    # Load .env
    env_file = Path(".env")
    if not env_file.exists():
        print("ERROR: .env file not found!")
        return False

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value.strip('"')

    bucket_name = "900a8c67-830b-40aa-9bc4-079f4c797735"
    s3_key = "datasets/knesset/"  # Separate path from existing data

    print(f"\nTarget: s3://{bucket_name}/{s3_key}")
    print(f"Endpoint: {os.environ['S3_ENDPOINT_URL']}")

    # Upload with aws s3 sync
    result = subprocess.run([
        "aws", "s3", "sync",
        str(data_dir),
        f"s3://{bucket_name}/{s3_key}",
        "--endpoint-url", os.environ['S3_ENDPOINT_URL'],
        "--region", os.environ['AWS_REGION']
    ])

    return result.returncode == 0


if __name__ == "__main__":
    main()
