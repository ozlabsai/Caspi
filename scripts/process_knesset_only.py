#!/usr/bin/env python3
"""
Process ONLY Knesset dataset using non-streaming mode.
Use this to fix the empty Knesset JSONL issue on GCP.

This script:
1. Uses the cached Knesset data from ~/.cache/huggingface/
2. Processes WITHOUT streaming mode (guarantees success)
3. Outputs to existing qwen3_asr_data/ directory
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_qwen_data import process_dataset, cpu_count

def main():
    print("=" * 70)
    print("Process Knesset Dataset (Non-Streaming Mode)")
    print("=" * 70)
    print()
    print("This will process ONLY Knesset dataset using cached data.")
    print("Streaming mode is DISABLED to ensure all examples are saved.")
    print()

    # Detect CPU count
    num_cpus = cpu_count()
    num_workers = max(1, min(num_cpus - 2, 8))
    print(f"System: {num_cpus} CPU cores detected, using {num_workers} workers")
    print()

    # Output directory (same as main script)
    out_dir = Path("./qwen3_asr_data")
    out_dir.mkdir(exist_ok=True)

    # Knesset dataset
    knesset_dataset = "ivrit-ai/knesset-plenums-whisper-training"

    # Delete empty/broken Knesset files first
    knesset_jsonl = out_dir / "train_ivrit-ai_knesset-plenums-whisper-training.jsonl"
    knesset_audio_dir = out_dir / "audio/train/ivrit-ai_knesset-plenums-whisper-training"

    if knesset_jsonl.exists() and knesset_jsonl.stat().st_size == 0:
        print(f"Removing empty Knesset JSONL: {knesset_jsonl}")
        knesset_jsonl.unlink()

    if knesset_audio_dir.exists():
        print(f"Removing Knesset audio folder: {knesset_audio_dir}")
        import shutil
        shutil.rmtree(knesset_audio_dir)

    print()
    print("=" * 70)
    print("Processing Knesset (this will take 2-4 hours)")
    print("=" * 70)
    print()

    try:
        # Process with streaming=False (use cached data)
        jsonl_path, count = process_dataset(
            knesset_dataset,
            out_dir,
            "train",
            num_workers=num_workers,
            streaming=False  # KEY: Non-streaming mode!
        )

        print()
        print("=" * 70)
        print(f"✓ Knesset processed successfully: {count:,} examples")
        print("=" * 70)
        print()
        print(f"Output files:")
        print(f"  - JSONL: {jsonl_path}")
        print(f"  - Audio: {knesset_audio_dir}")
        print()

        # Show file sizes
        if jsonl_path.exists():
            jsonl_size = jsonl_path.stat().st_size / (1024**2)  # MB
            print(f"JSONL size: {jsonl_size:.1f} MB")

        if knesset_audio_dir.exists():
            import subprocess
            result = subprocess.run(
                ["du", "-sh", str(knesset_audio_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"Audio size: {result.stdout.split()[0]}")

        print()
        print("Next steps:")
        print("1. Verify counts match expected (~500K examples, ~360GB)")
        print("2. Upload to Lambda S3: ./scripts/upload_to_lambda_s3.sh")
        print("=" * 70)

    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ Error processing Knesset: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
