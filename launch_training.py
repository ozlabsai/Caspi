#!/usr/bin/env python3
"""
Launch Qwen3-ASR Hebrew fine-tuning job on Hugging Face Jobs infrastructure.

This script uses UV for dependency management and submits a training job
to Hugging Face's cloud GPU infrastructure.
"""

import subprocess
import sys
from pathlib import Path


def check_hf_auth():
    """Verify Hugging Face authentication."""
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Authenticated as: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("✗ Not authenticated with Hugging Face")
        print("  Run: huggingface-cli login")
        return False


def launch_job():
    """Launch training job on Hugging Face Jobs."""

    print("=" * 60)
    print("Launching Qwen3-ASR Hebrew Fine-tuning Job")
    print("=" * 60)

    # Check authentication
    if not check_hf_auth():
        sys.exit(1)

    # Create job script with inline dependencies (PEP 723 format)
    job_script = '''#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "torch>=2.0.0",
#   "transformers>=4.45.0",
#   "datasets>=2.14.0",
#   "accelerate>=0.20.0",
#   "peft>=0.7.0",
#   "librosa>=0.10.0",
#   "soundfile>=0.12.0",
#   "evaluate>=0.4.0",
#   "jiwer>=3.0.0",
#   "tqdm>=4.65.0",
#   "numpy>=1.24.0",
# ]
# ///

import sys
sys.path.insert(0, "/job")
from train_hebrew_asr_enhanced import main

if __name__ == "__main__":
    main()
'''

    # Write job script
    job_script_path = Path("job_runner.py")
    job_script_path.write_text(job_script)
    print(f"✓ Created job script: {job_script_path}")

    # Prepare job command
    job_name = "qwen3-asr-hebrew-training"

    # HF Jobs uses "flavor" for hardware selection
    # Options: a100-large, a10g-large, l4x1, etc.
    flavor = "a100-large"  # A100 GPU
    # flavor = "a10g-large"  # Cheaper A10G option

    # Build the command using correct HF Jobs CLI syntax
    # Format: hf jobs run [--flavor FLAVOR] image command
    cmd = [
        "hf", "jobs", "run",
        "--flavor", flavor,
        "--timeout", "12h",  # Maximum 12 hours
        "-d",  # Detach (run in background)
        "--secrets", "HF_TOKEN",  # Pass HF token to job
        "python:3.11",  # Docker image
        "bash", "-c",
        # Command to run in the job
        f"""
        pip install torch transformers datasets peft accelerate librosa soundfile evaluate jiwer && \
        huggingface-cli login --token $HF_TOKEN && \
        python train_hebrew_asr_enhanced.py
        """
    ]

    print(f"\n{'=' * 60}")
    print(f"Job Configuration:")
    print(f"  Flavor: {flavor}")
    print(f"  Timeout: 12 hours")
    print(f"  Detached: Yes")
    print(f"{'=' * 60}\n")

    print("Submitting job to Hugging Face Jobs...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("\n✓ Job submitted successfully!")
        print("\nMonitor your job:")
        print(f"  Web: https://huggingface.co/jobs")
        print(f"  CLI: hf jobs ps")
        print(f"  Logs: hf jobs logs {job_name}")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Job submission failed: {e}")
        sys.exit(1)


def monitor_job(job_name: str):
    """Monitor job progress."""
    print(f"\nMonitoring job: {job_name}")
    print("Press Ctrl+C to stop monitoring (job will continue running)\n")

    try:
        subprocess.run(
            ["hf", "jobs", "logs", job_name, "--follow"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\nStopped monitoring. Job is still running.")
        print(f"Resume monitoring with: hf jobs logs {job_name} --follow")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch Qwen3-ASR Hebrew training job"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor job after submission"
    )
    parser.add_argument(
        "--job-name",
        default="qwen3-asr-hebrew-training",
        help="Name for the training job"
    )

    args = parser.parse_args()

    # Launch job
    launch_job()

    # Monitor if requested
    if args.monitor:
        monitor_job(args.job_name)
