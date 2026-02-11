#!/usr/bin/env python3
"""
Round 2 Training: Gradual Unfreezing Strategy

Optimized for 2x A100 (40GB) with conservative, proven approach:
- Epochs 1-2: Strategy B (projector + top LLM only)
- Epochs 3-5: Strategy A (+ unfreeze top 8 audio layers)

Expected improvement: 12.3% → 10.5-11.0% WER
Cost: ~$48 for 12 hours on 2x A100

Usage:
    uv run python train_round2_gradual.py
"""

import os
import sys
from pathlib import Path

# Import from train_hebrew_asr_enhanced
from train_hebrew_asr_enhanced import main, TrainingConfig

# Override default configuration for Round 2
import train_hebrew_asr_enhanced as train_module


def configure_round2():
    """
    Configure Round 2 specific settings.
    Only prints on rank 0.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        return

    print("="*70)
    print("Round 2 Training Configuration (DDP)")
    print("="*70)
    print("\nOptimized for: 2x A100 (40GB) with DDP")
    print("Strategy: Gradual Unfreezing (B → A)")
    print("\nConfiguration:")
    print("  Hardware: 2x A100 (40GB) DDP")
    print("  Batch size: 8 per GPU")
    print("  Gradient accumulation: 4 steps")
    print("  Effective batch: 64 (8 × 4 × 2 GPUs)")
    print("  Dataloader workers: 8 per process")
    print("  Max audio length: 15s")
    print("  Epochs: 5")
    print("  Strategy B (Epochs 1-2): Projector + Top 12 LLM")
    print("  Strategy A (Epochs 3-5): + Top 8 Audio layers")
    print("="*70)
    print()


def main():
    """Main entry point for Round 2 training."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    configure_round2()

    if local_rank == 0:
        print("\nNon-interactive mode: skipping Phase 0 prompt.")
        print("\n✓ Proceeding with Round 2 training...")
        print("="*70)
        print()

    # Start training with enhanced script
    train_module.main()


if __name__ == "__main__":
    main()
