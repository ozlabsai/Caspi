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

import sys
from pathlib import Path

# Import from train_hebrew_asr_enhanced
from train_hebrew_asr_enhanced import main, TrainingConfig

# Override default configuration for Round 2
import train_hebrew_asr_enhanced as train_module


def configure_round2():
    """
    Configure Round 2 specific settings.

    This overrides the TrainingConfig defaults with Round 2 optimized values.
    """
    print("="*70)
    print("Round 2 Training Configuration")
    print("="*70)
    print("\nOptimized for: 2x A100 (40GB)")
    print("Strategy: Gradual Unfreezing (B → A)")
    print("\nConfiguration:")
    print("  Hardware: 2x A100 (40GB)")
    print("  Batch size: 2 per GPU")
    print("  Gradient accumulation: 16 steps")
    print("  Effective batch: 64 (2 × 16 × 2 GPUs)")
    print("  Max audio length: 15s (after resegmentation)")
    print("  Epochs: 5")
    print("  Strategy B (Epochs 1-2): Projector + Top 12 LLM")
    print("  Strategy A (Epochs 3-5): + Top 8 Audio layers")
    print("\nLearning Rates:")
    print("  Projector: 2e-4 (epochs 1-2), 1e-4 (epochs 3-5)")
    print("  LLM top: 5e-5 (epochs 1-2), 3e-5 (epochs 3-5)")
    print("  Audio top: 3e-5 (epochs 3-5, when unfrozen)")
    print("  LM head: 1e-4")
    print("\nMemory Optimizations:")
    print("  ✓ BF16 mixed precision")
    print("  ✓ Gradient checkpointing")
    print("  ✓ Length-based bucketing (group_by_length)")
    print("  ✓ Fused AdamW optimizer")
    print("  ✓ Gradient clipping (max_norm=1.0)")
    print("="*70)
    print()


def main():
    """Main entry point for Round 2 training."""
    configure_round2()

    # Verify datasets exist
    print("Checking training data...")
    required_datasets = [
        "ivrit-ai/crowd-transcribe-v5",
        "ivrit-ai/crowd-recital-whisper-training"
    ]

    print("✓ Using datasets:")
    for ds in required_datasets:
        print(f"  - {ds}")

    # Remind about Phase 0
    print("\n" + "="*70)
    print("IMPORTANT: Phase 0 Reminder")
    print("="*70)
    print("\nBefore training, run Phase 0 data quality audit:")
    print("  uv run python scripts/phase0_align_audit.py")
    print("\nIf Phase 0 reports >15% low-quality samples, consider:")
    print("  - Filtering and resegmentation")
    print("  - Data cleaning before training")
    print("\nPhase 0 decision gate will determine if training should proceed.")
    print("="*70)

    response = input("\nHave you reviewed Phase 0 results? (y/n): ")
    if response.lower() != 'y':
        print("\n⚠️  Please run Phase 0 first:")
        print("  uv run python scripts/phase0_align_audit.py")
        print("\nThen review the alignment_report.json and return here.")
        sys.exit(1)

    print("\n✓ Proceeding with Round 2 training...")
    print("="*70)
    print()

    # Start training with enhanced script
    # The train_hebrew_asr_enhanced.py already has Round 2 configuration
    train_module.main()


if __name__ == "__main__":
    main()
