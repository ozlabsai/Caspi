#!/usr/bin/env python3
"""
Quick evaluation script for a single test set.

Useful for rapid iteration during development.

Usage:
    # Evaluate on eval-d1 (default)
    uv run python scripts/quick_eval.py --model ./qwen3-asr-hebrew-round2.5

    # Evaluate on specific test set with limited samples
    uv run python scripts/quick_eval.py \
        --model ./qwen3-asr-hebrew-round2.5 \
        --test-set whatsapp \
        --max-samples 50

    # Compare two models quickly
    uv run python scripts/quick_eval.py \
        --model ./qwen3-asr-hebrew-round1 \
        --model ./qwen3-asr-hebrew-round2.5 \
        --max-samples 100
"""

import sys
from pathlib import Path

# Import from the full evaluation script
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_ivrit_benchmarks import IvritBenchmarkEvaluator, IVRIT_TEST_SETS, SOTA_RESULTS

import argparse


def main():
    parser = argparse.ArgumentParser(description="Quick evaluation on single test set")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        action="append",
        help="Model path (can specify multiple)",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="eval-d1",
        choices=list(IVRIT_TEST_SETS.keys()),
        help="Test set to evaluate on (default: eval-d1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max samples to evaluate (default: 100 for quick eval)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("QUICK EVALUATION")
    print(f"{'='*70}")
    print(f"Test set: {args.test_set}")
    print(f"Max samples: {args.max_samples}")
    print(f"Models: {len(args.model)}")
    print(f"{'='*70}\n")

    # Evaluate all models
    results = []

    for model_path in args.model:
        evaluator = IvritBenchmarkEvaluator(model_path)
        result = evaluator.evaluate_dataset(
            args.test_set,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        results.append({
            "model_path": model_path,
            "result": result,
        })

    # Comparison if multiple models
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}\n")

        for r in results:
            model_name = Path(r["model_path"]).name
            wer = r["result"]["wer"] * 100
            vs_sota = r["result"]["vs_sota"]
            print(f"{model_name:40s}: {wer:6.2f}% WER (vs SOTA: {vs_sota})")

        # Show best model
        best = min(results, key=lambda x: x["result"]["wer"])
        best_name = Path(best["model_path"]).name
        print(f"\n✓ Best: {best_name}")


if __name__ == "__main__":
    main()
