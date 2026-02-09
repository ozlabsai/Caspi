#!/usr/bin/env python3
"""
Benchmark Qwen3-ASR Hebrew model on evaluation datasets.

Usage:
    python scripts/benchmark.py --server http://localhost:8000/v1 --max-samples 100
"""

import argparse

from src.benchmarks import run_benchmark
from src.qwen_asr import VLLMClient
from src.qwen_asr.datasets import HEBREW_DATASETS


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-ASR Hebrew model"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8000/v1",
        help="vLLM server URL (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(HEBREW_DATASETS.keys()),
        help="Dataset keys to evaluate (default: all)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (default: all)"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.csv",
        help="Output CSV file (default: benchmark_results.csv)"
    )

    args = parser.parse_args()

    # Create client
    client = VLLMClient(base_url=args.server)

    # Run benchmark
    run_benchmark(
        client=client,
        dataset_keys=args.datasets,
        max_samples=args.max_samples,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
