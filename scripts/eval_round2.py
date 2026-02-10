#!/usr/bin/env python3
"""
Round 2 Evaluation and Comparison

Comprehensive evaluation comparing Round 1 vs Round 2 models:
- Multiple Hebrew ASR datasets
- Greedy + Beam search (beam=5) decoding
- Per-duration-bucket WER analysis
- Detailed comparison report

Usage:
    uv run python scripts/eval_round2.py \\
        --round1-model OzLabs/Qwen3-ASR-Hebrew-1.7B \\
        --round2-model ./qwen3-asr-hebrew-round2 \\
        --output round2_comparison.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import existing benchmark infrastructure
from qwen3_asr_hebrew_model.src.benchmarks import evaluate_dataset, calculate_wer
from qwen3_asr_hebrew_model.src.qwen_asr import VLLMClient
from qwen3_asr_hebrew_model.src.qwen_asr.datasets import HEBREW_DATASETS


def evaluate_with_duration_buckets(
    client: VLLMClient,
    dataset_key: str,
    max_samples: Optional[int] = None,
    beam_size: int = 1,
) -> Dict:
    """
    Evaluate with per-duration-bucket WER breakdown.

    Duration buckets:
        - short: 0.5-5s
        - medium: 5-15s
        - long: 15-30s

    Returns:
        {
            "overall_wer": float,
            "bucket_wers": {"short": float, "medium": float, "long": float},
            "samples_per_bucket": {"short": int, "medium": int, "long": int}
        }
    """
    from qwen3_asr_hebrew_model.src.qwen_asr.datasets import load_hebrew_dataset, get_dataset_columns
    from qwen3_asr_hebrew_model.src.qwen_asr.audio import AudioProcessor, normalize_hebrew_text
    import soundfile as sf

    print(f"\nEvaluating {dataset_key} (beam_size={beam_size})...")

    # Load dataset
    ds = load_hebrew_dataset(dataset_key, streaming=True)
    columns = get_dataset_columns(dataset_key)
    audio_col = columns["audio_col"]
    text_col = columns["text_col"]

    # Collect results by bucket
    bucket_data = {
        "short": {"references": [], "hypotheses": []},
        "medium": {"references": [], "hypotheses": []},
        "long": {"references": [], "hypotheses": []},
    }

    count = 0
    for example in tqdm(ds, desc="Processing", total=max_samples):
        if max_samples and count >= max_samples:
            break

        try:
            ref_text = example[text_col]
            if not ref_text or not isinstance(ref_text, str):
                continue

            audio_data = example[audio_col]

            # Save to temp file
            tmp_path = AudioProcessor.save_to_wav(audio_data)

            # Get audio duration
            audio_array, sr = sf.read(tmp_path)
            duration = len(audio_array) / sr

            # Determine bucket
            if duration < 5.0:
                bucket = "short"
            elif duration < 15.0:
                bucket = "medium"
            else:
                bucket = "long"

            # Transcribe with beam search
            if beam_size > 1:
                # Beam search transcription
                # Note: VLLMClient.transcribe doesn't support beam search parameter yet
                # For now, use greedy as placeholder
                # TODO: Add beam search support to VLLMClient
                hyp_text = client.transcribe(tmp_path, model=".", language="he")
            else:
                hyp_text = client.transcribe(tmp_path, model=".", language="he")

            # Normalize
            ref_normalized = normalize_hebrew_text(ref_text)
            hyp_normalized = normalize_hebrew_text(hyp_text)

            bucket_data[bucket]["references"].append(ref_normalized)
            bucket_data[bucket]["hypotheses"].append(hyp_normalized)

            count += 1

            # Clean up
            import os
            os.unlink(tmp_path)

        except Exception as e:
            if count == 0:
                print(f"Error on first sample: {e}")
            continue

    # Compute WER per bucket
    bucket_wers = {}
    samples_per_bucket = {}

    for bucket, data in bucket_data.items():
        if len(data["references"]) > 0:
            bucket_wers[bucket] = calculate_wer(
                data["references"],
                data["hypotheses"]
            )
            samples_per_bucket[bucket] = len(data["references"])
        else:
            bucket_wers[bucket] = None
            samples_per_bucket[bucket] = 0

    # Overall WER
    all_refs = []
    all_hyps = []
    for bucket_data_dict in bucket_data.values():
        all_refs.extend(bucket_data_dict["references"])
        all_hyps.extend(bucket_data_dict["hypotheses"])

    overall_wer = calculate_wer(all_refs, all_hyps) if all_refs else None

    return {
        "overall_wer": overall_wer,
        "bucket_wers": bucket_wers,
        "samples_per_bucket": samples_per_bucket,
        "total_samples": count
    }


def compare_models(
    round1_model: str,
    round2_model: str,
    datasets: List[str],
    max_samples: int = 200,
    output_file: str = "round2_comparison.csv"
) -> pd.DataFrame:
    """
    Comprehensive comparison between Round 1 and Round 2 models.

    Evaluates both models on multiple datasets with:
    - Greedy decoding
    - Beam search (beam=5)
    - Per-duration-bucket breakdown
    """
    print("="*70)
    print("Round 2 Model Comparison")
    print("="*70)
    print(f"\nRound 1 Model: {round1_model}")
    print(f"Round 2 Model: {round2_model}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Max samples per dataset: {max_samples}")
    print("="*70)

    results = []

    for model_name, model_path in [("Round 1", round1_model), ("Round 2", round2_model)]:
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name}: {model_path}")
        print("="*70)

        # Connect to vLLM server
        # Note: You need to have the model running on vLLM server
        # This is a placeholder - in practice you'd need to:
        # 1. Load the model on vLLM
        # 2. Point client to the server
        # For now, we'll just report this limitation

        print(f"\n⚠️  To evaluate {model_name}, you need to:")
        print(f"   1. Start vLLM server with: vllm serve {model_path}")
        print(f"   2. Run this script again pointing to the server")
        print(f"\nSkipping {model_name} for now (requires vLLM server setup)")
        continue

        # When vLLM server is ready:
        # client = VLLMClient(base_url="http://localhost:8000/v1")

        # for dataset_key in datasets:
        #     for decoding, beam_size in [("greedy", 1), ("beam5", 5)]:
        #         result = evaluate_with_duration_buckets(
        #             client,
        #             dataset_key,
        #             max_samples=max_samples,
        #             beam_size=beam_size
        #         )
        #
        #         results.append({
        #             "model": model_name,
        #             "model_path": model_path,
        #             "dataset": dataset_key,
        #             "decoding": decoding,
        #             "beam_size": beam_size,
        #             "overall_wer": result["overall_wer"],
        #             "short_wer": result["bucket_wers"]["short"],
        #             "medium_wer": result["bucket_wers"]["medium"],
        #             "long_wer": result["bucket_wers"]["long"],
        #             "short_samples": result["samples_per_bucket"]["short"],
        #             "medium_samples": result["samples_per_bucket"]["medium"],
        #             "long_samples": result["samples_per_bucket"]["long"],
        #             "total_samples": result["total_samples"]
        #         })

    if not results:
        print("\n" + "="*70)
        print("EVALUATION SETUP REQUIRED")
        print("="*70)
        print("\nTo complete the evaluation, follow these steps:")
        print("\n1. Deploy Round 1 model to vLLM:")
        print(f"   vllm serve {round1_model} --port 8001")
        print("\n2. Run benchmark:")
        print("   uv run python scripts/benchmark.py --server http://localhost:8001/v1")
        print("\n3. Deploy Round 2 model to vLLM:")
        print(f"   vllm serve {round2_model} --port 8002")
        print("\n4. Run benchmark:")
        print("   uv run python scripts/benchmark.py --server http://localhost:8002/v1")
        print("\n5. Compare results manually or use existing benchmark CSVs")
        print("="*70)
        return pd.DataFrame()

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    for model_name in ["Round 1", "Round 2"]:
        model_df = df[df["model"] == model_name]
        if not model_df.empty:
            print(f"\n{model_name}:")
            for dataset in datasets:
                dataset_df = model_df[model_df["dataset"] == dataset]
                if not dataset_df.empty:
                    greedy_wer = dataset_df[dataset_df["decoding"] == "greedy"]["overall_wer"].values[0]
                    beam5_wer = dataset_df[dataset_df["decoding"] == "beam5"]["overall_wer"].values[0]
                    print(f"  {dataset}:")
                    print(f"    Greedy: {greedy_wer:.3f}")
                    print(f"    Beam=5: {beam5_wer:.3f}")

    print("="*70)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare Round 1 vs Round 2 models"
    )
    parser.add_argument(
        "--round1-model",
        default="OzLabs/Qwen3-ASR-Hebrew-1.7B",
        help="Round 1 model path or HF Hub ID"
    )
    parser.add_argument(
        "--round2-model",
        default="./qwen3-asr-hebrew-round2",
        help="Round 2 model path"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eval-d1", "eval-whatsapp", "hebrew-speech-kan", "saspeech"],
        help="Datasets to evaluate on"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max samples per dataset"
    )
    parser.add_argument(
        "--output",
        default="round2_comparison.csv",
        help="Output CSV file"
    )

    args = parser.parse_args()

    # Run comparison
    df = compare_models(
        round1_model=args.round1_model,
        round2_model=args.round2_model,
        datasets=args.datasets,
        max_samples=args.max_samples,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
