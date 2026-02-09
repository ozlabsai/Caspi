"""ASR evaluation utilities."""

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from ..qwen_asr.audio import AudioProcessor, normalize_hebrew_text
from ..qwen_asr.client import VLLMClient
from ..qwen_asr.datasets import get_dataset_columns, load_hebrew_dataset
from .metrics import calculate_wer


def evaluate_dataset(
    client: VLLMClient,
    dataset_key: str,
    max_samples: Optional[int] = None,
    model: str = ".",
    language: str = "he"
) -> Optional[dict]:
    """
    Evaluate ASR model on a single dataset.

    Args:
        client: VLLMClient instance
        dataset_key: Dataset key from HEBREW_DATASETS or full dataset name
        max_samples: Maximum number of samples to process
        model: Model name/path for vLLM
        language: Language code

    Returns:
        Dict with evaluation results, or None if evaluation failed
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_key}")
    print('='*70)

    # Load dataset
    try:
        ds = load_hebrew_dataset(dataset_key, streaming=True)
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None

    # Get column names
    columns = get_dataset_columns(dataset_key)
    audio_col = columns["audio_col"]
    text_col = columns["text_col"]

    references = []
    hypotheses = []
    latencies = []

    # Process samples
    count = 0
    for example in tqdm(ds, desc="Processing", total=max_samples):
        if max_samples and count >= max_samples:
            break

        try:
            # Get reference text
            ref_text = example[text_col]
            if not ref_text or not isinstance(ref_text, str):
                continue

            # Get audio
            audio_data = example[audio_col]

            # Save to temp file
            tmp_path = AudioProcessor.save_to_wav(audio_data)

            # Transcribe
            start = time.time()
            hyp_text = client.transcribe(tmp_path, model=model, language=language)
            elapsed = time.time() - start

            # Normalize
            ref_normalized = normalize_hebrew_text(ref_text)
            hyp_normalized = normalize_hebrew_text(hyp_text)

            references.append(ref_normalized)
            hypotheses.append(hyp_normalized)
            latencies.append(elapsed)

            count += 1

            # Clean up temp file
            os.unlink(tmp_path)

        except Exception as e:
            if count == 0:  # Print detailed error for first sample
                print(f"\nError on sample {count}: {e}")
                import traceback
                traceback.print_exc()
            continue

    # Calculate WER
    if len(references) > 0:
        wer_score = calculate_wer(references, hypotheses)
        avg_latency = sum(latencies) / len(latencies)

        print(f"\nResults:")
        print(f"  Samples processed: {len(references)}")
        print(f"  WER: {wer_score:.3f}")
        print(f"  Average latency: {avg_latency:.2f}s")
        print(f"  Total time: {sum(latencies):.1f}s")

        return {
            "dataset": dataset_key,
            "wer": wer_score,
            "samples": len(references),
            "avg_latency": avg_latency
        }
    else:
        print("✗ No valid samples processed")
        return None


def run_benchmark(
    client: VLLMClient,
    dataset_keys: list[str],
    max_samples: Optional[int] = None,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Run benchmark on multiple datasets.

    Args:
        client: VLLMClient instance
        dataset_keys: List of dataset keys to evaluate
        max_samples: Maximum samples per dataset
        output_file: Optional CSV file to save results

    Returns:
        DataFrame with results for all datasets
    """
    print("="*70)
    print("Qwen3-ASR Hebrew Benchmark")
    print("="*70)

    # Test connection
    print("\nTesting connection...")
    if not client.check_health():
        print("✗ Cannot connect to vLLM server")
        return pd.DataFrame()

    model_info = client.get_model_info()
    print(f"✓ Connected to vLLM server")
    print(f"  Model: {model_info.get('id', 'unknown')}")

    # Run evaluations
    results = []
    for dataset_key in dataset_keys:
        result = evaluate_dataset(client, dataset_key, max_samples=max_samples)
        if result:
            results.append(result)

    # Summary
    if results:
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)

        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        print(f"\n{'='*70}")
        print(f"Average WER: {df['wer'].mean():.3f}")
        print(f"Average latency: {df['avg_latency'].mean():.2f}s")
        print(f"Total samples: {df['samples'].sum()}")
        print("="*70)

        # Save results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\n✓ Results saved to {output_file}")

        return df

    return pd.DataFrame()
