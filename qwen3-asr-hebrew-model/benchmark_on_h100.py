#!/usr/bin/env python3
"""
Benchmark Qwen3-ASR Hebrew model on H100 using vLLM server.
"""

from openai import OpenAI
from datasets import load_dataset
from jiwer import wer
import tempfile
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import re
import time

# Datasets to evaluate
DATASETS = [
    ("ivrit-ai/eval-d1", "test", "audio", "text", None),
    ("ivrit-ai/eval-whatsapp", "test", "audio", "text", None),
    ("mozilla-foundation/common_voice_17_0", "test", "audio", "sentence", {"he"}),
]

def normalize_hebrew_text(text: str) -> str:
    """Normalize Hebrew text for WER calculation."""
    # Remove niqqud (Hebrew diacritics)
    text = re.sub(r'[\u0591-\u05C7]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def evaluate_dataset(client, dataset_name, split, audio_col, text_col, config=None, max_samples=None):
    """Evaluate on a single dataset."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_name}")
    print('='*70)

    # Load dataset
    try:
        if config:
            ds = load_dataset(dataset_name, list(config)[0], split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None

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

            # Handle different audio data formats
            if hasattr(audio_data, 'decode'):
                # torchcodec AudioDecoder object - decode it
                decoded = audio_data.decode()
                # Get audio tensor and convert to numpy
                audio_tensor = decoded['audio']
                # Shape is usually (channels, samples), soundfile expects (samples,) or (samples, channels)
                if audio_tensor.ndim == 2:
                    audio_array = audio_tensor[0].numpy()  # Take first channel
                else:
                    audio_array = audio_tensor.numpy()
                sr = int(decoded['sample_rate'])
            elif isinstance(audio_data, dict):
                audio_array = audio_data.get('array')
                sr = audio_data.get('sampling_rate', 16000)
            elif isinstance(audio_data, (list, tuple)) and len(audio_data) >= 2:
                # Format: (array, sampling_rate) or [array, sampling_rate]
                audio_array = audio_data[0]
                sr = audio_data[1]
            elif hasattr(audio_data, 'array') and hasattr(audio_data, 'sampling_rate'):
                # Object with attributes
                audio_array = audio_data.array
                sr = audio_data.sampling_rate
            else:
                # Assume it's just the array
                audio_array = audio_data
                sr = 16000

            if audio_array is None:
                print(f"\nSkipping sample - no audio array")
                continue

            # Ensure audio is 1D for soundfile (mono)
            import numpy as np
            if isinstance(audio_array, np.ndarray):
                if audio_array.ndim > 1:
                    # If stereo or multi-channel, take first channel or flatten
                    audio_array = audio_array.flatten() if audio_array.shape[0] > audio_array.shape[1] else audio_array[0]
                # Ensure it's 1D
                audio_array = audio_array.squeeze()

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_array, sr)
                tmp_path = tmp.name

            # Transcribe
            start = time.time()
            with open(tmp_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model=".",
                    file=f,
                    language="he"
                )
            elapsed = time.time() - start

            hyp_text = result.text

            # Normalize
            ref_normalized = normalize_hebrew_text(ref_text)
            hyp_normalized = normalize_hebrew_text(hyp_text)

            references.append(ref_normalized)
            hypotheses.append(hyp_normalized)
            latencies.append(elapsed)

            count += 1

            # Clean up temp file
            import os
            os.unlink(tmp_path)

        except Exception as e:
            if count == 0:  # Print detailed error for first sample
                print(f"\nError on sample {count}: {e}")
                print(f"Audio data type: {type(audio_data)}")
                if isinstance(audio_data, dict):
                    print(f"Audio dict keys: {audio_data.keys()}")
                try:
                    print(f"Audio array shape: {audio_array.shape}")
                    print(f"Audio array dtype: {audio_array.dtype}")
                except:
                    pass
                import traceback
                traceback.print_exc()
            continue

    # Calculate WER
    if len(references) > 0:
        wer_score = wer(references, hypotheses)
        avg_latency = sum(latencies) / len(latencies)

        print(f"\nResults:")
        print(f"  Samples processed: {len(references)}")
        print(f"  WER: {wer_score:.3f}")
        print(f"  Average latency: {avg_latency:.2f}s")
        print(f"  Total time: {sum(latencies):.1f}s")

        return {
            "dataset": dataset_name,
            "wer": wer_score,
            "samples": len(references),
            "avg_latency": avg_latency
        }
    else:
        print("✗ No valid samples processed")
        return None

def main():
    print("="*70)
    print("Qwen3-ASR Hebrew Benchmark on Lambda Labs H100")
    print("="*70)

    # Connect to vLLM server
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    # Test connection
    print("\nTesting connection...")
    try:
        models = client.models.list()
        print(f"✓ Connected to vLLM server")
        print(f"  Model: {models.data[0].id}")
    except Exception as e:
        print(f"✗ Cannot connect to vLLM: {e}")
        return

    # Run evaluations
    results = []

    for dataset_info in DATASETS:
        result = evaluate_dataset(client, *dataset_info, max_samples=100)
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
        df.to_csv("benchmark_results_h100.csv", index=False)
        print("\n✓ Results saved to benchmark_results_h100.csv")

if __name__ == "__main__":
    main()
