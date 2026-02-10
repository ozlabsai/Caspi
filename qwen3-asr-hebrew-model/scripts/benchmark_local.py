#!/usr/bin/env python3
"""
Benchmark Qwen3-ASR Hebrew model using local inference with qwen-asr package.

Usage:
    uv run python scripts/benchmark_local.py --model Qwen/Qwen3-ASR-1.7B --max-samples 100
    uv run python scripts/benchmark_local.py --model ./qwen3-asr-hebrew --max-samples 100
"""

import argparse
import io
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset, Audio
from jiwer import wer as calculate_wer
from tqdm import tqdm


# Dataset configurations
HEBREW_DATASETS = {
    "eval-d1": {
        "name": "ivrit-ai/eval-d1",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "config": None,
    },
    "eval-whatsapp": {
        "name": "ivrit-ai/eval-whatsapp",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "config": None,
    },
    "saspeech": {
        "name": "upai-inc/saspeech",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "config": None,
    },
    "hebrew-speech-kan": {
        "name": "imvladikon/hebrew_speech_kan",
        "split": "validation",
        "audio_col": "audio",
        "text_col": "sentence",
        "config": None,
    },
}


def normalize_hebrew_text(text: str) -> str:
    """Normalize Hebrew text for WER calculation."""
    # Remove niqqud (Hebrew diacritics)
    text = re.sub(r'[\u0591-\u05C7]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def decode_audio(audio_data, target_sr: int = 16000):
    """Decode audio from HuggingFace dataset format."""
    if isinstance(audio_data, dict):
        if 'bytes' in audio_data and audio_data['bytes'] is not None:
            return librosa.load(io.BytesIO(audio_data['bytes']), sr=target_sr)
        elif 'path' in audio_data and audio_data['path'] is not None:
            return librosa.load(audio_data['path'], sr=target_sr)
        elif 'array' in audio_data and audio_data['array'] is not None:
            arr = np.array(audio_data['array'])
            sr = audio_data.get('sampling_rate', target_sr)
            if sr != target_sr:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
            return arr, target_sr
    raise ValueError(f"Cannot decode audio from: {type(audio_data)}")


def transcribe_with_chunking(model, audio_array, sr, chunk_duration: float = 30.0):
    """
    Transcribe audio with chunking for long files.

    Args:
        model: Qwen3ASRModel instance
        audio_array: Audio samples as numpy array
        sr: Sample rate
        chunk_duration: Max duration per chunk in seconds

    Returns:
        Concatenated transcription text
    """
    duration = len(audio_array) / sr

    # If audio is short enough, transcribe directly
    if duration <= chunk_duration:
        results = model.transcribe(audio=(audio_array, sr), language=None)
        return results[0].text if results else ""

    # Chunk the audio
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    for start in range(0, len(audio_array), chunk_samples):
        end = min(start + chunk_samples, len(audio_array))
        chunks.append(audio_array[start:end])

    # Transcribe each chunk
    transcriptions = []
    for chunk in chunks:
        results = model.transcribe(audio=(chunk, sr), language=None)
        if results and results[0].text:
            transcriptions.append(results[0].text)

    return " ".join(transcriptions)


def load_hebrew_dataset(dataset_key: str, streaming: bool = True):
    """Load a Hebrew ASR dataset without automatic audio decoding."""
    config = HEBREW_DATASETS[dataset_key]
    dataset_name = config["name"]
    dataset_split = config["split"]
    dataset_config = config["config"]

    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=streaming)
    else:
        ds = load_dataset(dataset_name, split=dataset_split, streaming=streaming)

    # Disable automatic audio decoding (avoids torchcodec dependency)
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def load_model(model_path: str, use_vllm: bool = False, gpu_id: int = 0):
    """Load Qwen3-ASR model."""
    # Import from the installed qwen_asr package
    from qwen_asr import Qwen3ASRModel

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_path}")
    print(f"Backend: {'vLLM' if use_vllm else 'Transformers'}")
    print(f"Device: {device}")

    if use_vllm:
        model = Qwen3ASRModel.LLM(
            model=model_path,
            gpu_memory_utilization=0.8,
            max_inference_batch_size=32,
            max_new_tokens=4096,  # Increased for long audio
        )
    else:
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            max_inference_batch_size=32,
            max_new_tokens=4096,  # Increased for long audio
        )

    print("Model loaded successfully")
    return model


def evaluate_dataset(
    model,
    dataset_key: str,
    max_samples: int = None,
) -> dict:
    """
    Evaluate model on a single dataset.

    Returns:
        Dict with evaluation results
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_key}")
    print('='*70)

    # Load dataset
    try:
        ds = load_hebrew_dataset(dataset_key, streaming=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    # Get column names from config
    config = HEBREW_DATASETS[dataset_key]
    audio_col = config["audio_col"]
    text_col = config["text_col"]

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

            # Get and decode audio
            audio_data = example[audio_col]
            audio_array, sr = decode_audio(audio_data)
            # Ensure mono
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=0)

            # Transcribe with chunking for long audio
            start = time.time()
            hyp_text = transcribe_with_chunking(model, audio_array, sr, chunk_duration=30.0)
            elapsed = time.time() - start

            # Normalize
            ref_normalized = normalize_hebrew_text(ref_text)
            hyp_normalized = normalize_hebrew_text(hyp_text)

            references.append(ref_normalized)
            hypotheses.append(hyp_normalized)
            latencies.append(elapsed)

            count += 1

            # Print progress every 10 samples
            if count % 10 == 0:
                avg_lat = sum(latencies) / len(latencies)
                print(f"  Processed {count} samples, avg latency: {avg_lat:.2f}s")

        except Exception as e:
            if count == 0:
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
            "avg_latency": avg_latency,
            "total_time": sum(latencies),
        }
    else:
        print("No valid samples processed")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-ASR Hebrew model (local inference)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(HEBREW_DATASETS.keys()),
        help=f"Dataset keys to evaluate. Available: {list(HEBREW_DATASETS.keys())}"
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
        help="Output CSV file"
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM backend (faster, requires pip install qwen-asr[vllm])"
    )
    parser.add_argument(
        "--language",
        default="Hebrew",
        help="Language hint for ASR (default: Hebrew)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use (default: 0)"
    )

    args = parser.parse_args()

    print("="*70)
    print("Qwen3-ASR Hebrew Benchmark (Local Inference)")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Backend: {'vLLM' if args.vllm else 'Transformers'}")
    print(f"GPU: {args.gpu}")
    print("="*70)

    # Load model
    model = load_model(args.model, use_vllm=args.vllm, gpu_id=args.gpu)

    # Run evaluations
    results = []
    for dataset_key in args.datasets:
        if dataset_key not in HEBREW_DATASETS:
            print(f"Warning: Unknown dataset '{dataset_key}', skipping")
            continue

        result = evaluate_dataset(
            model,
            dataset_key,
            max_samples=args.max_samples,
        )
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
        print(f"Total time: {df['total_time'].sum():.1f}s")
        print("="*70)

        # Save results
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

        return df

    return pd.DataFrame()


if __name__ == "__main__":
    main()
