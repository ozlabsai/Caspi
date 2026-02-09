#!/usr/bin/env python3
"""
Benchmark Qwen3-ASR Hebrew model on Ivrit.AI leaderboard datasets.

This script evaluates the fine-tuned Qwen3-ASR model on the same datasets
used in the Hebrew Transcription Leaderboard:
- ivrit-ai/eval-d1
- ivrit-ai/eval-whatsapp
- ivrit-ai/saspeech
- google/fleurs/he
- mozilla-foundation/common_voice_17_0/he
- imvladikon/hebrew_speech_kan

Metrics: Word Error Rate (WER) for each dataset
"""

import torch
from datasets import load_dataset
from jiwer import wer
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path

# Import Qwen3-ASR (assumes qwen-asr package is installed)
try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    print("Error: qwen-asr package not installed. Install with: pip install qwen-asr")
    exit(1)


DATASETS = [
    ("ivrit-ai/eval-d1", "test", "audio", "transcription"),
    ("ivrit-ai/eval-whatsapp", "test", "audio", "transcription"),
    ("ivrit-ai/saspeech", "test", "audio", "text"),
    ("google/fleurs", "test", "audio", "transcription", {"he"}),  # Hebrew split
    ("mozilla-foundation/common_voice_17_0", "test", "audio", "sentence", {"he"}),
    ("imvladikon/hebrew_speech_kan", "test", "audio", "transcription"),
]


def normalize_hebrew_text(text: str) -> str:
    """Normalize Hebrew text for WER calculation."""
    import re

    # Remove niqqud (Hebrew diacritics)
    text = re.sub(r'[\u0591-\u05C7]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Lowercase (though Hebrew doesn't have case)
    text = text.strip()

    return text


def evaluate_on_dataset(model, dataset_info, max_samples=None):
    """Evaluate model on a single dataset."""
    if len(dataset_info) == 5:
        dataset_name, split, audio_col, text_col, lang_config = dataset_info
    else:
        dataset_name, split, audio_col, text_col = dataset_info
        lang_config = None

    print(f"\nEvaluating on {dataset_name}...")

    # Load dataset
    try:
        if lang_config:
            ds = load_dataset(dataset_name, list(lang_config)[0], split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return None

    # Limit samples if specified
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    print(f"  Processing {len(ds)} samples...")

    references = []
    hypotheses = []

    for i, example in enumerate(tqdm(ds, desc="  Transcribing")):
        try:
            # Get reference text
            ref_text = example[text_col]
            if not ref_text or not isinstance(ref_text, str):
                continue

            # Get audio
            audio_data = example[audio_col]

            # Handle different audio formats
            if isinstance(audio_data, dict):
                # Audio is a dict with 'array' and 'sampling_rate'
                audio_array = audio_data['array']
                sr = audio_data['sampling_rate']
            else:
                # Audio is already an array
                audio_array = audio_data
                sr = 16000  # Assume 16kHz

            # Transcribe with Qwen3-ASR
            result = model.transcribe(
                audio_array,
                language="Hebrew"
            )

            hyp_text = result['text'] if isinstance(result, dict) else result

            # Normalize texts
            ref_normalized = normalize_hebrew_text(ref_text)
            hyp_normalized = normalize_hebrew_text(hyp_text)

            references.append(ref_normalized)
            hypotheses.append(hyp_normalized)

        except Exception as e:
            print(f"    Error on sample {i}: {e}")
            continue

    # Calculate WER
    if len(references) > 0:
        wer_score = wer(references, hypotheses)
        print(f"  WER: {wer_score:.3f}")
        return wer_score
    else:
        print(f"  No valid samples processed")
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR Hebrew model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="ozlabs/qwen3-asr-hebrew",
        help="Path to fine-tuned model (local or HuggingFace)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for quick testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-ASR Hebrew Model Benchmark")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Max samples per dataset: {args.max_samples or 'All'}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    try:
        model = Qwen3ASRModel.from_pretrained(
            args.model_path,
            device_map=args.device,
            dtype=torch.float16 if args.device == "cuda" else torch.float32
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTip: If using local path, ensure all required files are present:")
        print("  - model.safetensors")
        print("  - config.json")
        print("  - tokenizer files")
        print("  - preprocessor_config.json")
        print("  - chat_template.json")
        return

    # Evaluate on each dataset
    results = {}

    for dataset_info in DATASETS:
        dataset_name = dataset_info[0]
        wer_score = evaluate_on_dataset(model, dataset_info, args.max_samples)
        if wer_score is not None:
            results[dataset_name] = wer_score

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for dataset_name, wer_score in results.items():
        print(f"{dataset_name:50s} WER: {wer_score:.3f}")

    if results:
        avg_wer = sum(results.values()) / len(results)
        print(f"\n{'Average WER':50s} WER: {avg_wer:.3f}")

    # Save to CSV
    df = pd.DataFrame([{
        'engine': 'qwen3-asr',
        'model': args.model_path,
        **{ds[0]: results.get(ds[0], None) for ds in DATASETS}
    }])

    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")
    print("\nTo add to leaderboard, format your results as:")
    print("  engine,model," + ",".join([ds[0] for ds in DATASETS]))


if __name__ == "__main__":
    main()
