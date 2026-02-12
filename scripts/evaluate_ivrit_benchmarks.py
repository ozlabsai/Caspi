#!/usr/bin/env python3
"""
Evaluate model on all 6 ivrit.ai benchmark test sets.

Reproduces the exact evaluation from:
https://huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard

Test sets:
1. eval-d1          - Clean, high-quality recordings
2. WhatsApp         - Noisy, real-world voice messages
3. saspeech         - Spontaneous Hebrew speech
4. FLEURS           - Formal reading (part of larger multilingual dataset)
5. CommonVoice      - Diverse accents and recording conditions
6. KAN              - Israeli public broadcasting

Usage:
    # Evaluate single model
    uv run python scripts/evaluate_ivrit_benchmarks.py \
        --model ./qwen3-asr-hebrew-round2.5 \
        --output results_round2.5.json

    # Compare multiple models
    uv run python scripts/evaluate_ivrit_benchmarks.py \
        --model ./qwen3-asr-hebrew-round1 \
        --model ./qwen3-asr-hebrew-round2.5 \
        --compare \
        --output comparison.json

Output:
    {
        "model_name": "qwen3-asr-hebrew-round2.5",
        "results": {
            "eval-d1": {"wer": 0.065, "cer": 0.032, "samples": 150},
            "whatsapp": {"wer": 0.089, "cer": 0.045, "samples": 200},
            ...
        },
        "average_wer": 0.092,
        "vs_sota": {
            "eval-d1": "+1.4%",  # 6.5% vs 5.1% SOTA
            ...
        }
    }
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import evaluate
import numpy as np


# ivrit.ai test set configurations
IVRIT_TEST_SETS = {
    "eval-d1": {
        "dataset": "ivrit-ai/eval-d1",
        "split": "test",
        "description": "Clean, high-quality recordings",
    },
    "whatsapp": {
        "dataset": "ivrit-ai/whatsapp-test",
        "split": "test",
        "description": "Noisy, real-world voice messages",
    },
    "saspeech": {
        "dataset": "ivrit-ai/saspeech-test",
        "split": "test",
        "description": "Spontaneous Hebrew speech",
    },
    "fleurs": {
        "dataset": "google/fleurs",
        "split": "test",
        "subset": "he_il",
        "description": "Formal reading (multilingual)",
    },
    "commonvoice": {
        "dataset": "mozilla-foundation/common_voice_16_1",
        "split": "test",
        "subset": "he",
        "description": "Diverse accents and conditions",
    },
    "kan": {
        "dataset": "ivrit-ai/kan-test",
        "split": "test",
        "description": "Israeli public broadcasting",
    },
}

# SOTA results from ivrit.ai leaderboard (as of 2025-05-13)
# Source: https://huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard/resolve/main/benchmark.csv
SOTA_RESULTS = {
    "eval-d1": 0.051,      # ivrit-ai/whisper-large-v3-ct2-20250513
    "whatsapp": 0.072,     # ivrit-ai/whisper-large-v3-ct2-20250513
    "saspeech": 0.064,     # ivrit-ai/whisper-large-v3-ct2-20250513
    "fleurs": 0.174,       # ivrit-ai/whisper-large-v3-ct2-20250513
    "commonvoice": 0.149,  # ivrit-ai/whisper-large-v3-ct2-20250513
    "kan": 0.081,          # ivrit-ai/whisper-large-v3-ct2-20250513
}


class IvritBenchmarkEvaluator:
    """Evaluate ASR models on ivrit.ai benchmark test sets."""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize evaluator.

        Args:
            model_path: Path to model directory or HuggingFace model ID
            device: Device to run on ("cuda", "cpu", or "auto")
        """
        print(f"\n{'='*70}")
        print(f"Loading model: {model_path}")
        print(f"{'='*70}")

        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        print(f"✓ Model loaded on {self.device}")
        print(f"  Total params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*70}\n")

        # Load metrics
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def transcribe_batch(self, audio_arrays: List[np.ndarray], sampling_rate: int = 16000) -> List[str]:
        """
        Transcribe batch of audio samples.

        Args:
            audio_arrays: List of audio numpy arrays
            sampling_rate: Audio sampling rate

        Returns:
            List of transcriptions
        """
        # Process audio features
        inputs = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        input_features = inputs.input_features.to(self.device)
        if hasattr(input_features, "dtype") and input_features.dtype != self.model.dtype:
            input_features = input_features.to(self.model.dtype)

        # Generate transcriptions
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(self.device == "cuda")):
            generated_ids = self.model.generate(
                input_features,
                max_length=225,
                num_beams=1,  # Greedy decoding for speed
            )

        # Decode
        transcriptions = self.processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return transcriptions

    def evaluate_dataset(
        self,
        test_set_name: str,
        batch_size: int = 8,
        max_samples: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate on a single test set.

        Args:
            test_set_name: Name of test set (e.g., "eval-d1")
            batch_size: Batch size for inference
            max_samples: Maximum samples to evaluate (for testing)

        Returns:
            Dictionary with WER, CER, and metadata
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {test_set_name}")
        print(f"{'='*70}")

        test_config = IVRIT_TEST_SETS[test_set_name]
        print(f"Dataset: {test_config['dataset']}")
        print(f"Description: {test_config['description']}")

        # Load dataset
        if "subset" in test_config:
            dataset = load_dataset(
                test_config["dataset"],
                test_config["subset"],
                split=test_config["split"],
            )
        else:
            dataset = load_dataset(
                test_config["dataset"],
                split=test_config["split"],
            )

        # Limit samples for testing
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            print(f"⚠️  Limited to {max_samples} samples (testing mode)")

        print(f"Samples: {len(dataset)}")

        # Evaluate in batches
        all_predictions = []
        all_references = []

        for i in tqdm(range(0, len(dataset), batch_size), desc=f"  {test_set_name}"):
            batch = dataset[i : i + batch_size]

            # Extract audio and references
            audio_arrays = [sample["array"] for sample in batch["audio"]]
            references = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]

            # Transcribe
            try:
                predictions = self.transcribe_batch(audio_arrays)
                all_predictions.extend(predictions)
                all_references.extend(references)
            except Exception as e:
                print(f"  ✗ Error on batch {i}: {e}")
                # Add empty predictions for failed batches
                all_predictions.extend([""] * len(audio_arrays))
                all_references.extend(references)

        # Filter out empty predictions
        valid_pairs = [(p, r) for p, r in zip(all_predictions, all_references) if p.strip()]
        if len(valid_pairs) < len(all_predictions):
            print(f"  ⚠️  {len(all_predictions) - len(valid_pairs)} failed transcriptions")

        pred_str, ref_str = zip(*valid_pairs) if valid_pairs else ([], [])

        # Compute metrics
        if pred_str:
            wer = self.wer_metric.compute(predictions=pred_str, references=ref_str)
            cer = self.cer_metric.compute(predictions=pred_str, references=ref_str)
        else:
            wer, cer = 1.0, 1.0  # 100% error if all failed

        # Compare to SOTA
        sota_wer = SOTA_RESULTS.get(test_set_name, None)
        vs_sota = f"{(wer - sota_wer)*100:+.1f}%" if sota_wer else "N/A"

        print(f"\n  Results:")
        print(f"    WER: {wer*100:.2f}%")
        print(f"    CER: {cer*100:.2f}%")
        print(f"    vs SOTA: {vs_sota} (SOTA: {sota_wer*100:.1f}%)" if sota_wer else "")
        print(f"    Valid samples: {len(valid_pairs)}/{len(all_predictions)}")

        return {
            "wer": wer,
            "cer": cer,
            "samples_total": len(all_predictions),
            "samples_valid": len(valid_pairs),
            "sota_wer": sota_wer,
            "vs_sota": vs_sota,
            "test_set_config": test_config,
        }

    def evaluate_all(
        self,
        batch_size: int = 8,
        max_samples: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate on all 6 ivrit.ai test sets.

        Args:
            batch_size: Batch size for inference
            max_samples: Maximum samples per test set (for testing)

        Returns:
            Complete evaluation results
        """
        results = {}

        for test_set_name in IVRIT_TEST_SETS.keys():
            try:
                results[test_set_name] = self.evaluate_dataset(
                    test_set_name,
                    batch_size=batch_size,
                    max_samples=max_samples,
                )
            except Exception as e:
                print(f"\n✗ Failed to evaluate {test_set_name}: {e}")
                results[test_set_name] = {
                    "error": str(e),
                    "wer": 1.0,
                    "cer": 1.0,
                }

        # Compute average WER
        valid_wers = [r["wer"] for r in results.values() if "error" not in r]
        average_wer = np.mean(valid_wers) if valid_wers else 1.0

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Model: {self.model_path}")
        print(f"\nPer-dataset results:")
        for name, result in results.items():
            if "error" not in result:
                print(f"  {name:15s}: {result['wer']*100:6.2f}% WER (vs SOTA: {result['vs_sota']})")
            else:
                print(f"  {name:15s}: ERROR - {result['error']}")

        print(f"\n  Average WER: {average_wer*100:.2f}%")
        print(f"  SOTA Average: {np.mean(list(SOTA_RESULTS.values()))*100:.2f}%")
        print(f"{'='*70}\n")

        return {
            "model_path": self.model_path,
            "results": results,
            "average_wer": average_wer,
            "average_cer": np.mean([r["cer"] for r in results.values() if "error" not in r]),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on ivrit.ai benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        action="append",
        help="Model path or HuggingFace ID (can specify multiple with --model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per test set (for quick testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models (requires multiple --model args)",
    )

    args = parser.parse_args()

    # Evaluate all models
    all_results = []

    for model_path in args.model:
        evaluator = IvritBenchmarkEvaluator(model_path, device=args.device)
        results = evaluator.evaluate_all(
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        all_results.append(results)

    # Save results
    output = {
        "models": all_results,
        "sota_baseline": SOTA_RESULTS,
    }

    # Add comparison if multiple models
    if args.compare and len(all_results) > 1:
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}\n")

        for test_set_name in IVRIT_TEST_SETS.keys():
            print(f"{test_set_name}:")
            for result in all_results:
                model_name = Path(result["model_path"]).name
                wer = result["results"][test_set_name].get("wer", 1.0) * 100
                print(f"  {model_name:40s}: {wer:6.2f}% WER")
            print()

        # Average comparison
        print("Average WER:")
        for result in all_results:
            model_name = Path(result["model_path"]).name
            avg_wer = result["average_wer"] * 100
            print(f"  {model_name:40s}: {avg_wer:6.2f}%")

    # Save to JSON
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
