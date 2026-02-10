#!/usr/bin/env python3
"""
Evaluate ASR model on Hebrew datasets.
Based on ivrit-ai evaluation framework: https://github.com/ivrit-ai/asr-training

Usage:
    uv run python scripts/evaluate_model.py \
        --engine scripts/engines/qwen_asr_engine.py \
        --model OzLabs/Qwen3-ASR-Hebrew-1.7B \
        --dataset ivrit-ai/eval-d1:test:text \
        --device cuda:0
"""

import argparse
import concurrent.futures
import json
import os

import datasets
import jiwer
import pandas
import whisper.normalizers
from hebrew import Hebrew
from tqdm import tqdm


def clean_some_unicode_from_text(text):
    chars_to_remove = "\u061C"  # Arabic letter mark
    chars_to_remove += "\u200B\u200C\u200D"  # Zero-width space, non-joiner, joiner
    chars_to_remove += "\u200E\u200F"  # LTR and RTL marks
    chars_to_remove += "\u202A\u202B\u202C\u202D\u202E"  # LTR/RTL embedding, pop, override
    chars_to_remove += "\u2066\u2067\u2068\u2069"  # Isolate controls
    chars_to_remove += "\uFEFF"  # Zero-width no-break space
    return text.translate({ord(c): None for c in chars_to_remove})


def remove_niqqud(text: str):
    """Remove niqqud from Hebrew text."""
    return Hebrew(text).no_niqqud().string


class HebrewTextNormalizer:
    def __init__(self):
        self.whisper_normalizer = whisper.normalizers.BasicTextNormalizer()

    def __call__(self, text):
        text = clean_some_unicode_from_text(text)
        text = remove_niqqud(text)
        text = text.replace('"', "").replace("'", "")
        return self.whisper_normalizer(text)


def process_entry(args):
    i, entry, transcribe_fn, text_column, normalizer, benchmark_timing = args

    raw_ref_text = entry[text_column]

    # transcribe_fn returns (text, transcription_time, audio_duration)
    result = transcribe_fn(entry)
    if len(result) == 3:
        raw_eval_text, transcription_time, audio_duration = result
    else:
        raw_eval_text, transcription_time = result
        # Try to get duration from decoded audio
        audio_data = entry.get("audio", {})
        if isinstance(audio_data, dict) and "array" in audio_data and audio_data["array"] is not None:
            audio_duration = len(audio_data["array"]) / audio_data.get("sampling_rate", 16000)
        else:
            audio_duration = 0.0

    # If benchmark_timing is set, run additional iterations and collect all times
    if benchmark_timing and benchmark_timing > 1:
        all_times = [transcription_time]
        for _ in range(benchmark_timing - 1):
            r = transcribe_fn(entry)
            t = r[1]
            all_times.append(t)
        transcription_times_json = json.dumps(all_times)
    else:
        transcription_times_json = None

    ref_text = normalizer(raw_ref_text)
    eval_text = normalizer(raw_eval_text)

    entry_metrics = jiwer.process_words([ref_text], [eval_text])

    entry_data = {
        "id": i,
        "reference_text": raw_ref_text,
        "predicted_text": raw_eval_text,
        "norm_reference_text": ref_text,
        "norm_predicted_text": eval_text,
        "wer": entry_metrics.wer,
        "wil": entry_metrics.wil,
        "substitutions": entry_metrics.substitutions,
        "deletions": entry_metrics.deletions,
        "insertions": entry_metrics.insertions,
        "hits": entry_metrics.hits,
        "audio_duration": audio_duration,
        "transcription_time": transcription_time,
    }

    if transcription_times_json:
        entry_data["transcription_times"] = transcription_times_json

    for key in entry.keys():
        if key not in ["audio", text_column]:
            entry_data[f"metadata_{key}"] = entry[key]

    return entry_data


def calculate_final_metrics(df: pandas.DataFrame):
    df = df.sort_values(by=["id"])
    df["reference_text"] = df["reference_text"].fillna("")
    df["predicted_text"] = df["predicted_text"].fillna("")

    entries_data = df.to_dict(orient="records")
    htn = HebrewTextNormalizer()

    results = jiwer.process_words(
        [htn(entry["reference_text"]) for entry in entries_data],
        [htn(entry["predicted_text"]) for entry in entries_data],
    )

    return results


def calculate_transcription_time_stats(df: pandas.DataFrame):
    """Calculate transcription time statistics for segments >= 5 seconds."""
    audio_durations = []
    text_lengths = []
    transcription_times = []

    for _, row in df.iterrows():
        audio_duration = row.get("audio_duration", 1.0)
        if audio_duration < 5.0:
            continue

        audio_durations.append(audio_duration)
        text_length = len(row["predicted_text"]) if row["predicted_text"] else 0
        text_lengths.append(text_length)
        transcription_time = row.get("transcription_time", 0)
        transcription_times.append(transcription_time)

    time_per_second = [t / d if d > 0 else 0 for t, d in zip(transcription_times, audio_durations)]
    time_per_char = [t / l if l > 0 else 0 for t, l in zip(transcription_times, text_lengths)]

    def calculate_percentiles(data):
        data = [x for x in data if x > 0]
        if not data:
            return {"mean": 0, "median": 0, "p90": 0, "p99": 0}

        data.sort()
        n = len(data)
        return {
            "mean": sum(data) / n,
            "median": data[n // 2] if n % 2 == 1 else (data[n // 2 - 1] + data[n // 2]) / 2,
            "p90": data[int(0.9 * n)] if n > 0 else 0,
            "p99": data[int(0.99 * n)] if n > 0 else 0
        }

    return {
        "time_per_second": calculate_percentiles(time_per_second),
        "time_per_char": calculate_percentiles(time_per_char),
        "raw_time": calculate_percentiles(transcription_times)
    }


def evaluate_model(transcribe_fn, ds, text_column, num_workers=1, benchmark_timing=None, max_samples=None):
    normalizer = HebrewTextNormalizer()

    # Limit samples if specified
    total_samples = len(ds) if max_samples is None else min(max_samples, len(ds))

    # Prepare arguments for parallel processing
    process_args = [(i, ds[i], transcribe_fn, text_column, normalizer, benchmark_timing) for i in range(total_samples)]

    # Process entries in parallel with progress tracking
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_entry, arg) for arg in process_args]

        entries_data = []
        with tqdm(total=len(futures), desc="Processing entries") as pbar:
            for future in concurrent.futures.as_completed(futures):
                entry = future.result()
                entries_data.append(entry)
                last_wer = entry["wer"]
                last_wil = entry["wil"]
                pbar.set_postfix(last_wer=f"{last_wer:.5f}", last_wil=f"{last_wil:.5f}")
                pbar.update(1)

    entries_data.sort(key=lambda x: x["id"])
    return pandas.DataFrame(entries_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a speech-to-text model.")
    parser.add_argument("--engine", type=str, required=True, help="Path to engine script")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset to evaluate in format dataset_name:<split>:<text_column>"
    )
    parser.add_argument("--name", type=str, required=False, help="Optional name parameter for dataset.load_dataset")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Output CSV file path")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda:0, cuda:1, cpu)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--benchmark-timing", type=int, default=None, metavar="N",
                        help="Run each transcription N times and store all timing results")

    args = parser.parse_args()

    # Parse dataset info
    dataset_parts = args.dataset.split(":")
    dataset_name = dataset_parts[0]
    dataset_split = dataset_parts[1] if len(dataset_parts) > 1 else "test"
    ds_text_column = dataset_parts[2] if len(dataset_parts) > 2 else "text"

    output_exists = os.path.exists(args.output)

    if output_exists and not args.overwrite:
        results_df = pandas.read_csv(args.output)
    else:
        # Import the engine module
        import importlib.util

        spec = importlib.util.spec_from_file_location("engine", args.engine)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)

        print(f"Loading engine {args.engine} with model {args.model}...")
        transcribe_fn = engine.create_app(model_path=args.model, device=args.device)

        print(f"Loading dataset {args.dataset}...")
        if args.name:
            ds = datasets.load_dataset(dataset_name, name=args.name)[dataset_split]
        else:
            ds = datasets.load_dataset(dataset_name)[dataset_split]

        # Disable auto audio decoding to avoid torchcodec dependency
        # Engine will handle audio decoding with librosa
        ds = ds.cast_column("audio", datasets.Audio(decode=False))

        print(f"Beginning evaluation with {args.workers} workers.")
        results_df = evaluate_model(transcribe_fn, ds, ds_text_column, args.workers, args.benchmark_timing, args.max_samples)

        results_df["model"] = args.model
        results_df["dataset"] = dataset_name
        results_df["dataset_split"] = dataset_split
        results_df["engine"] = args.engine

        results_df.to_csv(args.output, encoding="utf-8", index=False)
        print(f"Results saved to {args.output}")

    # Calculate final metrics
    metrics = calculate_final_metrics(results_df)
    time_stats = calculate_transcription_time_stats(results_df)

    print(f"\nEvaluation done. WER={metrics.wer:.4f}, WIL={metrics.wil:.4f}")

    workload = f"{dataset_name}:{dataset_split}"
    print(f"\n{workload}")
    print(f"  Raw time:      mean={time_stats['raw_time']['mean']:.3f}s, median={time_stats['raw_time']['median']:.3f}s, p90={time_stats['raw_time']['p90']:.3f}s")
    print(f"  Per audio sec: mean={time_stats['time_per_second']['mean']:.3f}s, median={time_stats['time_per_second']['median']:.3f}s, p90={time_stats['time_per_second']['p90']:.3f}s")
