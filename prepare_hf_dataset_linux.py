#!/usr/bin/env python3
"""
Prepare 100k Hebrew ASR dataset for HuggingFace Hub (Linux-compatible).

Features:
- Extracts 23k Knesset + 77k crowd-transcribe (balanced mix)
- Preserves context from Knesset (speaker continuity)
- Normalizes Hebrew text (removes niqqud, timestamps)
- Splits into train/eval (95/5)
- Uploads to HuggingFace Hub: OzLabs/qwen3-asr-hebrew-100k

Note: SpecAugment is NOT applied here (too expensive to store augmented versions).
      SpecAugment will be applied on-the-fly during training in the data collator.

Usage:
    # On Linux/Lambda server
    uv run python prepare_hf_dataset_linux.py
"""

import io
import json
import re
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict

import librosa
import numpy as np
from datasets import Dataset, DatasetDict, Audio, Features, Value, load_dataset
from tqdm import tqdm
import soundfile as sf


def normalize_text(text: str) -> str:
    """Clean Hebrew text."""
    if not text:
        return ""
    # Remove Whisper timestamps
    text = re.sub(r'<\|[\d.]+\|>', '', text)
    # Remove niqqud (Hebrew vowel marks U+0591-U+05C7)
    text = re.sub(r'[\u0591-\u05C7]', '', text)
    # Remove duplicate punctuation
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    return ' '.join(text.split()).strip()


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    knesset_samples: int = 23000
    crowd_samples: int = 77166
    sampling_rate: int = 16000
    min_duration: float = 0.5
    max_duration: float = 30.0
    context_prob: float = 0.5  # 50% of samples get previous context
    train_split: float = 0.90  # 90% train, 10% eval
    hf_repo: str = "OzLabs/qwen3-asr-hebrew-100k"
    random_seed: int = 42


def extract_knesset_with_context(config: DatasetConfig) -> List[Dict]:
    """
    Extract Knesset samples with speaker context preservation.

    Context strategy (ivrit.ai method):
    - Group by episode_id (same session = same speaker continuity)
    - For each sample, store previous transcript as context (50% probability)
    - This helps model learn speaker consistency and discourse flow
    """
    print(f"\n[1/3] Extracting {config.knesset_samples:,} Knesset samples with context...")

    ds = load_dataset(
        "ivrit-ai/knesset-plenums-whisper-training",
        split="train",
        streaming=True,
    )

    # Limit streaming to ~2x target (accounting for skipped samples)
    # This prevents processing the entire dataset
    ds_limited = ds.take(config.knesset_samples * 3)

    samples = []
    skip_reasons = defaultdict(int)  # Track why samples are skipped

    for idx, example in enumerate(tqdm(ds_limited, desc="Knesset", total=config.knesset_samples)):
        if len(samples) >= config.knesset_samples:
            break

        # Extract text (Knesset uses 'transcript' column)
        text = normalize_text(example.get('transcript', ''))
        if not text or len(text.strip()) < 3:
            skip_reasons['empty_text'] += 1
            continue

        try:
            # Extract audio (HuggingFace Audio feature on Linux)
            audio_dict = example['audio']
            audio_array = np.array(audio_dict['array'], dtype=np.float32)
            sampling_rate = int(audio_dict['sampling_rate'])

            # Duration filter
            duration = len(audio_array) / sampling_rate
            if duration < config.min_duration:
                skip_reasons['too_short'] += 1
                continue
            if duration > config.max_duration:
                skip_reasons['too_long'] += 1
                continue

            # Resample to 16kHz if needed
            if sampling_rate != config.sampling_rate:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sampling_rate,
                    target_sr=config.sampling_rate
                )

            # Get previous context (Knesset already provides this!)
            context = ""
            has_prev = example.get('has_prev', False)
            if has_prev and np.random.random() < config.context_prob:
                # Use the provided prev_transcript
                prev_text = example.get('prev_transcript', '')
                if prev_text:
                    context = normalize_text(prev_text)

            # Get metadata for episode tracking (optional, for stats)
            metadata = example.get('metadata', {})
            episode_id = metadata.get('episode_id', f"unknown_{idx}")

            # Create sample
            sample = {
                'audio': audio_array,
                'text': text,
                'context': context,
                'source': 'knesset',
                'duration': float(duration),
                'episode_id': episode_id,
            }
            samples.append(sample)

        except Exception as e:
            skip_reasons[f'exception: {type(e).__name__}'] += 1
            # Print first few exceptions for debugging
            if sum(1 for k in skip_reasons.keys() if 'exception' in k) <= 3:
                print(f"\n  Debug - Exception on example {idx}: {e}")
            continue

    print(f"✓ Extracted {len(samples):,} Knesset samples")
    if len(samples) > 0:
        print(f"  Samples with context: {sum(1 for s in samples if s['context']):,} ({sum(1 for s in samples if s['context'])/len(samples)*100:.1f}%)")
    else:
        print(f"  ⚠ WARNING: No valid samples extracted!")

    # Print skip reasons
    total_skipped = sum(skip_reasons.values())
    print(f"  Skipped: {total_skipped:,}")
    if skip_reasons:
        print(f"  Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count:,} ({count/total_skipped*100:.1f}%)")

    return samples


def load_crowd_transcribe_from_jsonl(config: DatasetConfig) -> List[Dict]:
    """
    Load existing crowd-transcribe samples from JSONL.

    Assumes you have the prepared JSONL from prepare_qwen_data.py.
    """
    print(f"\n[2/3] Loading {config.crowd_samples:,} crowd-transcribe samples...")

    jsonl_path = Path("qwen3_asr_data/train_ivrit-ai_crowd-transcribe-v5.jsonl")

    if not jsonl_path.exists():
        print(f"⚠ WARNING: {jsonl_path} not found!")
        print(f"  Attempting to stream from HuggingFace instead...")
        return load_crowd_transcribe_streaming(config)

    samples = []
    skip_reasons = defaultdict(int)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Crowd-transcribe", total=config.crowd_samples)):
            try:
                entry = json.loads(line.strip())

                # Load audio from WAV file
                audio_path = Path(entry['audio'])
                if not audio_path.exists():
                    skip_reasons['missing_audio_file'] += 1
                    continue

                audio_array, sr = sf.read(str(audio_path))
                audio_array = np.array(audio_array, dtype=np.float32)

                # Resample if needed
                if sr != config.sampling_rate:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sr,
                        target_sr=config.sampling_rate
                    )

                # Extract text (remove "language Hebrew<asr_text>" prefix if present)
                text = entry['text']
                if text.startswith("language Hebrew<asr_text>"):
                    text = text.replace("language Hebrew<asr_text>", "").strip()
                text = normalize_text(text)

                if not text or len(text.strip()) < 3:
                    skip_reasons['empty_text'] += 1
                    continue

                duration = len(audio_array) / config.sampling_rate
                if duration < config.min_duration:
                    skip_reasons['too_short'] += 1
                    continue
                if duration > config.max_duration:
                    skip_reasons['too_long'] += 1
                    continue

                sample = {
                    'audio': audio_array,
                    'text': text,
                    'context': "",  # Crowd data has no context
                    'source': 'crowd-transcribe',
                    'duration': float(duration),
                    'episode_id': "",
                }
                samples.append(sample)

            except Exception as e:
                skip_reasons[f'exception: {type(e).__name__}'] += 1
                if sum(1 for k in skip_reasons.keys() if 'exception' in k) <= 3:
                    print(f"\n  Debug - Exception on line {line_num}: {e}")
                continue

    print(f"✓ Loaded {len(samples):,} crowd-transcribe samples")
    total_skipped = sum(skip_reasons.values())
    print(f"  Skipped: {total_skipped:,}")
    if skip_reasons:
        print(f"  Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count:,} ({count/total_skipped*100:.1f}%)")
    return samples


def load_crowd_transcribe_streaming(config: DatasetConfig) -> List[Dict]:
    """
    Fallback: Stream crowd-transcribe from HuggingFace if JSONL not found.
    """
    print(f"  Streaming from HuggingFace...")

    ds = load_dataset(
        "ivrit-ai/crowd-transcribe-v5",
        split="train",
        streaming=True,
    )

    # Limit streaming to ~2x target (accounting for skipped samples)
    ds_limited = ds.take(config.crowd_samples * 2)

    samples = []
    skip_reasons = defaultdict(int)

    for idx, example in enumerate(tqdm(ds_limited, desc="Crowd-transcribe (streaming)", total=config.crowd_samples)):
        if len(samples) >= config.crowd_samples:
            break

        text = normalize_text(example.get('sentence') or example.get('text', ''))
        if not text or len(text.strip()) < 3:
            skip_reasons['empty_text'] += 1
            continue

        try:
            audio_dict = example['audio']
            audio_array = np.array(audio_dict['array'], dtype=np.float32)
            sampling_rate = int(audio_dict['sampling_rate'])

            duration = len(audio_array) / sampling_rate
            if duration < config.min_duration:
                skip_reasons['too_short'] += 1
                continue
            if duration > config.max_duration:
                skip_reasons['too_long'] += 1
                continue

            if sampling_rate != config.sampling_rate:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sampling_rate,
                    target_sr=config.sampling_rate
                )

            sample = {
                'audio': audio_array,
                'text': text,
                'context': "",
                'source': 'crowd-transcribe',
                'duration': float(duration),
                'episode_id': "",
            }
            samples.append(sample)

        except Exception as e:
            skip_reasons[f'exception: {type(e).__name__}'] += 1
            if sum(1 for k in skip_reasons.keys() if 'exception' in k) <= 3:
                print(f"\n  Debug - Exception on example {idx}: {e}")
            continue

    print(f"✓ Streamed {len(samples):,} crowd-transcribe samples")
    total_skipped = sum(skip_reasons.values())
    print(f"  Skipped: {total_skipped:,}")
    if skip_reasons:
        print(f"  Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count:,} ({count/total_skipped*100:.1f}%)")
    return samples


def create_hf_dataset(samples: List[Dict], config: DatasetConfig) -> DatasetDict:
    """
    Create HuggingFace Dataset with train/eval split.

    Features:
    - audio: Audio waveform (no augmentation - applied during training)
    - text: Normalized Hebrew transcript
    - context: Previous transcript (for Knesset samples)
    - source: Dataset source (knesset or crowd-transcribe)
    - duration: Audio duration in seconds
    """
    print(f"\n[3/3] Creating HuggingFace Dataset with train/eval split...")

    # Shuffle samples for balanced distribution
    np.random.seed(config.random_seed)
    np.random.shuffle(samples)

    # Split into train/eval
    split_idx = int(len(samples) * config.train_split)
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    print(f"  Train: {len(train_samples):,} samples ({config.train_split*100:.0f}%)")
    print(f"  Eval: {len(eval_samples):,} samples ({(1-config.train_split)*100:.0f}%)")

    # Convert to dict format
    def samples_to_dict(samples_list):
        return {
            'audio': [s['audio'] for s in samples_list],
            'text': [s['text'] for s in samples_list],
            'context': [s['context'] for s in samples_list],
            'source': [s['source'] for s in samples_list],
            'duration': [s['duration'] for s in samples_list],
        }

    train_dict = samples_to_dict(train_samples)
    eval_dict = samples_to_dict(eval_samples)

    # Define features
    features = Features({
        'audio': Audio(sampling_rate=config.sampling_rate),
        'text': Value('string'),
        'context': Value('string'),
        'source': Value('string'),
        'duration': Value('float32'),
    })

    # Create datasets
    print(f"\n  Creating train dataset...")
    train_ds = Dataset.from_dict(train_dict, features=features)

    print(f"  Creating eval dataset...")
    eval_ds = Dataset.from_dict(eval_dict, features=features)

    dataset_dict = DatasetDict({
        'train': train_ds,
        'validation': eval_ds,
    })

    # Print statistics
    print(f"\n✓ Dataset created successfully")
    print(f"\nDataset info:")
    print(f"  Train: {len(train_ds):,} examples")
    print(f"  Validation: {len(eval_ds):,} examples")
    print(f"  Total: {len(train_ds) + len(eval_ds):,} examples")

    # Source distribution
    train_sources = defaultdict(int)
    for source in train_dict['source']:
        train_sources[source] += 1

    print(f"\nSource distribution (train):")
    for source, count in sorted(train_sources.items()):
        pct = count / len(train_ds) * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")

    # Context statistics
    train_with_context = sum(1 for c in train_dict['context'] if c)
    print(f"\nContext statistics (train):")
    print(f"  Samples with context: {train_with_context:,} ({train_with_context/len(train_ds)*100:.1f}%)")

    # Duration statistics
    train_durations = train_dict['duration']
    print(f"\nDuration statistics (train):")
    print(f"  Mean: {np.mean(train_durations):.1f}s")
    print(f"  Median: {np.median(train_durations):.1f}s")
    print(f"  Min: {np.min(train_durations):.1f}s")
    print(f"  Max: {np.max(train_durations):.1f}s")

    return dataset_dict


def main():
    config = DatasetConfig()

    print("="*60)
    print("Qwen3-ASR Hebrew 100k Dataset Preparation")
    print("="*60)
    print(f"Target: ~{config.knesset_samples + config.crowd_samples:,} samples")
    print(f"  - Knesset: {config.knesset_samples:,} (with context)")
    print(f"  - Crowd-transcribe: ~{config.crowd_samples:,}")
    print(f"HuggingFace repo: {config.hf_repo}")
    print(f"Train/Eval split: {config.train_split*100:.0f}/{(1-config.train_split)*100:.0f}")
    print("="*60)

    # Step 1: Extract Knesset with context
    knesset_samples = extract_knesset_with_context(config)

    # Step 2: Load crowd-transcribe
    crowd_samples = load_crowd_transcribe_from_jsonl(config)

    # Combine samples
    all_samples = knesset_samples + crowd_samples
    print(f"\nTotal samples collected: {len(all_samples):,}")

    if len(all_samples) == 0:
        print("⚠ ERROR: No samples collected!")
        return

    # Step 3: Create HuggingFace Dataset with splits
    dataset = create_hf_dataset(all_samples, config)

    # Step 4: Upload to HuggingFace Hub
    print(f"\n{'='*60}")
    print("Uploading to HuggingFace Hub")
    print(f"{'='*60}")
    print(f"Repository: {config.hf_repo}")
    print("This may take 10-30 minutes depending on network speed...")

    try:
        dataset.push_to_hub(
            config.hf_repo,
            private=False,  # Public dataset
            commit_message=f"Initial upload: {len(all_samples):,} balanced Hebrew ASR samples with context"
        )
        print(f"\n✓ Dataset uploaded successfully!")
        print(f"\nDataset URL: https://huggingface.co/datasets/{config.hf_repo}")

    except Exception as e:
        print(f"\n⚠ Upload failed: {e}")
        print(f"\nTo upload manually:")
        print(f"  from datasets import DatasetDict")
        print(f"  # (after recreating dataset)")
        print(f"  dataset.push_to_hub('{config.hf_repo}')")

    print("\n" + "="*60)
    print("✓ Dataset Preparation Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Verify dataset: https://huggingface.co/datasets/{config.hf_repo}")
    print(f"  2. Update training script:")
    print(f"     ds = load_dataset('{config.hf_repo}')")
    print(f"  3. Launch training with SpecAugment applied on-the-fly")
    print("="*60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
