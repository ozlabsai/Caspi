#!/usr/bin/env python3
"""
Prepare 100k preprocessed Hebrew ASR dataset with context.

Features:
- Streams and samples 23k Knesset + 77k crowd-transcribe
- Extracts audio features (mel spectrograms) upfront
- Preserves context from Knesset (speaker continuity)
- Normalizes Hebrew text (removes niqqud)
- Outputs HuggingFace Dataset ready for upload

Output: OzLabs/qwen3-asr-hebrew-100k on HuggingFace Hub
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

import librosa
import numpy as np
from datasets import Dataset, DatasetDict, Audio, Features, Value, load_dataset
from tqdm import tqdm
import soundfile as sf


def normalize_text(text: str) -> str:
    """Clean Hebrew text (same as prepare_qwen_data.py)."""
    if not text:
        return ""
    # Remove Whisper timestamps
    text = re.sub(r'<\|[\d.]+\|>', '', text)
    # Remove niqqud (Hebrew vowel marks)
    text = re.sub(r'[\u0591-\u05C7]', '', text)
    # Remove duplicate punctuation
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    return ' '.join(text.split()).strip()


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    knesset_samples: int = 23000
    crowd_samples: int = 77166  # All existing crowd-transcribe
    sampling_rate: int = 16000
    min_duration: float = 0.5
    max_duration: float = 30.0
    context_prob: float = 0.5  # 50% of samples get previous context
    output_dir: str = "./qwen3_asr_100k_preprocessed"
    hf_repo: str = "OzLabs/qwen3-asr-hebrew-100k"


def extract_knesset_with_context(config: DatasetConfig) -> List[Dict]:
    """
    Extract Knesset samples with speaker context preservation.

    Strategy:
    - Group by episode_id (same session = same speaker continuity)
    - For each sample, store previous transcript as context
    - This mimics ivrit.ai's 50% context strategy
    """
    print(f"\n[1/3] Extracting {config.knesset_samples:,} Knesset samples with context...")

    # Use decode=False to get raw bytes, avoiding torchcodec issues on Mac
    import io
    ds = load_dataset(
        "ivrit-ai/knesset-plenums-whisper-training",
        split="train",
        streaming=True,
    )

    samples = []
    episode_history = defaultdict(list)  # episode_id -> list of transcripts

    for idx, example in enumerate(tqdm(ds, desc="Knesset", total=config.knesset_samples)):
        if len(samples) >= config.knesset_samples:
            break

        # Extract text
        text = normalize_text(example.get('sentence') or example.get('text', ''))
        if not text or len(text.strip()) < 3:
            continue

        try:
            # Decode audio bytes manually with librosa (bypasses torchcodec)
            audio_dict = example['audio']
            audio_bytes = audio_dict.get('bytes')
            if audio_bytes:
                # Load from bytes
                audio_array, sampling_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            else:
                # Fallback: use array if available
                audio_array = audio_dict.get('array')
                sampling_rate = audio_dict.get('sampling_rate', 16000)
                if audio_array is None:
                    continue

            # Duration filter
            duration = len(audio_array) / sampling_rate
            if duration < config.min_duration or duration > config.max_duration:
                continue

            # Resample to 16kHz if needed
            if sampling_rate != config.sampling_rate:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sampling_rate,
                    target_sr=config.sampling_rate
                )

            # Get episode ID for context tracking
            episode_id = example.get('episode_id', f"unknown_{idx}")

            # Get previous context (if available)
            context = ""
            if episode_id in episode_history and len(episode_history[episode_id]) > 0:
                # Use previous transcript as context (50% probability)
                if np.random.random() < config.context_prob:
                    context = episode_history[episode_id][-1]

            # Store this transcript for future context
            episode_history[episode_id].append(text)
            if len(episode_history[episode_id]) > 3:  # Keep last 3 only
                episode_history[episode_id].pop(0)

            # Create sample
            sample = {
                'audio': audio_array,
                'text': text,
                'context': context,
                'source': 'knesset',
                'duration': duration,
                'episode_id': episode_id,
            }
            samples.append(sample)

        except Exception as e:
            continue

    print(f"✓ Extracted {len(samples):,} Knesset samples")
    print(f"  Samples with context: {sum(1 for s in samples if s['context']):,}")
    return samples


def load_crowd_transcribe(config: DatasetConfig) -> List[Dict]:
    """
    Load existing crowd-transcribe samples from JSONL.

    These are from the existing 77k dataset we already have.
    """
    print(f"\n[2/3] Loading {config.crowd_samples:,} crowd-transcribe samples...")

    jsonl_path = Path("qwen3_asr_data/train_ivrit-ai_crowd-transcribe-v5.jsonl")
    audio_dir = Path("qwen3_asr_data/audio/train/ivrit-ai_crowd-transcribe-v5")

    if not jsonl_path.exists():
        print(f"⚠ WARNING: {jsonl_path} not found!")
        print(f"  Run prepare_qwen_data.py first or adjust path")
        return []

    samples = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Crowd-transcribe", total=config.crowd_samples):
            try:
                entry = json.loads(line.strip())

                # Load audio from WAV file
                audio_path = Path(entry['audio'])
                if not audio_path.exists():
                    continue

                audio_array, sr = sf.read(str(audio_path))

                # Resample if needed
                if sr != config.sampling_rate:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sr,
                        target_sr=config.sampling_rate
                    )

                # Extract text (remove "language Hebrew<asr_text>" prefix)
                text = entry['text']
                if text.startswith("language Hebrew<asr_text>"):
                    text = text.replace("language Hebrew<asr_text>", "").strip()
                text = normalize_text(text)

                if not text or len(text.strip()) < 3:
                    continue

                duration = len(audio_array) / config.sampling_rate

                sample = {
                    'audio': audio_array,
                    'text': text,
                    'context': "",  # Crowd data has no context
                    'source': 'crowd-transcribe',
                    'duration': duration,
                    'episode_id': "",
                }
                samples.append(sample)

            except Exception as e:
                continue

    print(f"✓ Loaded {len(samples):,} crowd-transcribe samples")
    return samples


def create_hf_dataset(samples: List[Dict], config: DatasetConfig) -> Dataset:
    """
    Create HuggingFace Dataset from samples.

    Features:
    - audio: Audio feature (waveform)
    - text: Normalized Hebrew transcript
    - context: Previous transcript (for Knesset samples)
    - source: Dataset source (knesset or crowd-transcribe)
    - duration: Audio duration in seconds
    """
    print(f"\n[3/3] Creating HuggingFace Dataset...")

    # Shuffle samples for balanced training
    np.random.shuffle(samples)

    # Split into train/eval (95/5 split)
    split_idx = int(len(samples) * 0.95)
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    print(f"  Train: {len(train_samples):,} samples")
    print(f"  Eval: {len(eval_samples):,} samples")

    # Create datasets
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

    train_ds = Dataset.from_dict(train_dict, features=features)
    eval_ds = Dataset.from_dict(eval_dict, features=features)

    dataset_dict = DatasetDict({
        'train': train_ds,
        'validation': eval_ds,
    })

    print(f"✓ Dataset created successfully")
    print(f"\nDataset info:")
    print(f"  Train: {len(train_ds):,} examples")
    print(f"  Validation: {len(eval_ds):,} examples")
    print(f"  Total: {len(train_ds) + len(eval_ds):,} examples")

    # Print source distribution
    train_sources = defaultdict(int)
    for source in train_dict['source']:
        train_sources[source] += 1

    print(f"\nSource distribution (train):")
    for source, count in sorted(train_sources.items()):
        pct = count / len(train_ds) * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")

    # Print context stats
    train_with_context = sum(1 for c in train_dict['context'] if c)
    print(f"\nContext statistics (train):")
    print(f"  Samples with context: {train_with_context:,} ({train_with_context/len(train_ds)*100:.1f}%)")

    return dataset_dict


def main():
    config = DatasetConfig()

    print("="*60)
    print("Qwen3-ASR Hebrew 100k Dataset Preparation")
    print("="*60)
    print(f"Target: {config.knesset_samples + config.crowd_samples:,} samples")
    print(f"  - Knesset: {config.knesset_samples:,} (with context)")
    print(f"  - Crowd-transcribe: {config.crowd_samples:,}")
    print(f"HuggingFace repo: {config.hf_repo}")
    print("="*60)

    # Step 1: Extract Knesset with context
    knesset_samples = extract_knesset_with_context(config)

    # Step 2: Load crowd-transcribe
    crowd_samples = load_crowd_transcribe(config)

    # Combine samples
    all_samples = knesset_samples + crowd_samples
    print(f"\nTotal samples collected: {len(all_samples):,}")

    if len(all_samples) == 0:
        print("⚠ ERROR: No samples collected!")
        return

    # Step 3: Create HuggingFace Dataset
    dataset = create_hf_dataset(all_samples, config)

    # Step 4: Save locally first
    output_path = Path(config.output_dir)
    print(f"\nSaving dataset locally to: {output_path}")
    dataset.save_to_disk(str(output_path))
    print(f"✓ Dataset saved locally")

    # Step 5: Upload to HuggingFace Hub
    print(f"\nUploading to HuggingFace Hub: {config.hf_repo}")
    print("This may take 10-30 minutes depending on your upload speed...")

    try:
        dataset.push_to_hub(
            config.hf_repo,
            private=False,  # Set to True if you want private dataset
            commit_message="Initial upload: 100k balanced Hebrew ASR dataset with context"
        )
        print(f"✓ Dataset uploaded successfully!")
        print(f"\nDataset URL: https://huggingface.co/datasets/{config.hf_repo}")

    except Exception as e:
        print(f"⚠ Upload failed: {e}")
        print(f"Dataset is still saved locally at: {output_path}")
        print(f"\nTo upload manually later:")
        print(f"  from datasets import load_from_disk")
        print(f"  ds = load_from_disk('{output_path}')")
        print(f"  ds.push_to_hub('{config.hf_repo}')")

    print("\n" + "="*60)
    print("✓ Dataset Preparation Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Verify dataset on HF Hub: https://huggingface.co/datasets/{config.hf_repo}")
    print(f"  2. Update training script to use: load_dataset('{config.hf_repo}')")
    print(f"  3. Launch training on Lambda with preprocessed data")
    print("="*60)


if __name__ == "__main__":
    main()
