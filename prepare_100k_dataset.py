#!/usr/bin/env python3
"""
Prepare 100k balanced dataset for efficient training:
- 77k existing crowd-transcribe (already in JSONL)
- 23k new Knesset samples (streaming download)

Output: ./qwen3_asr_data/train_100k.jsonl
"""

import json
import io
import re
from pathlib import Path
from tqdm import tqdm

import librosa
import soundfile as sf
from datasets import load_dataset


def normalize_text(text):
    """Clean Hebrew text (same as prepare_qwen_data.py)."""
    if not text:
        return ""
    text = re.sub(r'<\|[\d.]+\|>', '', text)  # Whisper timestamps
    text = re.sub(r'[\u0591-\u05C7]', '', text)  # Niqqud
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)  # Duplicate punctuation
    return ' '.join(text.split()).strip()


def process_knesset_streaming(num_samples=23000, output_dir="./qwen3_asr_data"):
    """
    Stream Knesset dataset and extract 23k samples.

    Args:
        num_samples: Number of Knesset samples to extract
        output_dir: Output directory for audio and JSONL
    """
    output_path = Path(output_dir)
    audio_dir = output_path / "audio" / "train" / "knesset"
    audio_dir.mkdir(parents=True, exist_ok=True)

    knesset_jsonl = output_path / "train_knesset_23k.jsonl"

    print(f"Streaming Knesset dataset (target: {num_samples:,} samples)...")
    ds = load_dataset("ivrit-ai/knesset-plenums-whisper-training", split="train", streaming=True)

    valid_count = 0
    skipped_count = 0

    with open(knesset_jsonl, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(ds, desc="Processing Knesset", total=num_samples)):
            if valid_count >= num_samples:
                break

            # Get text and audio
            text = normalize_text(example.get('sentence') or example.get('text', ''))
            if not text or len(text.strip()) < 3:
                skipped_count += 1
                continue

            try:
                # Extract audio
                audio_dict = example['audio']
                audio_array = audio_dict['array']
                sampling_rate = audio_dict['sampling_rate']

                # Skip if too short or too long
                duration = len(audio_array) / sampling_rate
                if duration < 0.5 or duration > 30.0:
                    skipped_count += 1
                    continue

                # Resample to 16kHz if needed
                if sampling_rate != 16000:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sampling_rate,
                        target_sr=16000
                    )
                    sampling_rate = 16000

                # Save WAV file
                wav_filename = f"knesset_{valid_count:06d}.wav"
                wav_path = audio_dir / wav_filename
                sf.write(str(wav_path), audio_array, sampling_rate)

                # Write JSONL entry
                entry = {
                    "audio": str(wav_path.absolute()),
                    "text": f"language Hebrew<asr_text>{text}"
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                valid_count += 1

            except Exception as e:
                skipped_count += 1
                continue

    print(f"\n✓ Knesset: {valid_count:,} samples extracted")
    print(f"  Skipped: {skipped_count:,} (empty text, bad audio, or wrong duration)")
    print(f"  Output: {knesset_jsonl}")

    return knesset_jsonl


def merge_datasets(output_dir="./qwen3_asr_data"):
    """
    Merge existing 77k crowd-transcribe with new 23k Knesset.

    Output: train_100k.jsonl
    """
    output_path = Path(output_dir)

    crowd_jsonl = output_path / "train_ivrit-ai_crowd-transcribe-v5.jsonl"
    knesset_jsonl = output_path / "train_knesset_23k.jsonl"
    merged_jsonl = output_path / "train_100k.jsonl"

    print(f"\nMerging datasets...")
    print(f"  Source 1: {crowd_jsonl} (77k crowd-transcribe)")
    print(f"  Source 2: {knesset_jsonl} (23k Knesset)")
    print(f"  Output: {merged_jsonl}")

    total_count = 0

    with open(merged_jsonl, 'w', encoding='utf-8') as out_f:
        # Add crowd-transcribe (all 77k)
        if crowd_jsonl.exists():
            with open(crowd_jsonl, 'r', encoding='utf-8') as in_f:
                for line in tqdm(in_f, desc="Adding crowd-transcribe"):
                    out_f.write(line)
                    total_count += 1
        else:
            print(f"⚠ WARNING: {crowd_jsonl} not found!")

        # Add Knesset (23k)
        if knesset_jsonl.exists():
            with open(knesset_jsonl, 'r', encoding='utf-8') as in_f:
                for line in tqdm(in_f, desc="Adding Knesset"):
                    out_f.write(line)
                    total_count += 1
        else:
            print(f"⚠ WARNING: {knesset_jsonl} not found!")

    print(f"\n✓ Merged dataset: {total_count:,} examples")
    print(f"  Output: {merged_jsonl}")

    # Print breakdown
    crowd_count = sum(1 for _ in open(crowd_jsonl)) if crowd_jsonl.exists() else 0
    knesset_count = sum(1 for _ in open(knesset_jsonl)) if knesset_jsonl.exists() else 0

    print(f"\nDataset composition:")
    print(f"  Crowd-transcribe: {crowd_count:,} ({crowd_count/total_count*100:.1f}%)")
    print(f"  Knesset: {knesset_count:,} ({knesset_count/total_count*100:.1f}%)")

    return merged_jsonl


def main():
    print("=" * 60)
    print("Preparing 100k Balanced Dataset")
    print("=" * 60)

    # Step 1: Extract 23k Knesset samples
    knesset_jsonl = process_knesset_streaming(num_samples=23000)

    # Step 2: Merge with existing 77k crowd-transcribe
    final_jsonl = merge_datasets()

    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Upload to Lambda server:")
    print(f"     scp -r qwen3_asr_data/ ubuntu@lambda-server:~/caspi/")
    print(f"  2. Adapt qwen3_asr_sft_official.py with Muon + SpecAugment")
    print(f"  3. Launch training with torchrun --nproc_per_node=8")


if __name__ == "__main__":
    main()
