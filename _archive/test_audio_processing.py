#!/usr/bin/env python3
"""
Test script to verify audio processing works with one sample.
"""

from datasets import load_dataset, Audio
import soundfile as sf
from pathlib import Path

print("Testing audio processing with one sample...")
print("=" * 60)

# Load just one example from the dataset
print("\n1. Loading one example from dataset...")
ds = load_dataset("ivrit-ai/crowd-transcribe-v5", split="train", streaming=True)

# Take first example
example = next(iter(ds))
print(f"✓ Loaded example")

# Cast audio to 16kHz
print("\n2. Resampling audio to 16kHz...")
audio_feature = Audio(sampling_rate=16000)
audio_data = audio_feature.decode_example(example["audio"])
print(f"✓ Audio resampled")
print(f"   Sampling rate: {audio_data['sampling_rate']}")
print(f"   Array shape: {audio_data['array'].shape}")
print(f"   Duration: {len(audio_data['array']) / audio_data['sampling_rate']:.2f} seconds")

# Try to save it
print("\n3. Saving audio to WAV file...")
test_dir = Path("./test_audio")
test_dir.mkdir(exist_ok=True)
test_file = test_dir / "test_sample.wav"

sf.write(
    str(test_file),
    audio_data['array'],
    audio_data['sampling_rate']
)
print(f"✓ Saved to {test_file}")

# Check text
print("\n4. Checking text transcription...")
if "transcript" in example:
    text = example["transcript"]
elif "text" in example:
    text = example["text"]
else:
    text = example.get("transcription", "")

print(f"✓ Text: {text[:100]}...")

print("\n" + "=" * 60)
print("SUCCESS! Audio processing works correctly.")
print("The fixed script should work on Lambda Labs.")
print("=" * 60)
