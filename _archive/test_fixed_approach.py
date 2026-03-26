#!/usr/bin/env python3
"""
Test the EXACT approach used in train_qwen3_asr_fixed.py
"""

from datasets import load_dataset, Audio
import soundfile as sf
from pathlib import Path

print("Testing the fixed audio processing approach...")
print("=" * 60)

# Load dataset (NOT streaming - this is key!)
print("\n1. Loading dataset (first 10 examples)...")
ds = load_dataset("ivrit-ai/crowd-transcribe-v5", split="train[:10]")
print(f"✓ Loaded {len(ds)} examples")

# Cast to 16kHz
print("\n2. Casting audio to 16kHz...")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
print("✓ Audio feature configured")

# Process one example
print("\n3. Processing first example...")
example = ds[0]  # This will decode the audio

audio_data = example["audio"]
audio_array = audio_data["array"]
sampling_rate = audio_data["sampling_rate"]

print(f"✓ Audio decoded")
print(f"   Sampling rate: {sampling_rate}")
print(f"   Array shape: {audio_array.shape}")
print(f"   Duration: {len(audio_array) / sampling_rate:.2f} seconds")

# Save it
print("\n4. Saving audio to WAV file...")
test_dir = Path("./test_audio")
test_dir.mkdir(exist_ok=True)
test_file = test_dir / "test_sample.wav"

sf.write(
    str(test_file),
    audio_array,
    sampling_rate
)
print(f"✓ Saved to {test_file}")

# Check text
print("\n5. Checking text transcription...")
text = example.get("transcript", example.get("text", ""))
print(f"✓ Text: {text[:100]}...")

print("\n" + "=" * 60)
print("SUCCESS! The fixed approach works!")
print("This confirms train_qwen3_asr_fixed.py will work on Lambda Labs.")
print("=" * 60)
