#!/usr/bin/env python3
"""
Test manual audio processing on ONE sample to verify it works.
This bypasses torchcodec/FFmpeg completely.
"""

import io
import json
import re
from pathlib import Path

from datasets import load_dataset
import librosa
import soundfile as sf


print("=" * 60)
print("Testing Manual Audio Processing (1 sample)")
print("=" * 60)

# Step 1: Load dataset
print("\n1. Loading dataset...")
ds = load_dataset("ivrit-ai/crowd-transcribe-v5", split="train[:1]")
print("   ✓ Dataset loaded")

# Step 2: Access raw PyArrow table directly (bypasses ALL formatting)
print("\n2. Accessing raw PyArrow table...")
import pyarrow as pa

# Get the underlying Arrow table
arrow_table = ds.data
print(f"   ✓ Got Arrow table with {len(arrow_table)} rows")
print(f"   Columns: {arrow_table.column_names}")

# Convert first row to Python dict WITHOUT using datasets formatting
first_row = arrow_table.slice(0, 1).to_pydict()
print(f"   ✓ Extracted first row")

# Step 3: Extract audio bytes from PyArrow struct
print("\n3. Extracting audio bytes...")
# Audio column is a list (one element per row)
audio_list = first_row['audio']
audio_struct = audio_list[0]  # Get first (and only) element

print(f"   Audio struct type: {type(audio_struct)}")
print(f"   Audio struct keys: {list(audio_struct.keys()) if isinstance(audio_struct, dict) else 'N/A'}")

# Extract bytes
if isinstance(audio_struct, dict) and 'bytes' in audio_struct:
    audio_bytes = audio_struct['bytes']
elif isinstance(audio_struct, bytes):
    audio_bytes = audio_struct
else:
    print(f"   ✗ Unexpected audio structure!")
    print(f"   Audio struct: {audio_struct}")
    exit(1)

if not audio_bytes:
    print("   ✗ No audio bytes found!")
    exit(1)

print(f"   ✓ Got {len(audio_bytes)} bytes")

# Step 4: Decode with librosa
print("\n4. Decoding with librosa (no FFmpeg)...")
audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
duration = len(audio_array) / sr
print(f"   ✓ Decoded!")
print(f"   - SR: {sr} Hz")
print(f"   - Shape: {audio_array.shape}")
print(f"   - Duration: {duration:.2f}s")

# Step 5: Save WAV
print("\n5. Saving WAV...")
test_dir = Path("./test_output")
test_dir.mkdir(exist_ok=True)
wav_path = test_dir / "sample.wav"
sf.write(str(wav_path), audio_array, sr)
print(f"   ✓ Saved: {wav_path}")
print(f"   - Size: {wav_path.stat().st_size} bytes")

# Step 6: Process text
print("\n6. Processing text...")
# Find text column (different datasets use different names)
text_columns = ['sentence', 'transcript', 'text', 'transcription']
text = None

for col in text_columns:
    if col in first_row:
        text_list = first_row[col]
        text = text_list[0]
        print(f"   Using column: {col}")
        break

if not text:
    print(f"   ✗ No text column found!")
    print(f"   Available columns: {list(first_row.keys())}")
    exit(1)

print(f"   Original: {text[:60]}...")

# Normalize
text = re.sub(r'<\|[\d.]+\|>', '', text)  # Timestamps
text = re.sub(r'[\u0591-\u05C7]', '', text)  # Niqqud
text = ' '.join(text.split()).strip()
print(f"   Normalized: {text[:60]}...")

# Step 7: Create JSONL
print("\n7. Creating JSONL...")
jsonl_path = test_dir / "test.jsonl"
entry = {
    "audio": str(wav_path.absolute()),
    "text": f"language Hebrew<asr_text>{text}"
}

with open(jsonl_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"   ✓ Saved: {jsonl_path}")

# Success!
print("\n" + "=" * 60)
print("SUCCESS! ✓")
print("=" * 60)
print("\nThe approach works! run_training_job.py should work.")
print("\nOutput:")
print(f"  WAV: {wav_path}")
print(f"  JSONL: {jsonl_path}")
print("=" * 60)
