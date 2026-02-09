#!/usr/bin/env python3
"""
Test Qwen3-ASR Hebrew with a sample from the evaluation datasets.

This downloads a sample from one of the Hebrew datasets and tests transcription.
"""

import subprocess
import tempfile
import time
from pathlib import Path

from datasets import load_dataset
import soundfile as sf
from openai import OpenAI

LAMBDA_IP = "209.20.159.167"
LOCAL_PORT = 8000

# Hebrew datasets to test
DATASETS = [
    ("ivrit-ai/eval-d1", "test", "audio", "transcription", None),
    ("mozilla-foundation/common_voice_17_0", "test", "audio", "sentence", {"he"}),
]

def create_ssh_tunnel():
    """Create SSH tunnel if needed."""
    # Check if tunnel exists
    check_cmd = f"lsof -ti:{LOCAL_PORT}"
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)

    if result.stdout.strip():
        print(f"✓ SSH tunnel already active on port {LOCAL_PORT}")
        return None

    print(f"Creating SSH tunnel to {LAMBDA_IP}:{LOCAL_PORT}...")
    tunnel_cmd = [
        "ssh", "-N", "-L", f"{LOCAL_PORT}:localhost:{LOCAL_PORT}",
        f"ubuntu@{LAMBDA_IP}"
    ]

    process = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(2)

    if process.poll() is not None:
        print("✗ Failed to create SSH tunnel")
        return None

    print(f"✓ SSH tunnel created (PID: {process.pid})")
    return process

def test_with_dataset_sample(dataset_name, split, audio_col, text_col, config=None):
    """Test with a sample from a Hebrew dataset."""

    print("\n" + "=" * 70)
    print(f"Testing with: {dataset_name}")
    print("=" * 70)

    # Load dataset
    print(f"Loading dataset...")
    try:
        if config:
            ds = load_dataset(dataset_name, list(config)[0], split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None

    # Get first sample
    print("Getting first sample...")
    try:
        sample = next(iter(ds))
    except Exception as e:
        print(f"✗ Failed to get sample: {e}")
        return None

    # Extract reference text
    reference = sample[text_col]
    print(f"\nReference text: {reference}")

    # Extract audio
    audio_data = sample[audio_col]

    if isinstance(audio_data, dict):
        audio_array = audio_data['array']
        sr = audio_data['sampling_rate']
    else:
        audio_array = audio_data
        sr = 16000

    audio_duration = len(audio_array) / sr
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Sample rate: {sr}Hz")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio_array, sr)

    print(f"Saved to: {tmp_path}")

    # Transcribe via SSH tunnel
    print("\nTranscribing with vLLM on H100...")

    client = OpenAI(
        base_url=f"http://localhost:{LOCAL_PORT}/v1",
        api_key="EMPTY"
    )

    start_time = time.time()

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="qwen3-asr-hebrew",
                file=f,
                language="he"
            )

        elapsed = time.time() - start_time
        hypothesis = result.text

        print("\n" + "-" * 70)
        print("RESULTS")
        print("-" * 70)
        print(f"Reference:    {reference}")
        print(f"Transcription: {hypothesis}")
        print("-" * 70)
        print(f"Latency: {elapsed:.2f}s")
        print(f"Real-time factor: {elapsed / audio_duration:.2f}x")
        print(f"Processing speed: {audio_duration / elapsed:.2f}x real-time")

        # Calculate simple character error rate (approximation)
        if reference and hypothesis:
            # Simple comparison (not proper WER, but gives an idea)
            ref_clean = reference.strip().lower()
            hyp_clean = hypothesis.strip().lower()

            if ref_clean == hyp_clean:
                print("✓ EXACT MATCH!")
            else:
                print(f"\nCharacter-level similarity: {len(set(ref_clean) & set(hyp_clean)) / max(len(set(ref_clean)), len(set(hyp_clean))) * 100:.1f}%")

        # Clean up temp file
        Path(tmp_path).unlink()

        return {
            "reference": reference,
            "hypothesis": hypothesis,
            "latency": elapsed,
            "audio_duration": audio_duration,
            "dataset": dataset_name
        }

    except Exception as e:
        print(f"\n✗ Transcription failed: {e}")
        Path(tmp_path).unlink()
        return None

def main():
    print("=" * 70)
    print("Testing Qwen3-ASR Hebrew on Lambda Labs H100")
    print("Using samples from Hebrew evaluation datasets")
    print("=" * 70)

    # Create SSH tunnel
    tunnel_process = create_ssh_tunnel()

    # Give tunnel time to establish
    if tunnel_process:
        time.sleep(1)

    # Test server
    import requests
    try:
        response = requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is accessible\n")
        else:
            print("✗ Server returned unexpected status")
            if tunnel_process:
                tunnel_process.terminate()
            return
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("\nTroubleshooting:")
        print(f"1. Ensure vLLM is running: ssh ubuntu@{LAMBDA_IP} 'curl localhost:8000/health'")
        print(f"2. Or create tunnel manually: ssh -L {LOCAL_PORT}:localhost:{LOCAL_PORT} ubuntu@{LAMBDA_IP}")
        if tunnel_process:
            tunnel_process.terminate()
        return

    # Test with samples from different datasets
    results = []

    for dataset_info in DATASETS:
        result = test_with_dataset_sample(*dataset_info)
        if result:
            results.append(result)

        # Small delay between tests
        time.sleep(1)

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        avg_latency = sum(r['latency'] for r in results) / len(results)
        avg_audio_duration = sum(r['audio_duration'] for r in results) / len(results)
        avg_rtf = sum(r['latency'] / r['audio_duration'] for r in results) / len(results)

        print(f"Tested: {len(results)} samples")
        print(f"Average latency: {avg_latency:.2f}s")
        print(f"Average audio duration: {avg_audio_duration:.2f}s")
        print(f"Average RTF: {avg_rtf:.2f}x")
        print(f"Average processing speed: {1/avg_rtf:.2f}x real-time")
        print("\nH100 Performance:")
        print(f"  - Processing {avg_audio_duration:.1f}s audio in {avg_latency:.2f}s")
        print(f"  - That's {1/avg_rtf:.1f}x faster than real-time!")
        print("=" * 70)

    # Clean up tunnel
    if tunnel_process:
        print("\nClosing SSH tunnel...")
        tunnel_process.terminate()
        tunnel_process.wait()

if __name__ == "__main__":
    main()
