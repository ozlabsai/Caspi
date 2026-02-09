#!/usr/bin/env python3
"""
Quick test of H100 vLLM server with a Hebrew audio sample.
"""

import subprocess
import tempfile
import time
import requests
from openai import OpenAI

LAMBDA_IP = "209.20.159.167"
LOCAL_PORT = 8000

# Public Hebrew audio sample from HuggingFace
SAMPLE_URL = "https://huggingface.co/datasets/ivrit-ai/eval-d1/resolve/main/audio/test/audio_0.wav"

def create_ssh_tunnel():
    """Create SSH tunnel if needed."""
    check_cmd = f"lsof -ti:{LOCAL_PORT}"
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)

    if result.stdout.strip():
        print(f"✓ SSH tunnel already active")
        return None

    print(f"Creating SSH tunnel to {LAMBDA_IP}...")
    process = subprocess.Popen(
        ["ssh", "-N", "-L", f"{LOCAL_PORT}:localhost:{LOCAL_PORT}", f"ubuntu@{LAMBDA_IP}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(2)

    if process.poll() is not None:
        print("✗ Failed to create SSH tunnel")
        return None

    print(f"✓ SSH tunnel created (PID: {process.pid})")
    return process

def main():
    print("=" * 70)
    print("Quick Test: Qwen3-ASR Hebrew on Lambda Labs H100")
    print("=" * 70)

    # Create tunnel
    tunnel_process = create_ssh_tunnel()
    if tunnel_process:
        time.sleep(1)

    # Check server
    try:
        response = requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is accessible\n")
        else:
            print("✗ Server health check failed")
            if tunnel_process:
                tunnel_process.terminate()
            return
    except Exception as e:
        print(f"✗ Cannot connect: {e}")
        if tunnel_process:
            tunnel_process.terminate()
        return

    # Download sample audio
    print("Downloading Hebrew audio sample...")
    try:
        audio_response = requests.get(SAMPLE_URL, timeout=30)
        audio_response.raise_for_status()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_response.content)
            tmp_path = tmp.name

        print(f"✓ Downloaded sample ({len(audio_response.content) / 1024:.1f} KB)")

    except Exception as e:
        print(f"✗ Failed to download sample: {e}")
        if tunnel_process:
            tunnel_process.terminate()
        return

    # Transcribe
    print("\nTranscribing with vLLM on H100...")
    print("-" * 70)

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

        print("\nRESULT:")
        print(f"  Transcription: {result.text}")
        print(f"  Latency: {elapsed:.2f}s")
        print(f"  Model: Qwen3-ASR Hebrew (fine-tuned)")
        print(f"  Hardware: Lambda Labs H100 SXM5")
        print("-" * 70)

        print("\n✓ Test successful!")

        # Clean up
        import os
        os.unlink(tmp_path)

    except Exception as e:
        print(f"\n✗ Transcription failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if tunnel_process:
            print("\nClosing SSH tunnel...")
            tunnel_process.terminate()
            tunnel_process.wait()

if __name__ == "__main__":
    main()
