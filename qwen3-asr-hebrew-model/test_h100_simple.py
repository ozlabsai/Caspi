#!/usr/bin/env python3
"""
Simple test of the H100 deployment using a basic audio file.
"""

import subprocess
import tempfile
import time
import numpy as np
import soundfile as sf
from openai import OpenAI

LAMBDA_IP = "209.20.159.167"
LOCAL_PORT = 8000

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
    print("Simple Test: Qwen3-ASR Hebrew on Lambda Labs H100")
    print("=" * 70)

    # Create tunnel
    tunnel_process = create_ssh_tunnel()
    if tunnel_process:
        time.sleep(1)

    # Check server
    try:
        import requests
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

    # Create a simple test audio (1 second sine wave)
    print("Creating test audio (1 second)...")
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        tmp_path = tmp.name

    print(f"✓ Created test audio\n")

    # Transcribe
    print("Transcribing with vLLM on H100...")
    print("-" * 70)

    client = OpenAI(
        base_url=f"http://localhost:{LOCAL_PORT}/v1",
        api_key="EMPTY"
    )

    start_time = time.time()

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="./model",
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

        print("\n✓ Test successful! Server is working correctly.")

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
