#!/usr/bin/env python3
"""
Test Qwen3-ASR Hebrew model on Lambda Labs H100.

Usage:
    python test_h100_server.py audio_file.wav
    python test_h100_server.py --health  # Check server health
"""

import argparse
import time
from pathlib import Path
from openai import OpenAI

SERVER_URL = "http://192.222.55.73:8000/v1"

def check_health():
    """Check server health."""
    import requests

    print("Checking server health...")
    response = requests.get("http://192.222.55.73:8000/health")

    if response.status_code == 200:
        print("✓ Server is healthy")
        return True
    else:
        print(f"✗ Server health check failed: {response.status_code}")
        return False

def check_models():
    """List available models."""
    client = OpenAI(base_url=SERVER_URL, api_key="EMPTY")

    print("\nAvailable models:")
    models = client.models.list()
    for model in models.data:
        print(f"  - {model.id}")
    return models.data

def transcribe_audio(audio_path: str, language: str = "he"):
    """Transcribe audio file."""
    client = OpenAI(base_url=SERVER_URL, api_key="EMPTY")

    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"✗ Audio file not found: {audio_path}")
        return None

    print(f"\nTranscribing: {audio_file.name}")
    print(f"File size: {audio_file.stat().st_size / 1024:.1f} KB")

    start_time = time.time()

    with open(audio_file, "rb") as f:
        result = client.audio.transcriptions.create(
            model="qwen3-asr-hebrew",
            file=f,
            language=language
        )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT")
    print("=" * 60)
    print(result.text)
    print("=" * 60)
    print(f"\nLatency: {elapsed:.2f}s")

    return result.text

def benchmark_performance(audio_path: str, num_requests: int = 10):
    """Benchmark server performance."""
    client = OpenAI(base_url=SERVER_URL, api_key="EMPTY")

    print(f"\nBenchmarking with {num_requests} requests...")

    latencies = []

    for i in range(num_requests):
        start_time = time.time()

        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="qwen3-asr-hebrew",
                file=f,
                language="he"
            )

        elapsed = time.time() - start_time
        latencies.append(elapsed)

        print(f"  Request {i+1}/{num_requests}: {elapsed:.2f}s")

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total requests: {num_requests}")
    print(f"Average latency: {sum(latencies)/len(latencies):.2f}s")
    print(f"Min latency: {min(latencies):.2f}s")
    print(f"Max latency: {max(latencies):.2f}s")
    print(f"Throughput: {num_requests / sum(latencies):.2f} req/s")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR Hebrew on H100")
    parser.add_argument("audio_file", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--health", action="store_true", help="Check server health")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests for benchmark")
    parser.add_argument("--language", default="he", help="Language code (default: he)")

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-ASR Hebrew on Lambda Labs H100")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print("=" * 60)

    # Check health
    if args.health or not args.audio_file:
        healthy = check_health()
        check_models()
        if not args.audio_file:
            return
        if not healthy:
            print("\n✗ Server is not healthy. Exiting.")
            return

    # Transcribe
    if args.audio_file and not args.benchmark:
        transcribe_audio(args.audio_file, args.language)

    # Benchmark
    if args.benchmark and args.audio_file:
        benchmark_performance(args.audio_file, args.num_requests)

if __name__ == "__main__":
    main()
