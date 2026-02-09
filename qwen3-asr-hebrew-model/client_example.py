#!/usr/bin/env python3
"""
Example client for Qwen3-ASR Hebrew API.

Usage:
    python client_example.py audio_file.wav
    python client_example.py audio_file.mp3 --server http://localhost:8000
"""

import base64
import argparse
from pathlib import Path

import requests


def transcribe_file(audio_path: str, server_url: str = "http://localhost:8000"):
    """Upload audio file and get transcription."""

    # Read audio file
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, "audio/wav")}

        response = requests.post(
            f"{server_url}/transcribe/file",
            files=files,
            params={"language": "Hebrew"}
        )

    response.raise_for_status()
    return response.json()


def transcribe_base64(audio_path: str, server_url: str = "http://localhost:8000"):
    """Send base64-encoded audio and get transcription."""

    # Read and encode audio
    with open(audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Send request
    response = requests.post(
        f"{server_url}/transcribe",
        json={
            "audio_base64": audio_base64,
            "language": "Hebrew",
            "sample_rate": 16000
        }
    )

    response.raise_for_status()
    return response.json()


def check_health(server_url: str = "http://localhost:8000"):
    """Check server health."""
    response = requests.get(f"{server_url}/health")
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Transcribe Hebrew audio")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--method",
        choices=["file", "base64"],
        default="file",
        help="Upload method (default: file)"
    )

    args = parser.parse_args()

    # Check health first
    print(f"Checking server health at {args.server}...")
    try:
        health = check_health(args.server)
        print(f"✓ Server status: {health['status']}")
        print(f"  Device: {health['device']}")
        if health.get('gpu_name'):
            print(f"  GPU: {health['gpu_name']}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Server not reachable: {e}")
        return

    # Transcribe
    print(f"\nTranscribing {args.audio_file}...")
    try:
        if args.method == "file":
            result = transcribe_file(args.audio_file, args.server)
        else:
            result = transcribe_base64(args.audio_file, args.server)

        print("\n" + "=" * 60)
        print("TRANSCRIPTION")
        print("=" * 60)
        print(result['text'])
        print("=" * 60)

        if 'duration_seconds' in result:
            print(f"\nDuration: {result['duration_seconds']:.2f}s")
        print(f"Language: {result['language']}")
        print(f"Model: {result['model']}")

    except requests.exceptions.RequestException as e:
        print(f"✗ Transcription failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"  Error: {e.response.text}")


if __name__ == "__main__":
    main()
