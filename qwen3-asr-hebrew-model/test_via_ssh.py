#!/usr/bin/env python3
"""
Test Qwen3-ASR Hebrew via SSH tunnel to Lambda Labs H100.

This script creates an SSH tunnel to access the vLLM server running on Lambda.

Usage:
    python test_via_ssh.py audio_file.wav
"""

import subprocess
import time
import argparse
from pathlib import Path
from openai import OpenAI

LAMBDA_IP = "192.222.55.73"
LAMBDA_USER = "ubuntu"
LOCAL_PORT = 8000
REMOTE_PORT = 8000

def create_ssh_tunnel():
    """Create SSH tunnel to Lambda instance."""
    print(f"Creating SSH tunnel: localhost:{LOCAL_PORT} -> {LAMBDA_IP}:{REMOTE_PORT}")

    # Check if tunnel already exists
    check_cmd = f"lsof -ti:{LOCAL_PORT}"
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)

    if result.stdout.strip():
        print(f"✓ Port {LOCAL_PORT} already in use (tunnel may already exist)")
        return None

    # Create SSH tunnel in background
    tunnel_cmd = [
        "ssh",
        "-N",  # No remote command
        "-L", f"{LOCAL_PORT}:localhost:{REMOTE_PORT}",  # Port forwarding
        f"{LAMBDA_USER}@{LAMBDA_IP}"
    ]

    print(f"  Command: {' '.join(tunnel_cmd)}")
    print("  (This will run in background)")

    process = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for tunnel to establish
    time.sleep(2)

    # Check if process is still running
    if process.poll() is not None:
        print("✗ SSH tunnel failed to start")
        return None

    print(f"✓ SSH tunnel created (PID: {process.pid})")
    return process

def check_server():
    """Check if server is accessible."""
    import requests

    try:
        response = requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is accessible")
            return True
    except:
        pass

    print("✗ Server not accessible")
    return False

def transcribe_audio(audio_path: str, language: str = "he"):
    """Transcribe audio file."""
    client = OpenAI(
        base_url=f"http://localhost:{LOCAL_PORT}/v1",
        api_key="EMPTY"
    )

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
    print(f"Server: Lambda Labs H100 (via SSH tunnel)")

    return result.text

def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR Hebrew via SSH tunnel")
    parser.add_argument("audio_file", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--language", default="he", help="Language code (default: he)")
    parser.add_argument("--no-tunnel", action="store_true", help="Don't create SSH tunnel (assume it exists)")

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-ASR Hebrew on Lambda Labs H100 (via SSH)")
    print("=" * 60)

    # Create SSH tunnel
    tunnel_process = None
    if not args.no_tunnel:
        tunnel_process = create_ssh_tunnel()
        if tunnel_process is None and not check_server():
            print("\n✗ Failed to create tunnel or connect to server")
            print("\nTroubleshooting:")
            print("1. Ensure SSH access works: ssh ubuntu@192.222.55.73")
            print("2. Check if vLLM is running on Lambda:")
            print(f"   ssh {LAMBDA_USER}@{LAMBDA_IP} 'curl localhost:{REMOTE_PORT}/health'")
            print("3. Try creating tunnel manually:")
            print(f"   ssh -L {LOCAL_PORT}:localhost:{REMOTE_PORT} {LAMBDA_USER}@{LAMBDA_IP}")
            return

    # Wait a bit more for tunnel
    time.sleep(1)

    # Check server
    if not check_server():
        print("\n✗ Cannot connect to server")
        if tunnel_process:
            tunnel_process.terminate()
        return

    # Transcribe
    if args.audio_file:
        try:
            transcribe_audio(args.audio_file, args.language)
        finally:
            if tunnel_process:
                print("\nClosing SSH tunnel...")
                tunnel_process.terminate()
                tunnel_process.wait()
    else:
        print("\n✓ Server is ready for transcription")
        print(f"\nServer accessible at: http://localhost:{LOCAL_PORT}")
        print("\nUsage:")
        print(f"  python {Path(__file__).name} audio_file.wav")
        print("\nOr use OpenAI SDK:")
        print("  from openai import OpenAI")
        print(f"  client = OpenAI(base_url='http://localhost:{LOCAL_PORT}/v1', api_key='EMPTY')")
        print("  result = client.audio.transcriptions.create(model='qwen3-asr-hebrew', file=open('audio.wav', 'rb'), language='he')")

        if tunnel_process:
            print(f"\nSSH tunnel is running (PID: {tunnel_process.pid})")
            print("Press Ctrl+C to close tunnel")
            try:
                tunnel_process.wait()
            except KeyboardInterrupt:
                print("\nClosing SSH tunnel...")
                tunnel_process.terminate()

if __name__ == "__main__":
    main()
