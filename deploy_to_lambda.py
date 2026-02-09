#!/usr/bin/env python3
"""
Automated Lambda Labs deployment script for Qwen3-ASR Hebrew training.
Handles instance launch, file upload, training execution, and cleanup.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Lambda Labs API configuration
LAMBDA_API_KEY = "secret_caspi_8cd6b8fc387c4be199fdfaa8b45c6c74.DGXQ21Z2ITZUSfxpW9UUU5xxxYtMw3XB"
INSTANCE_TYPE = "gpu_1x_a10"  # 1x A10 (24GB)
REGION = "us-west-1"  # Default region
INSTANCE_NAME = "qwen3-asr-hebrew-training"

# Files to upload
FILES_TO_UPLOAD = [
    "train_qwen3_asr_official.py",
    "train_on_lambda.sh"
]

# SSH configuration
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")


def run_command(cmd, check=True, capture_output=True):
    """Run a shell command and return output."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=capture_output, text=True)
    if capture_output:
        return result.stdout.strip()
    return ""


def setup_lambda_cli():
    """Configure Lambda Labs CLI with API key."""
    print("\n" + "=" * 60)
    print("Configuring Lambda Labs CLI")
    print("=" * 60)

    # Set API key in environment
    os.environ["LAMBDA_API_KEY"] = LAMBDA_API_KEY

    # Test API connection
    try:
        import lambda_cloud
        client = lambda_cloud.LambdaCloudClient(api_key=LAMBDA_API_KEY)
        print("✓ Lambda Labs API authenticated successfully")
        return client
    except Exception as e:
        print(f"✗ Failed to authenticate: {e}")
        sys.exit(1)


def check_ssh_key():
    """Ensure SSH key exists."""
    if not Path(SSH_KEY_PATH).exists():
        print(f"✗ SSH key not found at {SSH_KEY_PATH}")
        print("  Please generate one with: ssh-keygen -t rsa -b 4096")
        sys.exit(1)
    print(f"✓ SSH key found at {SSH_KEY_PATH}")


def list_available_instances(client):
    """List available instance types."""
    print("\n" + "=" * 60)
    print("Checking Available Instance Types")
    print("=" * 60)

    try:
        instance_types = client.instance_types()
        print(f"✓ Found {len(instance_types)} instance types")

        # Find A10 instances
        for it in instance_types:
            if "a10" in it["name"].lower():
                print(f"  {it['name']}: ${it['price_cents_per_hour']/100:.2f}/hour")
                print(f"    Regions: {', '.join(it['regions_with_capacity_available'])}")

        return instance_types
    except Exception as e:
        print(f"✗ Failed to list instances: {e}")
        return []


def launch_instance(client):
    """Launch a Lambda Labs instance."""
    print("\n" + "=" * 60)
    print("Launching Lambda Labs Instance")
    print("=" * 60)

    try:
        # Upload SSH public key
        with open(SSH_KEY_PATH + ".pub", "r") as f:
            ssh_pub_key = f.read().strip()

        # Launch instance
        print(f"  Instance type: {INSTANCE_TYPE}")
        print(f"  Name: {INSTANCE_NAME}")
        print("  Launching... (this may take 1-2 minutes)")

        instance = client.launch_instance(
            instance_type_name=INSTANCE_TYPE,
            region_name=REGION,
            ssh_key_names=[],  # Will use the key we provide
            name=INSTANCE_NAME
        )

        instance_id = instance["data"]["instance_ids"][0]
        print(f"✓ Instance launched: {instance_id}")

        # Wait for instance to be ready
        print("  Waiting for instance to be active...")
        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            instances = client.list_instances()
            for inst in instances["data"]:
                if inst["id"] == instance_id:
                    if inst["status"] == "active":
                        ip_address = inst["ip"]
                        print(f"✓ Instance active! IP: {ip_address}")
                        return instance_id, ip_address

            time.sleep(10)
            print("  Still waiting...")

        print("✗ Timeout waiting for instance to become active")
        return None, None

    except Exception as e:
        print(f"✗ Failed to launch instance: {e}")
        print(f"  Error details: {str(e)}")
        return None, None


def upload_files(ip_address):
    """Upload training files to the instance."""
    print("\n" + "=" * 60)
    print("Uploading Training Files")
    print("=" * 60)

    # Wait a bit for SSH to be ready
    print("  Waiting for SSH to be ready...")
    time.sleep(30)

    for file in FILES_TO_UPLOAD:
        if not Path(file).exists():
            print(f"✗ File not found: {file}")
            continue

        print(f"  Uploading {file}...")
        try:
            run_command([
                "scp",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-i", SSH_KEY_PATH,
                file,
                f"ubuntu@{ip_address}:~/"
            ])
            print(f"✓ Uploaded {file}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to upload {file}: {e}")
            return False

    # Make script executable
    try:
        run_command([
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-i", SSH_KEY_PATH,
            f"ubuntu@{ip_address}",
            "chmod +x train_on_lambda.sh"
        ])
        print("✓ Made script executable")
    except subprocess.CalledProcessError:
        print("✗ Failed to make script executable")
        return False

    return True


def start_training(ip_address):
    """Start training on the remote instance."""
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("✗ HF_TOKEN environment variable not set")
        return False

    print("  Starting training in tmux session...")

    training_cmd = f"""
        tmux new-session -d -s training
        tmux send-keys -t training 'export HF_TOKEN={hf_token}' C-m
        tmux send-keys -t training './train_on_lambda.sh 2>&1 | tee training.log' C-m
    """

    try:
        run_command([
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-i", SSH_KEY_PATH,
            f"ubuntu@{ip_address}",
            training_cmd.strip()
        ])
        print("✓ Training started in tmux session 'training'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to start training: {e}")
        return False


def monitor_training(ip_address):
    """Monitor training progress."""
    print("\n" + "=" * 60)
    print("Training Monitoring Instructions")
    print("=" * 60)

    print(f"\nTo monitor training progress:")
    print(f"  ssh -i {SSH_KEY_PATH} ubuntu@{ip_address}")
    print(f"  tmux attach -t training")
    print(f"\nTo detach from tmux: Ctrl+B, then D")
    print(f"\nTo check logs: tail -f ~/training.log")
    print(f"\nInstance IP: {ip_address}")
    print(f"Estimated training time: 6-8 hours")
    print(f"Estimated cost: $6-7 total")


def main():
    """Main deployment orchestration."""
    print("=" * 60)
    print("Qwen3-ASR Hebrew Training - Lambda Labs Deployment")
    print("=" * 60)

    # Setup
    check_ssh_key()
    client = setup_lambda_cli()

    # Check available instances
    list_available_instances(client)

    # Launch instance
    instance_id, ip_address = launch_instance(client)
    if not instance_id:
        print("\n✗ Failed to launch instance. Exiting.")
        sys.exit(1)

    # Upload files
    if not upload_files(ip_address):
        print("\n✗ Failed to upload files. Instance is still running.")
        print(f"  Terminate manually: lambda cloud instance terminate {instance_id}")
        sys.exit(1)

    # Start training
    if not start_training(ip_address):
        print("\n✗ Failed to start training. Instance is still running.")
        print(f"  Terminate manually: lambda cloud instance terminate {instance_id}")
        sys.exit(1)

    # Show monitoring instructions
    monitor_training(ip_address)

    print("\n" + "=" * 60)
    print("Deployment Complete!")
    print("=" * 60)
    print(f"\nInstance ID: {instance_id}")
    print(f"Instance IP: {ip_address}")
    print(f"\n⚠️  IMPORTANT: Don't forget to terminate the instance when done!")
    print(f"  lambda cloud instance terminate {instance_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
