#!/bin/bash
# Optimize Lambda GPU for maximum parallel processing

echo "========================================="
echo "Optimizing Lambda GPU for Data Prep"
echo "========================================="
echo ""

# Check current limits
echo "Current file descriptor limits:"
ulimit -n
ulimit -Sn  # Soft limit
ulimit -Hn  # Hard limit

echo ""
echo "Setting file descriptor limit to maximum..."
# Set to maximum (usually 1048576 on Lambda)
ulimit -n 1048576

echo ""
echo "New limits:"
ulimit -n

echo ""
echo "System info:"
echo "  CPUs: $(nproc)"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "  Disk: $(df -h . | tail -1 | awk '{print $4}') free"

echo ""
echo "========================================="
echo "Recommended settings for 124 cores:"
echo "========================================="
echo ""
echo "Download workers (num_proc): 32-64"
echo "  - Network I/O bound, diminishing returns beyond 32"
echo ""
echo "Processing workers: 100-110"
echo "  - CPU bound (audio decoding), use ~90% of cores"
echo ""
echo "Suggested command:"
echo "  uv run python prepare_qwen_data.py --workers 100"
echo ""
