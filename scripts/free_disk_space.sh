#!/bin/bash

echo "========================================="
echo "Disk Space Analysis"
echo "========================================="
echo ""

echo "Current disk usage:"
df -h /

echo ""
echo "Largest directories in home (~/):"
du -sh ~/* 2>/dev/null | sort -h | tail -20

echo ""
echo "========================================="
echo "Cleanup Options"
echo "========================================="
echo ""

# Check HuggingFace cache
HF_CACHE=~/.cache/huggingface
if [ -d "$HF_CACHE" ]; then
    HF_SIZE=$(du -sh $HF_CACHE 2>/dev/null | cut -f1)
    echo "1. HuggingFace cache: $HF_SIZE"
    echo "   Clear: rm -rf $HF_CACHE/*"
fi

# Check pip cache
PIP_CACHE=~/.cache/pip
if [ -d "$PIP_CACHE" ]; then
    PIP_SIZE=$(du -sh $PIP_CACHE 2>/dev/null | cut -f1)
    echo "2. Pip cache: $PIP_SIZE"
    echo "   Clear: rm -rf $PIP_CACHE/*"
fi

# Check uv cache
UV_CACHE=~/.cache/uv
if [ -d "$UV_CACHE" ]; then
    UV_SIZE=$(du -sh $UV_CACHE 2>/dev/null | cut -f1)
    echo "3. UV cache: $UV_SIZE"
    echo "   Clear: rm -rf $UV_CACHE/*"
fi

# Check docker
if command -v docker &> /dev/null; then
    DOCKER_SIZE=$(docker system df 2>/dev/null | grep "Total" | awk '{print $4}')
    if [ ! -z "$DOCKER_SIZE" ]; then
        echo "4. Docker: $DOCKER_SIZE"
        echo "   Clear: docker system prune -a"
    fi
fi

# Check journal logs
JOURNAL_SIZE=$(du -sh /var/log/journal 2>/dev/null | cut -f1)
if [ ! -z "$JOURNAL_SIZE" ]; then
    echo "5. System logs: $JOURNAL_SIZE"
    echo "   Clear: sudo journalctl --vacuum-time=7d"
fi

echo ""
echo "========================================="
echo "Quick cleanup (run this):"
echo "========================================="
echo "rm -rf ~/.cache/huggingface/*"
echo "rm -rf ~/.cache/pip/*"
echo "rm -rf ~/.cache/uv/*"
echo ""
