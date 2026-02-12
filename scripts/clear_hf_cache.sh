#!/bin/bash

# Clear HuggingFace cache to free up disk space

echo "Current HuggingFace cache usage:"
du -sh ~/.cache/huggingface/ 2>/dev/null || echo "No cache found"

echo ""
echo "Clearing HuggingFace datasets cache..."
rm -rf ~/.cache/huggingface/datasets/*

echo "Clearing HuggingFace hub cache..."
rm -rf ~/.cache/huggingface/hub/*

echo ""
echo "Cache cleared!"
echo ""
echo "Free disk space:"
df -h / | grep -E 'Filesystem|/dev/root'
