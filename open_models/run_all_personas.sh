#!/bin/bash
set -e

cd "$(dirname "$0")"

# echo "=========================================="
# echo "Training empty persona..."
# echo "=========================================="
# python training.py train_empty.json

echo "=========================================="
echo "Training default persona..."
echo "=========================================="
python training.py train_default.json

echo "=========================================="
echo "Training genie persona..."
echo "=========================================="
python training.py train_genie.json

echo "=========================================="
echo "Training anti-genie persona..."
echo "=========================================="
python training.py train_anti-genie.json

echo "=========================================="
echo "All persona training complete!"
echo "=========================================="
