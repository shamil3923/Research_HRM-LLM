#!/bin/bash
# Quick training script for testing with small model
# Uses reduced model size and dataset for rapid iteration

echo "🧪 HRM Small Model Training (for testing)"
echo "=========================================="
echo "🔧 Small configuration:"
echo "   📏 Model: 256d, 1×1 cycles, 2+2 layers"
echo "   📊 Data: 50 training, 10 validation samples"
echo "   ⚡ Fast: 50 warmup steps, 4 ACT steps"
echo "=========================================="
echo ""

python train_yaml.py \
    --config config/cfg_small.yaml \
    "$@"

echo ""
echo "🎯 Small model training completed!"