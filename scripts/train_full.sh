#!/bin/bash
# Full-scale HRM training with all original parameters
# Uses the complete configuration matching the PyTorch implementation

echo "🚀 HRM Full-Scale Training"
echo "=========================="
echo "🔧 Full configuration:"
echo "   📏 Model: 512d, 2×2 cycles, 4+4 layers (17.8M params)"
echo "   📊 Data: 1000 training, 200 validation samples"
echo "   ⏱️  Schedule: 2000 warmup steps, cosine decay"
echo "   🎯 Target: 20000 epochs, halt_max_steps=8"
echo "=========================="
echo ""

python train_yaml.py \
    --config config/cfg_pretrain.yaml \
    --halt_max_steps 8 \
    --train_samples 1000 \
    --val_samples 200 \
    --min_difficulty 20 \
    "$@"

echo ""
echo "🎯 Full-scale training completed!"