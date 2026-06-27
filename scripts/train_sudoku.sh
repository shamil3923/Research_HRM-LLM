#!/bin/bash
# Training script for HRM on Sudoku-Extreme dataset
# Now uses complete YAML configuration system with all original HRM features

echo "🧩 HRM Sudoku Training Script"
echo "============================================"
echo "🔧 Features enabled:"
echo "   ✅ AdamATan2 optimizer (exact PyTorch port)"
echo "   ✅ Learning rate scheduling with warmup"
echo "   ✅ Dual optimizer (separate embedding LR)"
echo "   ✅ Q-learning exploration"
echo "   ✅ Stablemax cross entropy loss"
echo "   ✅ YAML configuration management"
echo "============================================"
echo ""

# Use the new YAML-based training script with Sudoku configuration
python train_yaml.py \
    --config config/cfg_sudoku.yaml \
    "$@"  # Allow additional args to override config

echo ""
echo "🎯 Training completed!"
echo "📊 Check checkpoints/ directory for saved models"
echo "📋 Configuration saved to checkpoints/config.yaml"