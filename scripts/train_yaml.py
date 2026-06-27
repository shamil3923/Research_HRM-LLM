#!/usr/bin/env python3
"""
HRM Training Script with YAML Configuration Support
Based on the original HRM pretrain.py with config system
"""

import argparse
import time
import numpy as np
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_adam_atan2_exact import AdamATan2Exact
from dual_optimizer import DualAdamATan2
from lr_scheduler import CosineScheduleWithWarmup
from config_loader import load_config, HRMConfig
import mlx.utils

# Import model components from the new structure
from models.hrm import HierarchicalReasoningModel, HRMCarry
from models.losses import compute_act_loss


def load_sudoku_data(data_path: str, split: str = "train", max_samples: int = 1000, min_difficulty: int = 0):
    """Load official Sudoku-Extreme data and add special tokens"""
    
    from load_official_sudoku import load_official_sudoku_data

    csv_file = f"{data_path}/sudoku-extreme/{split}.csv"

    print(f"Loading official Sudoku-Extreme data from {csv_file}")

    puzzles, solutions = load_official_sudoku_data(
        csv_file,
        max_samples=max_samples,
        min_difficulty=min_difficulty,
        shuffle_augment=True,
        num_augmentations=1 if split == "train" else 0
    )

    # Use raw sequences (length 81) to match dataset builder format
    # No special tokens needed - vocab is 0-10 where 0=blank, 1-9=digits, 10=pad
    
    processed_puzzles = []
    processed_solutions = []

    for puzzle, solution in zip(puzzles, solutions):
        # Convert to list if numpy array
        if hasattr(puzzle, 'tolist'):
            puzzle = puzzle.tolist()
        if hasattr(solution, 'tolist'):
            solution = solution.tolist()

        # Use sequences as-is (length 81, no special tokens)
        processed_puzzles.append(puzzle)
        processed_solutions.append(solution)

    return processed_puzzles, processed_solutions


def train_hrm(config: HRMConfig):
    """Main training function"""
    
    print("🧩 HRM Training (MLX Implementation)")
    print("=" * 60)
    print("🔧 Based on github.com/sapientinc/HRM")
    print("💾 Complete implementation with ACT and Q-learning")
    print("🎯 Exact mathematical match to original PyTorch version")
    print()

    # Load data
    print("📊 Loading official Sudoku-Extreme datasets...")
    train_puzzles, train_solutions = load_sudoku_data(
        config.data_path, "train", config.train_samples, config.min_difficulty
    )
    val_puzzles, val_solutions = load_sudoku_data(
        config.data_path, "test", config.val_samples, config.min_difficulty
    )

    print(f"✅ Train samples: {len(train_puzzles):,}")
    print(f"✅ Val samples: {len(val_puzzles):,}")
    print()

    # Create model
    print("🤖 Creating HRM model...")
    model = HierarchicalReasoningModel(
        vocab_size=11,
        d_model=config.d_model,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        halt_max_steps=config.halt_max_steps,
        halt_exploration_prob=config.halt_exploration_prob,
        expansion=config.expansion,
    )

    # Count parameters
    total_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    print(f"✅ Model parameters: {total_params:,}")
    print(f"✅ Architecture: {config.H_cycles}×{config.L_cycles} cycles, {config.H_layers}+{config.L_layers} layers")
    print()

    # Create trainer (reuse the existing HRMTrainer class from pretrain.py)
    from pretrain import HRMTrainer
    
    trainer = HRMTrainer(
        model=model,
        train_data=(train_puzzles, train_solutions),
        val_data=(val_puzzles, val_solutions),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        eval_interval=config.eval_interval,
        warmup_steps=config.warmup_steps,
        min_lr_ratio=config.min_lr_ratio,
        embedding_lr=config.embedding_lr,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # Set checkpoint directory
    trainer.checkpoint_dir = config.checkpoint_dir
    os.makedirs(trainer.checkpoint_dir, exist_ok=True)

    # Auto-resume logic
    from pretrain import find_latest_checkpoint
    
    checkpoint_to_load = None
    if config.load_checkpoint:
        checkpoint_to_load = config.load_checkpoint
        print(f"📂 Loading specified checkpoint: {checkpoint_to_load}")
    elif not config.no_auto_resume:
        latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)
        if latest_checkpoint:
            checkpoint_to_load = latest_checkpoint
            print(f"🔄 Auto-resuming from latest checkpoint: {os.path.basename(latest_checkpoint)}")
        else:
            print("🆕 No existing checkpoints found, starting fresh training")
    else:
        print("🆕 Auto-resume disabled, starting fresh training")

    if checkpoint_to_load:
        trainer.load_checkpoint(checkpoint_to_load)

    # Save configuration
    config_save_path = os.path.join(config.checkpoint_dir, "config.yaml")
    config.save_yaml(config_save_path)
    print(f"💾 Configuration saved to: {config_save_path}")

    print()
    print("=" * 60)
    print("🚀 STARTING HRM TRAINING")
    print("=" * 60)

    # Train
    trainer.train()

    print()
    print("=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Best Validation Accuracy: {trainer.best_val_accuracy:.3f}")


def main():
    parser = argparse.ArgumentParser(description="HRM Training with YAML Configuration")
    
    # Configuration file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    
    # All the same arguments as before (for overrides)
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--H_cycles", type=int, help="High-level cycles")
    parser.add_argument("--L_cycles", type=int, help="Low-level cycles")
    parser.add_argument("--H_layers", type=int, help="High-level layers")
    parser.add_argument("--L_layers", type=int, help="Low-level layers")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--max_epochs", type=int, help="Maximum epochs")
    parser.add_argument("--train_samples", type=int, help="Training samples")
    parser.add_argument("--val_samples", type=int, help="Validation samples")
    parser.add_argument("--min_difficulty", type=int, help="Minimum puzzle difficulty")
    parser.add_argument("--data_path", type=str, help="Path to data directory")
    parser.add_argument("--checkpoint_dir", type=str, help="Checkpoint directory")
    parser.add_argument("--load_checkpoint", type=str, help="Path to checkpoint to load")
    parser.add_argument("--save_every", type=int, help="Save checkpoint every N steps")
    parser.add_argument("--halt_max_steps", type=int, help="Maximum ACT steps")
    parser.add_argument("--halt_exploration_prob", type=float, help="Q-learning exploration probability")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for optimizer")
    parser.add_argument("--warmup_steps", type=int, help="Learning rate warmup steps")
    parser.add_argument("--min_lr_ratio", type=float, help="Minimum LR ratio for cosine schedule")
    parser.add_argument("--embedding_lr", type=float, help="Separate learning rate for embeddings")
    parser.add_argument("--no_auto_resume", action="store_true", help="Disable automatic checkpoint resuming")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args)
    
    print("📋 Final configuration:")
    for key, value in config.to_dict().items():
        print(f"   {key}: {value}")
    print()

    # Train
    train_hrm(config)


if __name__ == "__main__":
    main()