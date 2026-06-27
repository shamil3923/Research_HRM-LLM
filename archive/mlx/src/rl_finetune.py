"""
RL Fine-tuning script for HRMForMath using REINFORCE.

Applies Policy Gradient (REINFORCE) with dense reward shaping to optimize
the model for exact-match numerical accuracy, avoiding the local optimum
of just predicting the most common digits.

Usage:
    python src/rl_finetune.py --checkpoint checkpoints/gsm8k/best_model.npz
"""

import os
import sys
import argparse
import json
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np

from src.dataset import (
    GSM8KGraphDataset, OP_VOCAB, DIGIT_VOCAB, DIGIT_VOCAB_SIZE,
    MAX_DIGITS, decode_digits_to_number
)
from src.model import HRMForMath
from src.optimizer import TrainingOptimizer
from src.train import evaluate, final_node_digit_loss


# ─── Dense Reward Shaping ────────────────────────────────────────────────────

def compute_rewards(pred_digits, true_digits, true_raw, verbose=False):
    """
    Compute dense rewards for digit classification.
    
    Args:
        pred_digits: (B, MAX_DIGITS) — sampled digits
        true_digits: (B, MAX_DIGITS) — ground truth digits
        true_raw: (B,) — ground truth integer
        
    Returns:
        total_rewards: (B,)
        metrics: dict of reward components
    """
    B, D = pred_digits.shape
    
    # 1. Per-digit reward: +1.0 for each correct non-PAD digit position
    pad_token = DIGIT_VOCAB['PAD']
    valid_mask = true_digits != pad_token
    
    correct_digits = (pred_digits == true_digits) & valid_mask
    digit_reward = correct_digits.sum(axis=1) / mx.maximum(valid_mask.sum(axis=1), mx.array(1.0))
    # digit_reward is in range [0, 1]
    
    # 2. Magnitude reward: correct length (number of non-PAD digits)
    pred_len = (pred_digits != pad_token).sum(axis=1)
    true_len = valid_mask.sum(axis=1)
    magnitude_reward = (pred_len == true_len).astype(mx.float32) * 0.5
    
    # 3. Exact match reward (the ultimate goal)
    # Decode to integer to be safe against varying PAD representations
    pred_raw = mx.zeros((B,), dtype=mx.int32)
    for b in range(B):
        # Decode one by one (MLX doesn't have vectorized string decoding easily)
        digits_list = [int(pred_digits[b, d].item()) for d in range(D)]
        pred_raw[b] = decode_digits_to_number(digits_list)
    
    exact_match = (pred_raw == true_raw).astype(mx.float32)
    near_match = (mx.abs(pred_raw - true_raw) == 1).astype(mx.float32)
    
    # High reward for exact match, small consolation for near match
    exact_reward = exact_match * 5.0 + near_match * 1.0
    
    # Total dense reward
    total = digit_reward + magnitude_reward + exact_reward
    
    if verbose and mx.any(exact_match).item():
        print(f"RL Sample Hit! pred={pred_raw[0].item()} true={true_raw[0].item()} reward={total[0].item():.2f}")
        
    return total, {
        "reward/total": total.mean(),
        "reward/digit": digit_reward.mean(),
        "reward/exact": exact_match.mean(),
        "reward/magnitude": magnitude_reward.mean()
    }


# ─── REINFORCE Loss ──────────────────────────────────────────────────────────

def reinforce_loss(model, batch, baseline):
    """
    Compute REINFORCE policy gradient loss with cross-entropy regularization.
    """
    B = batch["inputs"].shape[0]
    
    # Forward pass
    digit_logits, q_halt, q_continue = model(batch)
    
    # Extract final node logits
    num_real = batch["num_real_nodes"]
    last_idx = mx.clip(num_real - 1, 0, digit_logits.shape[1] - 1)
    onehot = (mx.arange(digit_logits.shape[1]).reshape(1, -1) == last_idx.reshape(B, 1))
    
    # (B, MAX_DIGITS, VOCAB)
    final_logits = mx.sum(digit_logits * onehot[:, :, None, None].astype(mx.float32), axis=1)
    
    # Categorical sampling with Temperature for better exploration
    temperature = 1.5
    final_logits_scaled = final_logits / temperature
    gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=final_logits.shape) + 1e-10))
    sampled_digits = mx.argmax(final_logits_scaled + gumbel_noise, axis=-1)  # (B, MAX_DIGITS)
    
    # Compute rewards (stop gradient so rewards are constants)
    rewards, metrics = compute_rewards(
        sampled_digits, 
        batch["final_digit_target"], 
        batch["raw_target"]
    )
    rewards = mx.stop_gradient(rewards)
    
    # Compute advantage
    advantage = rewards - baseline
    
    # Policy Gradient Loss: -log π(sampled) * advantage
    # log_probs: (B, D, V)
    log_probs = final_logits - mx.logsumexp(final_logits, axis=-1, keepdims=True)
    
    # Gather log prob of the sampled digits
    targets_safe = mx.clip(sampled_digits, 0, DIGIT_VOCAB_SIZE - 1)
    targets_flat = targets_safe.reshape(B * MAX_DIGITS)
    log_probs_flat = log_probs.reshape(B * MAX_DIGITS, DIGIT_VOCAB_SIZE)
    idx = mx.arange(B * MAX_DIGITS)
    
    sampled_log_probs = log_probs_flat[idx, targets_flat].reshape(B, MAX_DIGITS)
    
    # Mask PAD tokens from the target (don't train on PAD)
    pad_mask = (batch["final_digit_target"] != DIGIT_VOCAB['PAD']).astype(mx.float32)
    
    # PG Loss = - mean( sum(log_prob * mask) * advantage )
    pg_loss = -mx.mean(mx.sum(sampled_log_probs * pad_mask, axis=1) * advantage)
    
    # CE Regularization (prevents policy collapse / entropy collapse)
    ce_loss = final_node_digit_loss(digit_logits, batch["final_digit_target"], num_real)
    
    # Entropy Bonus (encourages exploration)
    probs = mx.softmax(final_logits, axis=-1)
    entropy = -mx.sum(probs * log_probs, axis=-1)
    # Mask PAD positions for entropy
    entropy_masked = mx.sum(entropy * pad_mask, axis=1) / mx.maximum(mx.sum(pad_mask, axis=1), mx.array(1.0))
    entropy_bonus = mx.mean(entropy_masked)
    
    # Total loss: PG + CE - Entropy
    total_loss = pg_loss + 0.1 * ce_loss - 0.05 * entropy_bonus
    
    metrics["loss/pg"] = pg_loss
    metrics["loss/ce"] = ce_loss
    metrics["loss/entropy"] = entropy_bonus
    metrics["loss/total"] = total_loss
    metrics["advantage"] = advantage.mean()
    
    return total_loss, metrics


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.npz")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=64)  # Larger batch for RL stability
    p.add_argument("--peak_lr",    type=float, default=3e-5) # Slightly higher LR for exploration
    p.add_argument("--d_model",    type=int,   default=256)
    p.add_argument("--n_heads",    type=int,   default=8)
    p.add_argument("--H_cycles",   type=int,   default=2)
    p.add_argument("--L_cycles",   type=int,   default=4)
    p.add_argument("--H_layers",   type=int,   default=4)
    p.add_argument("--L_layers",   type=int,   default=4)
    p.add_argument("--data",       type=str,   default="data/gsm8k_train_clean.json")
    p.add_argument("--save_dir",   type=str,   default="checkpoints/gsm8k_rl")
    return p.parse_args()


def load_model(args):
    model = HRMForMath(
        vocab_size=len(OP_VOCAB),
        d_model=args.d_model,
        n_heads=args.n_heads,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        H_layers=args.H_layers,
        L_layers=args.L_layers,
        seq_len=50,
    )
    
    print(f"Loading checkpoint: {args.checkpoint}")
    weights = mx.load(args.checkpoint)
    
    # Convert flat / back to nested .
    from mlx.utils import tree_unflatten
    flat_weights = [(k.replace("/", "."), mx.array(v)) for k, v in weights.items()]
    nested_weights = tree_unflatten(flat_weights)
        
    model.update(nested_weights)
    mx.eval(model.parameters())
    return model


def train():
    args = parse_args()

    model = load_model(args)
    
    dataset = GSM8KGraphDataset(args.data, max_nodes=50)
    print(f"Dataset: {len(dataset)} samples")

    opt = TrainingOptimizer(
        peak_lr=args.peak_lr,
        warmup_steps=10,
        max_grad_norm=1.0,
        schedule="flat",
    )

    epochs = args.epochs
    batch_size = args.batch_size
    best_acc = 0.0
    
    # RL Baseline (Exponential Moving Average)
    baseline = mx.array(0.0)
    baseline_momentum = 0.95

    # Need a wrapper that returns just the loss for value_and_grad
    def loss_fn(model, batch, current_baseline):
        loss, metrics = reinforce_loss(model, batch, current_baseline)
        return loss, metrics

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"\nStarting REINFORCE Fine-tuning for {epochs} epochs")
    print(f"  Peak LR={args.peak_lr:.0e} | Batch={batch_size}")
    print("─" * 90)

    for epoch in range(epochs):
        epoch_metrics = {}
        steps = 0

        # We train on the full dataset for RL (no phase curriculum)
        for batch in dataset.get_batches(batch_size):
            (loss, metrics), grads = loss_and_grad(model, batch, baseline)
            gnorm = opt.step(model, grads)
            
            # Update baseline
            reward_mean = metrics["reward/total"]
            baseline = baseline_momentum * baseline + (1 - baseline_momentum) * reward_mean
            
            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v.item()
            epoch_metrics["gnorm"] = epoch_metrics.get("gnorm", 0.0) + gnorm
            steps += 1

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= max(steps, 1)

        # Evaluate
        exact_acc, digit_acc, near_acc = evaluate(model, dataset)
        improved = exact_acc > best_acc
        
        if improved:
            best_acc = exact_acc
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, "best_rl_model.npz")
            flat = tree_flatten(model.parameters())
            weights = {k.replace(".", "/"): v for k, v in flat}
            mx.savez(path, **weights)

        marker = " ★" if improved else ""
        print(
            f"Epoch {epoch+1:2d} | "
            f"R={epoch_metrics['reward/total']:.2f} (base={baseline.item():.2f}) "
            f"Loss(PG={epoch_metrics['loss/pg']:.3f} CE={epoch_metrics['loss/ce']:.3f}) | "
            f"Acc: Exact={exact_acc*100:.1f}% Digit={digit_acc*100:.1f}% Best={best_acc*100:.1f}%{marker}"
        )

    print("─" * 90)
    print(f"Done. Best RL exact-match accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    train()
