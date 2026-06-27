"""
Training script for HRMForMath on GSM8K.

Trains the HRM with digit-level classification (cross-entropy) instead of
regression, matching the paper's approach for Sudoku.

Usage:
    python src/train.py                     # defaults
    python src/train.py --epochs 120        # override

Design:
  - Loss: cross-entropy over digit tokens (per-node deep supervision)
  - ACT loss: Q-halt/Q-continue binary cross-entropy
  - Curriculum: Phase 0 (1-2 steps) → Phase 1 (3-5 steps) → Phase 2 (all)
  - Optimizer: Adam + flat LR with warmup, warm-restart at phase transitions
  - Gradient clipping: global norm ≤ 0.5
  - Checkpoint: saves when exact-match accuracy improves
"""

import os
import sys
import argparse
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from src.dataset import (
    GSM8KGraphDataset, OP_VOCAB, DIGIT_VOCAB, DIGIT_VOCAB_SIZE,
    MAX_DIGITS, decode_digits_to_number
)
from src.model import HRMForMath
from src.optimizer import TrainingOptimizer


# ─── Loss Functions ──────────────────────────────────────────────────────────

def digit_cross_entropy_loss(digit_logits, digit_targets, num_real_nodes):
    """
    Cross-entropy loss over digit tokens for all real nodes.
    
    Args:
        digit_logits: (B, N, MAX_DIGITS, VOCAB) — model predictions
        digit_targets: (B, N, MAX_DIGITS) — ground truth digit tokens
        num_real_nodes: (B,) — number of non-PAD nodes per sample
    
    Returns:
        Scalar loss
    """
    B, N, D, V = digit_logits.shape
    
    # Create node mask: which nodes are real (not padding)
    positions = mx.arange(N).reshape(1, N)  # (1, N)
    node_mask = (positions < num_real_nodes.reshape(B, 1))  # (B, N) bool
    
    # Create digit mask: which digit positions are not PAD (token 0)
    digit_mask = (digit_targets != DIGIT_VOCAB['PAD'])  # (B, N, D) bool
    
    # Combined mask: real node AND non-PAD digit
    combined_mask = (node_mask[:, :, None] & digit_mask).astype(mx.float32)  # (B, N, D)
    
    # Log-softmax for numerical stability
    logits_f32 = digit_logits.astype(mx.float32)
    log_probs = logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True)
    
    # Gather log probabilities for target tokens
    # We need to index: log_probs[b, n, d, digit_targets[b, n, d]]
    targets_safe = mx.clip(digit_targets, 0, V - 1)  # Safety clamp
    
    # Reshape for gather
    targets_flat = targets_safe.reshape(B * N * D)
    log_probs_flat = log_probs.reshape(B * N * D, V)
    
    # Gather: pick the log probability of the target token at each position
    idx = mx.arange(B * N * D)
    target_log_probs = log_probs_flat[idx, targets_flat]  # (B*N*D,)
    target_log_probs = target_log_probs.reshape(B, N, D)
    
    # Masked cross-entropy
    masked_loss = -target_log_probs * combined_mask
    total = mx.sum(masked_loss)
    count = mx.maximum(mx.sum(combined_mask), mx.array(1.0))
    
    return total / count


def final_node_digit_loss(digit_logits, final_digit_target, num_real_nodes):
    """
    Cross-entropy loss specifically on the final answer node's digit predictions.
    
    This is the primary objective — getting the final answer right.
    """
    B, N, D, V = digit_logits.shape
    
    # Get the index of the last real node per sample
    last_idx = mx.clip(num_real_nodes - 1, 0, N - 1)  # (B,)
    
    # One-hot mask to select the final node
    onehot = (mx.arange(N).reshape(1, N) == last_idx.reshape(B, 1))  # (B, N) bool
    onehot_f = onehot.astype(mx.float32)  # (B, N)
    
    # Gather final node logits: (B, N, D, V) × (B, N, 1, 1) → sum → (B, D, V)
    final_logits = mx.sum(
        digit_logits * onehot_f[:, :, None, None],
        axis=1
    )  # (B, D, V)
    
    # Digit mask: non-PAD positions in the target
    digit_mask = (final_digit_target != DIGIT_VOCAB['PAD']).astype(mx.float32)  # (B, D)
    
    # Log-softmax
    logits_f32 = final_logits.astype(mx.float32)
    log_probs = logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True)
    
    # Gather target log probs
    targets_safe = mx.clip(final_digit_target, 0, V - 1)
    targets_flat = targets_safe.reshape(B * D)
    log_probs_flat = log_probs.reshape(B * D, V)
    idx = mx.arange(B * D)
    target_log_probs = log_probs_flat[idx, targets_flat].reshape(B, D)
    
    # Masked loss
    masked_loss = -target_log_probs * digit_mask
    total = mx.sum(masked_loss)
    count = mx.maximum(mx.sum(digit_mask), mx.array(1.0))
    
    return total / count


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, dataset, batch_size=64, verbose=False):
    """
    Evaluate exact-match accuracy and per-digit accuracy.
    
    Returns: (exact_match_acc, digit_acc, near_match_acc)
    """
    model.eval()
    exact_correct = 0
    near_correct = 0  # Within ±1
    total_digits_correct = 0
    total_digits = 0
    total = 0
    errors = []

    for batch in dataset.get_batches(batch_size, shuffle=False):
        digit_logits, _, _ = model(batch)
        num_real = batch["num_real_nodes"]
        raw_targets = batch["raw_target"]
        
        mx.eval(digit_logits, num_real, raw_targets)
        
        B, N, D, V = digit_logits.shape
        
        # Get final node predictions
        last_idx = mx.clip(num_real - 1, 0, N - 1)
        
        for b in range(B):
            li = int(last_idx[b].item())
            logits_b = digit_logits[b, li]  # (MAX_DIGITS, VOCAB)
            pred_digits = mx.argmax(logits_b, axis=-1)  # (MAX_DIGITS,)
            mx.eval(pred_digits)
            
            pred_list = [int(pred_digits[d].item()) for d in range(D)]
            pred_int = decode_digits_to_number(pred_list)
            true_int = int(raw_targets[b].item())
            
            # Exact match
            if pred_int == true_int:
                exact_correct += 1
            elif abs(pred_int - true_int) <= 1:
                near_correct += 1
            elif verbose and len(errors) < 5:
                errors.append(f"  pred={pred_int}, true={true_int}, digits={pred_list[:6]}")
            
            # Per-digit accuracy (on final answer digits)
            true_digits = batch["final_digit_target"][b]
            mx.eval(true_digits)
            for d in range(D):
                td = int(true_digits[d].item())
                if td != DIGIT_VOCAB['PAD']:
                    total_digits += 1
                    if int(pred_digits[d].item()) == td:
                        total_digits_correct += 1
            
            total += 1

    if verbose and errors:
        print("  Sample errors:")
        for e in errors:
            print(e)

    exact_acc = exact_correct / max(1, total)
    near_acc = (exact_correct + near_correct) / max(1, total)
    digit_acc = total_digits_correct / max(1, total_digits)
    
    return exact_acc, digit_acc, near_acc


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(model, save_dir, epoch, acc, meta=None):
    """Save model weights + metadata."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "best_model.npz")
    
    flat = tree_flatten(model.parameters())
    weights = {k.replace(".", "/"): v for k, v in flat}
    mx.savez(path, **weights)
    
    meta_path = os.path.join(save_dir, "best_meta.json")
    info = {"epoch": epoch, "accuracy": acc, **(meta or {})}
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"  ✓ Checkpoint saved to {path} (epoch={epoch}, acc={acc*100:.2f}%)")


# ─── Curriculum Dataset ──────────────────────────────────────────────────────

class CurriculumDataset:
    """
    Three-phase curriculum ordering by computation step count.
    Phase 0: ≤2 steps (simplest problems)
    Phase 1: ≤5 steps (medium)
    Phase 2: all problems
    """
    def __init__(self, dataset, phase0_max=2, phase1_max=5):
        self.dataset = dataset
        self.phase = 0
        self.phase0_idx = [i for i, d in enumerate(dataset.data) if d["step_count"] <= phase0_max]
        self.phase1_idx = [i for i, d in enumerate(dataset.data) if phase0_max < d["step_count"] <= phase1_max]
        self.phase2_idx = [i for i, d in enumerate(dataset.data) if d["step_count"] > phase1_max]
        print(f"Curriculum: P0={len(self.phase0_idx)} (≤{phase0_max} steps) | "
              f"P1={len(self.phase1_idx)} ({phase0_max+1}-{phase1_max} steps) | "
              f"P2={len(self.phase2_idx)} (>{phase1_max} steps)")

    def set_phase(self, epoch, total_epochs):
        if epoch < total_epochs // 3:
            self.phase = 0
        elif epoch < 2 * total_epochs // 3:
            self.phase = 1
        else:
            self.phase = 2

    def active_indices(self):
        if self.phase == 0:
            return self.phase0_idx
        elif self.phase == 1:
            return self.phase0_idx + self.phase1_idx
        else:
            return list(range(len(self.dataset.data)))

    def get_batches(self, batch_size):
        indices = np.random.permutation(self.active_indices())
        data = self.dataset.data
        for i in range(0, len(indices), batch_size):
            idx = indices[i:i + batch_size]
            yield {
                "inputs": mx.array(
                    [data[j]["node_ids"] for j in idx], dtype=mx.int32),
                "node_values": mx.array(
                    [data[j]["node_values"] for j in idx], dtype=mx.float32),
                "adj_mask": mx.array(
                    [data[j]["adj_mask"] for j in idx], dtype=mx.float32),
                "node_digit_targets": mx.array(
                    [data[j]["node_digit_targets"] for j in idx], dtype=mx.int32),
                "final_digit_target": mx.array(
                    [data[j]["final_digit_target"] for j in idx], dtype=mx.int32),
                "raw_target": mx.array(
                    [data[j]["raw_target"] for j in idx], dtype=mx.int32),
                "num_real_nodes": mx.array(
                    [data[j]["num_real_nodes"] for j in idx], dtype=mx.int32),
            }


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=300)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--peak_lr",    type=float, default=1e-4)
    p.add_argument("--d_model",    type=int,   default=256)
    p.add_argument("--n_heads",    type=int,   default=8)
    p.add_argument("--H_cycles",   type=int,   default=2)
    p.add_argument("--L_cycles",   type=int,   default=4)
    p.add_argument("--H_layers",   type=int,   default=4)
    p.add_argument("--L_layers",   type=int,   default=4)
    p.add_argument("--data",       type=str,   default="data/gsm8k_train_clean.json")
    p.add_argument("--save_dir",   type=str,   default="checkpoints/gsm8k")
    p.add_argument("--no_save",    action="store_true")
    return p.parse_args()


def train():
    args = parse_args()

    # ── Model ────────────────────────────────────────────────────────────────
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
    mx.eval(model.parameters())
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"HRMForMath — {num_params:,} parameters")
    print(f"  d_model={args.d_model}, H_cycles={args.H_cycles}, L_cycles={args.L_cycles}")
    print(f"  H_layers={args.H_layers}, L_layers={args.L_layers}")
    print(f"  Output: {MAX_DIGITS} digits × {DIGIT_VOCAB_SIZE} vocab = "
          f"{MAX_DIGITS * DIGIT_VOCAB_SIZE} classes per node")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = GSM8KGraphDataset(args.data, max_nodes=50)
    if len(dataset) == 0:
        print("No data found. Run: python src/cache_gsm8k.py")
        return
    print(f"Dataset: {len(dataset)} samples")

    curriculum = CurriculumDataset(dataset)

    # ── Optimizer ────────────────────────────────────────────────────────────
    opt = TrainingOptimizer(
        peak_lr=args.peak_lr,
        warmup_steps=30,
        max_grad_norm=0.3,
        betas=[0.9, 0.95],
        schedule="cosine",  # Cosine decay helps generalization
        cycle_steps=args.epochs * (len(dataset) // args.batch_size + 1),
    )

    # ── Training loop ────────────────────────────────────────────────────────
    epochs = args.epochs
    batch_size = args.batch_size
    best_acc = 0.0
    last_phase = -1

    def make_loss_fn(epoch):
        """Loss: ONLY final answer node. Deep supervision removed to prevent mode collapse."""
        def _loss(model, batch):
            model.train()
            digit_logits, q_halt, q_continue = model(batch)
            
            # ONLY the final answer node — no deep supervision
            final_ce = final_node_digit_loss(
                digit_logits,
                batch["final_digit_target"],
                batch["num_real_nodes"]
            )
            
            return final_ce
        return _loss

    print(f"\nTraining for {epochs} epochs | ALL data from epoch 1 (no curriculum)")
    print(f"  Peak LR={args.peak_lr:.0e} | Grad clip=0.5 | Batch={batch_size}")
    print(f"  Loss: final-answer-only cross-entropy (no deep supervision)")
    print("─" * 80)

    for epoch in range(epochs):
        phase = 2  # Always use all data

        loss_and_grad = nn.value_and_grad(model, make_loss_fn(epoch))

        epoch_loss = 0.0
        epoch_gnorm = 0.0
        steps = 0

        for batch in curriculum.get_batches(batch_size):
            loss, grads = loss_and_grad(model, batch)
            gnorm = opt.step(model, grads)
            mx.eval(model.parameters(), loss)

            epoch_loss += loss.item()
            epoch_gnorm += gnorm
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        avg_gnorm = epoch_gnorm / max(steps, 1)
        lr = opt.current_lr

        do_eval = ((epoch + 1) % 10 == 0) or epoch == 0
        verbose = (epoch + 1) % 50 == 0

        if do_eval:
            exact_acc, digit_acc, near_acc = evaluate(
                model, dataset, verbose=verbose
            )
            improved = exact_acc > best_acc
            if improved:
                best_acc = exact_acc
                if not args.no_save:
                    save_checkpoint(model, args.save_dir, epoch + 1, exact_acc,
                                    {"loss": avg_loss, "digit_acc": digit_acc,
                                     "near_acc": near_acc})
            marker = " ★" if improved else ""
            print(
                f"Epoch {epoch+1:4d}/{epochs} [P{phase}] "
                f"Loss={avg_loss:.4f}  ‖g‖={avg_gnorm:.2f}  lr={lr:.1e} | "
                f"Exact={exact_acc*100:.2f}%  Digit={digit_acc*100:.1f}%  "
                f"Near={near_acc*100:.2f}%  Best={best_acc*100:.2f}%{marker}"
            )
        else:
            print(
                f"Epoch {epoch+1:4d}/{epochs} [P{phase}] "
                f"Loss={avg_loss:.4f}  ‖g‖={avg_gnorm:.2f}  lr={lr:.1e}"
            )

    print("─" * 80)
    print(f"Done. Best exact-match accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    train()
