"""
Inference script for HRMForMath — run predictions on GSM8K samples.

Usage:
    python src/predict.py                                          # Use RL checkpoint
    python src/predict.py --checkpoint checkpoints/gsm8k/best_model.npz  # Use supervised checkpoint
    python src/predict.py --num_samples 20                         # Show more samples
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from src.dataset import (
    GSM8KGraphDataset, OP_VOCAB, DIGIT_VOCAB, DIGIT_VOCAB_SIZE,
    MAX_DIGITS, decode_digits_to_number, IDX_TO_DIGIT
)
from src.model import HRMForMath


def load_model(checkpoint_path, d_model=256, n_heads=8, H_cycles=2, L_cycles=4, H_layers=4, L_layers=4):
    model = HRMForMath(
        vocab_size=len(OP_VOCAB),
        d_model=d_model,
        n_heads=n_heads,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        H_layers=H_layers,
        L_layers=L_layers,
        seq_len=50,
    )
    
    weights = mx.load(checkpoint_path)
    flat_weights = [(k.replace("/", "."), mx.array(v)) for k, v in weights.items()]
    nested_weights = tree_unflatten(flat_weights)
    model.update(nested_weights)
    mx.eval(model.parameters())
    model.eval()
    return model


def format_digit_tokens(digit_list):
    """Convert digit token list to human-readable string."""
    result = ""
    for tok in digit_list:
        label = IDX_TO_DIGIT.get(tok, '?')
        if label == 'EOS':
            break
        elif label == 'PAD':
            continue
        elif label == 'NEG':
            result += '-'
        else:
            result += label
    return result if result else "0"


def predict(args):
    # Find the best checkpoint
    checkpoint = args.checkpoint
    if not checkpoint:
        # Try RL checkpoint first, fall back to supervised
        rl_path = "checkpoints/gsm8k_rl/best_rl_model.npz"
        sv_path = "checkpoints/gsm8k/best_model.npz"
        if os.path.exists(rl_path):
            checkpoint = rl_path
            print(f"Using RL checkpoint: {checkpoint}")
        elif os.path.exists(sv_path):
            checkpoint = sv_path
            print(f"Using supervised checkpoint: {checkpoint}")
        else:
            print("No checkpoint found! Train first with: python src/train.py")
            return
    
    model = load_model(checkpoint)
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model: {num_params:,} parameters")
    
    dataset = GSM8KGraphDataset(args.data, max_nodes=50)
    if len(dataset) == 0:
        print("No data found.")
        return
    
    print(f"\n{'='*80}")
    print(f"  HRM-MLX GSM8K Predictions")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Samples: {min(args.num_samples, len(dataset))}")
    print(f"{'='*80}\n")
    
    correct = 0
    near = 0
    total = 0
    
    for batch in dataset.get_batches(1, shuffle=False):
        if total >= args.num_samples:
            break
        
        digit_logits, q_halt, q_continue = model(batch)
        B, N, D, V = digit_logits.shape
        num_real = batch["num_real_nodes"]
        
        li = int(mx.clip(num_real[0] - 1, 0, N - 1).item())
        logits = digit_logits[0, li]  # (MAX_DIGITS, VOCAB)
        pred_digits = mx.argmax(logits, axis=-1)
        mx.eval(pred_digits)
        
        pred_list = [int(pred_digits[d].item()) for d in range(D)]
        pred_int = decode_digits_to_number(pred_list)
        true_int = int(batch["raw_target"][0].item())
        
        # Also get confidence
        probs = mx.softmax(logits, axis=-1)
        max_probs = mx.max(probs, axis=-1)
        mx.eval(max_probs)
        avg_confidence = sum(float(max_probs[d].item()) for d in range(D)) / D
        
        is_correct = pred_int == true_int
        is_near = abs(pred_int - true_int) <= 1
        
        if is_correct:
            correct += 1
            status = "✅ CORRECT"
        elif is_near:
            near += 1
            status = "🔶 NEAR (±1)"
        else:
            status = "❌ WRONG"
        
        total += 1
        
        # Get the true digit tokens for comparison
        true_digit_list = [int(batch["final_digit_target"][0, d].item()) for d in range(D)]
        
        print(f"  Sample {total:3d}  {status}")
        print(f"    True answer:  {true_int:>10}   (digits: {format_digit_tokens(true_digit_list)})")
        print(f"    Predicted:    {pred_int:>10}   (digits: {format_digit_tokens(pred_list)})")
        print(f"    Confidence:   {avg_confidence:.1%}")
        print(f"    Nodes used:   {int(num_real[0].item())}")
        print()
    
    print(f"{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  Total samples:     {total}")
    print(f"  Exact match:       {correct}/{total} ({correct/max(1,total)*100:.1f}%)")
    print(f"  Near match (±1):   {correct+near}/{total} ({(correct+near)/max(1,total)*100:.1f}%)")
    print(f"  Wrong:             {total-correct-near}/{total}")
    print(f"{'='*80}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--data", type=str, default="data/gsm8k_train_parsed.json")
    p.add_argument("--num_samples", type=int, default=49)
    return p.parse_args()


if __name__ == "__main__":
    predict(parse_args())
