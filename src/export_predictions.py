"""
Export HRM pipeline intermediate outputs for UI visualization.
Generates a JSON file with per-sample component outputs.

Usage: python src/export_predictions.py
"""
import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from src.dataset import (
    GSM8KGraphDataset, OP_VOCAB, DIGIT_VOCAB, DIGIT_VOCAB_SIZE,
    MAX_DIGITS, decode_digits_to_number, IDX_TO_DIGIT
)
from src.model import HRMForMath

# Reverse OP_VOCAB
IDX_TO_OP = {v: k for k, v in OP_VOCAB.items()}


def load_model(checkpoint_path):
    model = HRMForMath(
        vocab_size=len(OP_VOCAB), d_model=256, n_heads=8,
        H_cycles=2, L_cycles=4, H_layers=4, L_layers=4, seq_len=50,
    )
    weights = mx.load(checkpoint_path)
    flat_weights = [(k.replace("/", "."), mx.array(v)) for k, v in weights.items()]
    model.update(tree_unflatten(flat_weights))
    mx.eval(model.parameters())
    model.eval()
    return model


def export_sample(model, batch, sample_idx=0):
    """Run model and capture intermediate outputs for one sample."""
    inputs = batch["inputs"]
    node_values = batch["node_values"]
    adj_mask = batch["adj_mask"]
    B, N = inputs.shape

    # Step 1: Bridge
    graph_ctx = model.bridge(inputs, node_values, adj_mask)
    pos_emb = model.embed_pos.embedding_weight.astype(mx.float32)
    input_embeddings = model.embed_scale * 0.707106781 * (graph_ctx + pos_emb[None, :N, :])
    
    # Step 2: H/L Reasoning
    pad_mask = (inputs == 0)
    attention_mask = mx.where(pad_mask[:, None, None, :], mx.array(-1e9), mx.array(0.0))
    
    z_H = mx.broadcast_to(model.H_init, (B, N, model.d_model))
    z_L = mx.broadcast_to(model.L_init, (B, N, model.d_model))
    
    h_states = []
    l_states = []
    
    for h_step in range(model.H_cycles):
        for l_step in range(model.L_cycles):
            z_L_new = model.L_level(z_L, z_H + input_embeddings, attention_mask=attention_mask)
            z_L = mx.stop_gradient(z_L_new) if not (h_step == model.H_cycles-1 and l_step == model.L_cycles-1) else z_L_new
            l_norm = float(mx.mean(mx.abs(z_L[sample_idx])).item())
            l_states.append({"h_cycle": h_step, "l_cycle": l_step, "norm": round(l_norm, 4)})
        
        z_H_new = model.H_level(z_H, z_L, attention_mask=attention_mask)
        z_H = mx.stop_gradient(z_H_new) if h_step != model.H_cycles-1 else z_H_new
        h_norm = float(mx.mean(mx.abs(z_H[sample_idx])).item())
        h_states.append({"h_cycle": h_step, "norm": round(h_norm, 4)})

    # Step 3: Digit classification
    digit_logits, q_halt, q_continue = model(batch)
    mx.eval(digit_logits, q_halt, q_continue, graph_ctx)
    
    num_real = int(batch["num_real_nodes"][sample_idx].item())
    li = min(num_real - 1, N - 1)
    
    final_logits = digit_logits[sample_idx, li]  # (MAX_DIGITS, VOCAB)
    probs = mx.softmax(final_logits, axis=-1)
    pred_digits = mx.argmax(probs, axis=-1)
    mx.eval(probs, pred_digits)
    
    # Build per-digit probability distributions
    digit_probs = []
    for d in range(MAX_DIGITS):
        dist = {}
        for v in range(DIGIT_VOCAB_SIZE):
            label = IDX_TO_DIGIT.get(v, '?')
            dist[label] = round(float(probs[d, v].item()), 4)
        digit_probs.append({
            "position": d,
            "predicted": IDX_TO_DIGIT.get(int(pred_digits[d].item()), '?'),
            "distribution": dist
        })
    
    pred_list = [int(pred_digits[d].item()) for d in range(MAX_DIGITS)]
    pred_int = decode_digits_to_number(pred_list)
    true_int = int(batch["raw_target"][sample_idx].item())
    true_digits = [int(batch["final_digit_target"][sample_idx, d].item()) for d in range(MAX_DIGITS)]
    
    # Build graph nodes
    nodes = []
    for n in range(num_real):
        op_id = int(inputs[sample_idx, n].item())
        op_name = IDX_TO_OP.get(op_id, "PAD")
        v1 = round(float(node_values[sample_idx, n, 0].item()), 3)
        v2 = round(float(node_values[sample_idx, n, 1].item()), 3)
        
        # Get this node's prediction too
        node_logits = digit_logits[sample_idx, n]
        node_pred = mx.argmax(node_logits, axis=-1)
        mx.eval(node_pred)
        node_pred_list = [int(node_pred[d].item()) for d in range(MAX_DIGITS)]
        node_pred_int = decode_digits_to_number(node_pred_list)
        
        # Node target
        node_target_list = [int(batch["node_digit_targets"][sample_idx, n, d].item()) for d in range(MAX_DIGITS)]
        node_target_int = decode_digits_to_number(node_target_list)
        
        # Bridge embedding norm for this node
        emb_norm = round(float(mx.sqrt(mx.sum(graph_ctx[sample_idx, n] ** 2)).item()), 3)
        
        nodes.append({
            "id": n,
            "op": op_name,
            "arg1_norm": v1,
            "arg2_norm": v2,
            "embedding_norm": emb_norm,
            "predicted": node_pred_int,
            "target": node_target_int,
            "correct": node_pred_int == node_target_int
        })
    
    # Build edges from adjacency
    edges = []
    for i in range(num_real):
        for j in range(num_real):
            if float(adj_mask[sample_idx, i, j].item()) > 0.5:
                edges.append({"source": i, "target": j})
    
    return {
        "sample_id": sample_idx,
        "true_answer": true_int,
        "predicted_answer": pred_int,
        "correct": pred_int == true_int,
        "near_match": abs(pred_int - true_int) <= 1,
        "true_digits": [IDX_TO_DIGIT.get(d, '?') for d in true_digits],
        "pred_digits": [IDX_TO_DIGIT.get(d, '?') for d in pred_list],
        "num_nodes": num_real,
        "graph": {"nodes": nodes, "edges": edges},
        "h_states": h_states,
        "l_states": l_states,
        "digit_probs": digit_probs,
        "q_halt": round(float(q_halt[sample_idx].item()), 4),
        "q_continue": round(float(q_continue[sample_idx].item()), 4),
    }


def main():
    # Find best checkpoint
    rl_path = "checkpoints/gsm8k_rl/best_rl_model.npz"
    sv_path = "checkpoints/gsm8k/best_model.npz"
    ckpt = rl_path if os.path.exists(rl_path) else sv_path
    
    print(f"Loading model from {ckpt}")
    model = load_model(ckpt)
    
    dataset = GSM8KGraphDataset("data/gsm8k_train_parsed.json", max_nodes=50)
    print(f"Dataset: {len(dataset)} samples")
    
    results = []
    for batch in dataset.get_batches(1, shuffle=False):
        if len(results) >= len(dataset):
            break
        sample = export_sample(model, batch, 0)
        sample["sample_id"] = len(results)
        results.append(sample)
    
    correct = sum(1 for r in results if r["correct"])
    print(f"Exact match: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    
    out = {"samples": results, "total": len(results), "accuracy": correct / max(1, len(results)),
           "checkpoint": ckpt, "model_params": "5.8M", "architecture": "HRMForMath"}
    
    os.makedirs("ui", exist_ok=True)
    with open("ui/predictions.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Exported to ui/predictions.json")


if __name__ == "__main__":
    main()
