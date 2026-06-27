"""
GSM8K dataset with digit-level classification targets.

Each node's arithmetic result is encoded as a sequence of digit tokens:
  72 → [8, 3, 12, 0, 0, 0, 0, 0]   (token for '7', '2', EOS, PAD...)

This matches the HRM paper's approach: discrete classification (like Sudoku
cells) rather than continuous regression.
"""

import json
import os
import mlx.core as mx
import numpy as np

# ─── Operation Vocabulary ────────────────────────────────────────────────────
OP_VOCAB = {
    "PAD": 0,
    "add": 1,
    "sub": 2,
    "mul": 3,
    "div": 4,
    "eq": 5,
    "const": 6,
    "var": 7,
    "final_answer": 8
}

# ─── Digit Vocabulary ────────────────────────────────────────────────────────
# Matches the paper's discrete classification approach
DIGIT_VOCAB = {
    'PAD': 0,   # Padding (ignored in loss)
    '0': 1, '1': 2, '2': 3, '3': 4, '4': 5,
    '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
    'NEG': 11,  # Negative sign
    'EOS': 12,  # End of number
}
DIGIT_VOCAB_SIZE = 13
MAX_DIGITS = 8  # Covers answers up to 99,999,999

# Reverse mapping for decoding
IDX_TO_DIGIT = {v: k for k, v in DIGIT_VOCAB.items()}


def encode_number_to_digits(value: float) -> list:
    """
    Encode a number as a sequence of digit tokens.
    
    Examples:
        72   → [8, 3, 12, 0, 0, 0, 0, 0]   (7=token8, 2=token3, EOS=12, PAD=0)
        -5   → [11, 6, 12, 0, 0, 0, 0, 0]  (NEG=11, 5=token6, EOS=12)
        624  → [7, 3, 5, 12, 0, 0, 0, 0]    (6=token7, 2=token3, 4=token5, EOS=12)
        0    → [1, 12, 0, 0, 0, 0, 0, 0]    (0=token1, EOS=12)
    """
    n = int(round(value))
    digits = []
    
    if n < 0:
        digits.append(DIGIT_VOCAB['NEG'])
        n = abs(n)
    
    # Convert each digit character to its token
    for ch in str(n):
        digits.append(DIGIT_VOCAB[ch])
    
    # Add EOS
    digits.append(DIGIT_VOCAB['EOS'])
    
    # Pad to MAX_DIGITS
    while len(digits) < MAX_DIGITS:
        digits.append(DIGIT_VOCAB['PAD'])
    
    return digits[:MAX_DIGITS]


def decode_digits_to_number(digit_tokens: list) -> int:
    """
    Decode a digit token sequence back to an integer.
    
    Examples:
        [8, 3, 12, 0, ...] → 72
        [11, 6, 12, 0, ...] → -5
    """
    is_negative = False
    num_str = ""
    
    for tok in digit_tokens:
        tok = int(tok)
        label = IDX_TO_DIGIT.get(tok, 'PAD')
        if label == 'PAD':
            continue
        elif label == 'EOS':
            break
        elif label == 'NEG':
            is_negative = True
        else:
            num_str += label
    
    if not num_str:
        return 0
    
    result = int(num_str)
    return -result if is_negative else result


def parse_graph_from_json(json_trace, max_nodes=50):
    """
    Parses a JSON reasoning trace into:
      - node_ids: operation type per step
      - node_values: [arg1_val, arg2_val] per step (log1p normalized)
      - node_digit_targets: digit-encoded result at each step (MAX_DIGITS tokens)
      - adj_mask: adjacency matrix
      - num_real_nodes: how many non-PAD nodes exist
      - raw_node_results: unencoded float results (for debugging)
    """
    node_ids = []
    node_values = []
    node_digit_targets = []
    raw_node_results = []
    
    var_to_idx = {}
    var_to_value = {}
    
    steps = json_trace.get("steps", [])
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    
    for i, step in enumerate(steps):
        if i >= max_nodes - 1:  # Reserve last slot for final_answer
            break
            
        op = step.get("op", "PAD")
        op_id = OP_VOCAB.get(op, OP_VOCAB["PAD"])
        node_ids.append(op_id)
        
        arg1 = step.get("arg1", 0)
        arg2 = step.get("arg2", 0)
        
        def resolve_value(arg):
            if isinstance(arg, (int, float)):
                return float(arg)
            elif isinstance(arg, str) and arg in var_to_value:
                return var_to_value[arg]
            else:
                try:
                    return float(arg)
                except (ValueError, TypeError):
                    return 0.0
        
        val1 = resolve_value(arg1)
        val2 = resolve_value(arg2)
        
        # Compute the ground truth result of this step
        if op == "add":
            result_val = val1 + val2
        elif op == "sub":
            result_val = val1 - val2
        elif op == "mul":
            result_val = val1 * val2
        elif op == "div" and val2 != 0:
            result_val = val1 / val2
        else:
            result_val = val1
        
        # Log1p normalize input values (for bridge)
        norm_val1 = np.sign(val1) * np.log1p(np.abs(val1))
        norm_val2 = np.sign(val2) * np.log1p(np.abs(val2))
        
        node_values.append([norm_val1, norm_val2])
        
        # Digit-encode the result (for classification target)
        node_digit_targets.append(encode_number_to_digits(result_val))
        raw_node_results.append(result_val)
        
        # Store result mapping
        res = step.get("result", "")
        if res:
            var_to_idx[res] = i
            var_to_value[res] = result_val
            
        # Add edges
        arg1_str = step.get("arg1", "")
        arg2_str = step.get("arg2", "")
        if isinstance(arg1_str, str) and arg1_str in var_to_idx:
            adj[var_to_idx[arg1_str], i] = 1.0
        if isinstance(arg2_str, str) and arg2_str in var_to_idx:
            adj[var_to_idx[arg2_str], i] = 1.0

    # Add final answer node
    num_real_nodes = len(node_ids)
    if num_real_nodes < max_nodes:
        final_idx = num_real_nodes
        node_ids.append(OP_VOCAB["final_answer"])
        node_values.append([0.0, 0.0])
        
        # The final answer node's target is the last computed result
        ans_var = json_trace.get("final_answer", "")
        if ans_var in var_to_value:
            final_val = var_to_value[ans_var]
            node_digit_targets.append(encode_number_to_digits(final_val))
            raw_node_results.append(final_val)
            adj[var_to_idx.get(ans_var, 0), final_idx] = 1.0
        else:
            node_digit_targets.append(encode_number_to_digits(0))
            raw_node_results.append(0.0)
        num_real_nodes += 1

    # Pad to max_nodes
    while len(node_ids) < max_nodes:
        node_ids.append(OP_VOCAB["PAD"])
        node_values.append([0.0, 0.0])
        node_digit_targets.append([DIGIT_VOCAB['PAD']] * MAX_DIGITS)
        raw_node_results.append(0.0)
        
    return (node_ids, node_values, node_digit_targets, adj.tolist(),
            num_real_nodes, raw_node_results)


class GSM8KGraphDataset:
    def __init__(self, json_file, max_nodes=50):
        self.max_nodes = max_nodes
        self.data = []
        
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                raw_data = json.load(f)
            
            for i, item in enumerate(raw_data):
                trace = item.get("trace", {})
                target = item.get("target", 0.0)
                
                (node_ids, node_values, node_digit_targets, adj_mask,
                 num_real, raw_results) = parse_graph_from_json(trace, max_nodes)
                
                step_count = len(trace.get("steps", []))
                
                self.data.append({
                    "node_ids": node_ids,
                    "node_values": node_values,
                    "adj_mask": adj_mask,
                    "node_digit_targets": node_digit_targets,  # (N, MAX_DIGITS)
                    "final_digit_target": encode_number_to_digits(target),  # (MAX_DIGITS,)
                    "raw_target": int(round(target)),  # For RL reward
                    "num_real_nodes": num_real,
                    "step_count": step_count,
                })
            
            print(f"Loaded {len(self.data)} samples with digit targets "
                  f"(max_digits={MAX_DIGITS}, vocab={DIGIT_VOCAB_SIZE})")
        else:
            print(f"Warning: Dataset file {json_file} not found.")

    def __len__(self):
        return len(self.data)

    def get_batches(self, batch_size, shuffle=True):
        if shuffle:
            indices = np.random.permutation(len(self.data))
        else:
            indices = np.arange(len(self.data))
        
        for i in range(0, len(self.data), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            yield {
                "inputs": mx.array(
                    [self.data[idx]["node_ids"] for idx in batch_indices],
                    dtype=mx.int32),
                "node_values": mx.array(
                    [self.data[idx]["node_values"] for idx in batch_indices],
                    dtype=mx.float32),
                "adj_mask": mx.array(
                    [self.data[idx]["adj_mask"] for idx in batch_indices],
                    dtype=mx.float32),
                "node_digit_targets": mx.array(
                    [self.data[idx]["node_digit_targets"] for idx in batch_indices],
                    dtype=mx.int32),
                "final_digit_target": mx.array(
                    [self.data[idx]["final_digit_target"] for idx in batch_indices],
                    dtype=mx.int32),
                "raw_target": mx.array(
                    [self.data[idx]["raw_target"] for idx in batch_indices],
                    dtype=mx.int32),
                "num_real_nodes": mx.array(
                    [self.data[idx]["num_real_nodes"] for idx in batch_indices],
                    dtype=mx.int32),
            }
