"""
HRM for GSM8K — Full PyTorch Port for Kaggle/Colab GPU Training.

This is a self-contained script that ports the entire HRM-MLX pipeline to PyTorch.
Upload this + your data/gsm8k_train_clean.json to Kaggle and run with GPU enabled.

Usage (Kaggle notebook cell):
    !python hrm_gsm8k_pytorch.py --epochs 500 --d_model 512 --batch_size 128

Architecture (paper-faithful):
    1. GraphAwareBridge (GAT): op tokens + numerical args → embeddings
    2. H-level (planner): slow, abstract reasoning
    3. L-level (executor): fast, detailed computation
    4. Digit classification head: per-digit cross-entropy
    5. 1-step gradient approximation: only final H/L iteration gets gradients
"""

import os
import json
import math
import argparse
import numpy as np
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Vocabulary & Encoding (identical to MLX version)
# ═══════════════════════════════════════════════════════════════════════════════

OP_VOCAB = {
    "PAD": 0, "add": 1, "sub": 2, "mul": 3, "div": 4,
    "eq": 5, "const": 6, "var": 7, "final_answer": 8
}

DIGIT_VOCAB = {
    'PAD': 0,
    '0': 1, '1': 2, '2': 3, '3': 4, '4': 5,
    '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
    'NEG': 11, 'EOS': 12,
}
DIGIT_VOCAB_SIZE = 13
MAX_DIGITS = 8
IDX_TO_DIGIT = {v: k for k, v in DIGIT_VOCAB.items()}


def encode_number_to_digits(value: float) -> list:
    n = int(round(value))
    digits = []
    if n < 0:
        digits.append(DIGIT_VOCAB['NEG'])
        n = abs(n)
    for ch in str(n):
        digits.append(DIGIT_VOCAB[ch])
    digits.append(DIGIT_VOCAB['EOS'])
    while len(digits) < MAX_DIGITS:
        digits.append(DIGIT_VOCAB['PAD'])
    return digits[:MAX_DIGITS]


def decode_digits_to_number(digit_tokens: list) -> int:
    is_negative = False
    num_str = ""
    for tok in digit_tokens:
        label = IDX_TO_DIGIT.get(int(tok), 'PAD')
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def parse_graph_from_json(json_trace, max_nodes=50):
    node_ids, node_values, node_digit_targets, raw_node_results = [], [], [], []
    var_to_idx, var_to_value = {}, {}
    steps = json_trace.get("steps", [])
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)

    for i, step in enumerate(steps):
        if i >= max_nodes - 1:
            break
        op = step.get("op", "PAD")
        op_id = OP_VOCAB.get(op, OP_VOCAB["PAD"])
        node_ids.append(op_id)

        arg1, arg2 = step.get("arg1", 0), step.get("arg2", 0)

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

        val1, val2 = resolve_value(arg1), resolve_value(arg2)

        if op == "add":     result_val = val1 + val2
        elif op == "sub":   result_val = val1 - val2
        elif op == "mul":   result_val = val1 * val2
        elif op == "div" and val2 != 0: result_val = val1 / val2
        else:               result_val = val1

        norm_val1 = np.sign(val1) * np.log1p(np.abs(val1))
        norm_val2 = np.sign(val2) * np.log1p(np.abs(val2))
        node_values.append([norm_val1, norm_val2])
        node_digit_targets.append(encode_number_to_digits(result_val))
        raw_node_results.append(result_val)

        res = step.get("result", "")
        if res:
            var_to_idx[res] = i
            var_to_value[res] = result_val

        arg1_str, arg2_str = step.get("arg1", ""), step.get("arg2", "")
        if isinstance(arg1_str, str) and arg1_str in var_to_idx:
            adj[var_to_idx[arg1_str], i] = 1.0
        if isinstance(arg2_str, str) and arg2_str in var_to_idx:
            adj[var_to_idx[arg2_str], i] = 1.0

    num_real_nodes = len(node_ids)
    if num_real_nodes < max_nodes:
        final_idx = num_real_nodes
        node_ids.append(OP_VOCAB["final_answer"])
        node_values.append([0.0, 0.0])
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

    while len(node_ids) < max_nodes:
        node_ids.append(OP_VOCAB["PAD"])
        node_values.append([0.0, 0.0])
        node_digit_targets.append([DIGIT_VOCAB['PAD']] * MAX_DIGITS)
        raw_node_results.append(0.0)

    return node_ids, node_values, node_digit_targets, adj, num_real_nodes, raw_node_results


class GSM8KDataset(Dataset):
    def __init__(self, json_file, max_nodes=50):
        self.data = []
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        for item in raw_data:
            trace = item.get("trace", {})
            target = item.get("target", 0.0)
            node_ids, node_values, node_digit_targets, adj_mask, num_real, _ = \
                parse_graph_from_json(trace, max_nodes)
            step_count = len(trace.get("steps", []))
            self.data.append({
                "node_ids": torch.tensor(node_ids, dtype=torch.long),
                "node_values": torch.tensor(node_values, dtype=torch.float32),
                "adj_mask": torch.tensor(adj_mask, dtype=torch.float32),
                "node_digit_targets": torch.tensor(node_digit_targets, dtype=torch.long),
                "final_digit_target": torch.tensor(encode_number_to_digits(target), dtype=torch.long),
                "raw_target": torch.tensor(int(round(target)), dtype=torch.long),
                "num_real_nodes": torch.tensor(num_real, dtype=torch.long),
                "step_count": step_count,
            })
        print(f"Loaded {len(self.data)} samples (max_digits={MAX_DIGITS}, vocab={DIGIT_VOCAB_SIZE})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) if k != "step_count"
            else [b[k] for b in batch] for k in batch[0].keys()}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Model Architecture (PyTorch port of MLX HRM)
# ═══════════════════════════════════════════════════════════════════════════════

class DenseGATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, concat=True, dropout=0.1):
        super().__init__()
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a_src = nn.Linear(out_features, 1, bias=False)
        self.a_dst = nn.Linear(out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask):
        B, N, _ = x.shape
        x_proj = self.W(x).reshape(B, N, self.heads, self.out_features)
        src_scores = self.a_src(x_proj).squeeze(-1)  # (B, N, heads)
        dst_scores = self.a_dst(x_proj).squeeze(-1)
        e = F.leaky_relu(src_scores.unsqueeze(2) + dst_scores.unsqueeze(1), 0.2)  # (B, N, N, heads)
        mask = adj_mask.unsqueeze(-1)
        e = e.masked_fill(mask == 0, -1e9)
        attention = F.softmax(e, dim=2)
        attention = self.dropout(attention)
        h_prime = torch.einsum('bnjh,bjhd->bnhd', attention, x_proj)
        if self.concat:
            return h_prime.reshape(B, N, self.heads * self.out_features)
        return h_prime.mean(dim=2)


class GraphAwareBridge(nn.Module):
    def __init__(self, vocab_size, d_model, num_value_features=2, gat_hidden=128, gat_layers=3, heads=4):
        super().__init__()
        self.node_embedding = nn.Embedding(vocab_size, d_model - num_value_features)
        self.value_proj = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList()
        self.layers.append(DenseGATLayer(d_model, gat_hidden, heads=heads, concat=True))
        for _ in range(gat_layers - 2):
            self.layers.append(DenseGATLayer(gat_hidden * heads, gat_hidden, heads=heads, concat=True))
        self.layers.append(DenseGATLayer(gat_hidden * heads, d_model, heads=heads, concat=False))

    def forward(self, node_ids, node_values, adj_mask):
        op_embed = self.node_embedding(node_ids)
        x = torch.cat([op_embed, node_values], dim=-1)
        x = self.value_proj(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_mask)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x


def rms_norm(x, eps=1e-5):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, expansion=2.0):
        super().__init__()
        inter = int(round(expansion * d_model * 2 / 3))
        inter = ((inter + 255) // 256) * 256  # align to 256
        self.gate_up = nn.Linear(d_model, inter * 2, bias=False)
        self.down = nn.Linear(inter, d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class HRMBlock(nn.Module):
    def __init__(self, d_model, n_heads, expansion=2.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = SwiGLU(d_model, expansion)

    def forward(self, x, attention_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = rms_norm(x + self.o_proj(out))
        x = rms_norm(x + self.mlp(x))
        return x


class HRMReasoningModule(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, expansion=2.0):
        super().__init__()
        self.layers = nn.ModuleList([HRMBlock(d_model, n_heads, expansion) for _ in range(n_layers)])

    def forward(self, hidden_states, input_injection, attention_mask=None):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class HRMForMath(nn.Module):
    def __init__(self, vocab_size=9, d_model=512, n_heads=8, H_cycles=4, L_cycles=8,
                 H_layers=8, L_layers=8, expansion=2.0, seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.embed_scale = math.sqrt(d_model)

        self.bridge = GraphAwareBridge(vocab_size, d_model, num_value_features=2,
                                       gat_hidden=128, gat_layers=3, heads=4)
        self.embed_pos = nn.Embedding(seq_len, d_model)
        self.H_level = HRMReasoningModule(H_layers, d_model, n_heads, expansion)
        self.L_level = HRMReasoningModule(L_layers, d_model, n_heads, expansion)
        self.H_init = nn.Parameter(torch.randn(d_model) * 0.02)
        self.L_init = nn.Parameter(torch.randn(d_model) * 0.02)
        self.digit_head = nn.Linear(d_model, MAX_DIGITS * DIGIT_VOCAB_SIZE)
        self.q_head = nn.Linear(d_model, 2)
        nn.init.zeros_(self.q_head.weight)
        self.q_head.bias.data.fill_(-5.0)

    def forward(self, batch):
        inputs = batch["node_ids"]
        node_values = batch["node_values"]
        adj_mask = batch["adj_mask"]
        B, N = inputs.shape

        graph_ctx = self.bridge(inputs, node_values, adj_mask)
        pos_ids = torch.arange(N, device=inputs.device).unsqueeze(0)
        pos_emb = self.embed_pos(pos_ids)
        input_embeddings = self.embed_scale * 0.707106781 * (graph_ctx + pos_emb)

        pad_mask = (inputs == 0)
        attention_mask = pad_mask[:, None, None, :].float() * -1e9

        z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(B, N, -1)

        # 1-step gradient approximation
        with torch.no_grad():
            for h_step in range(self.H_cycles):
                for l_step in range(self.L_cycles):
                    if not (h_step == self.H_cycles - 1 and l_step == self.L_cycles - 1):
                        z_L = self.L_level(z_L, z_H + input_embeddings, attention_mask)
                if h_step != self.H_cycles - 1:
                    z_H = self.H_level(z_H, z_L, attention_mask)

        # Final iteration WITH gradient
        z_L = self.L_level(z_L, z_H + input_embeddings, attention_mask)
        z_H = self.H_level(z_H, z_L, attention_mask)

        digit_logits = self.digit_head(z_H).reshape(B, N, MAX_DIGITS, DIGIT_VOCAB_SIZE)
        q_logits = self.q_head(z_H[:, 0])
        return digit_logits, q_logits[:, 0], q_logits[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

def final_node_digit_loss(digit_logits, final_digit_target, num_real_nodes):
    B, N, D, V = digit_logits.shape
    last_idx = (num_real_nodes - 1).clamp(0, N - 1)
    # Gather final node logits
    idx = last_idx.view(B, 1, 1, 1).expand(B, 1, D, V)
    final_logits = digit_logits.gather(1, idx).squeeze(1)  # (B, D, V)
    digit_mask = (final_digit_target != DIGIT_VOCAB['PAD']).float()
    log_probs = F.log_softmax(final_logits, dim=-1)
    target_log_probs = log_probs.gather(-1, final_digit_target.unsqueeze(-1)).squeeze(-1)
    masked_loss = -target_log_probs * digit_mask
    return masked_loss.sum() / digit_mask.sum().clamp(min=1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    exact_correct, near_correct, total = 0, 0, 0
    total_digits_correct, total_digits = 0, 0

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        digit_logits, _, _ = model(batch)
        B, N, D, V = digit_logits.shape
        last_idx = (batch["num_real_nodes"] - 1).clamp(0, N - 1)

        for b in range(B):
            li = last_idx[b].item()
            pred_digits = digit_logits[b, li].argmax(dim=-1).cpu().tolist()
            pred_int = decode_digits_to_number(pred_digits)
            true_int = batch["raw_target"][b].item()

            if pred_int == true_int:
                exact_correct += 1
            elif abs(pred_int - true_int) <= 1:
                near_correct += 1

            true_digits = batch["final_digit_target"][b].cpu().tolist()
            for d in range(D):
                if true_digits[d] != DIGIT_VOCAB['PAD']:
                    total_digits += 1
                    if pred_digits[d] == true_digits[d]:
                        total_digits_correct += 1
            total += 1

    return (exact_correct / max(1, total),
            total_digits_correct / max(1, total_digits),
            (exact_correct + near_correct) / max(1, total))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = HRMForMath(
        vocab_size=len(OP_VOCAB), d_model=args.d_model, n_heads=args.n_heads,
        H_cycles=args.H_cycles, L_cycles=args.L_cycles,
        H_layers=args.H_layers, L_layers=args.L_layers, seq_len=50,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"HRMForMath — {num_params:,} parameters")
    print(f"  d_model={args.d_model}, H_cycles={args.H_cycles}, L_cycles={args.L_cycles}")

    # Dataset
    dataset = GSM8KDataset(args.data, max_nodes=50)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.peak_lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.05,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_acc = 0.0
    print(f"\nTraining for {args.epochs} epochs | batch_size={args.batch_size}")
    print("─" * 90)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_gnorm, steps = 0, 0, 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                digit_logits, _, _ = model(batch)
                loss = final_node_digit_loss(digit_logits, batch["final_digit_target"], batch["num_real_nodes"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_gnorm += gnorm.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        avg_gnorm = epoch_gnorm / max(steps, 1)
        lr = scheduler.get_last_lr()[0]

        do_eval = ((epoch + 1) % 10 == 0) or epoch == 0

        if do_eval:
            exact_acc, digit_acc, near_acc = evaluate(model, eval_loader, device)
            improved = exact_acc > best_acc
            if improved:
                best_acc = exact_acc
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
                with open(os.path.join(args.save_dir, "best_meta.json"), "w") as f:
                    json.dump({"epoch": epoch+1, "accuracy": exact_acc,
                               "loss": avg_loss, "digit_acc": digit_acc, "near_acc": near_acc}, f, indent=2)
            marker = " ★" if improved else ""
            print(f"Epoch {epoch+1:4d}/{args.epochs} Loss={avg_loss:.4f} ‖g‖={avg_gnorm:.2f} lr={lr:.1e} | "
                  f"Exact={exact_acc*100:.2f}% Digit={digit_acc*100:.1f}% Near={near_acc*100:.2f}% "
                  f"Best={best_acc*100:.2f}%{marker}")
        else:
            print(f"Epoch {epoch+1:4d}/{args.epochs} Loss={avg_loss:.4f} ‖g‖={avg_gnorm:.2f} lr={lr:.1e}")

    print("─" * 90)
    print(f"Done. Best exact-match accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=500)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--peak_lr",    type=float, default=3e-4)
    p.add_argument("--d_model",    type=int,   default=512)
    p.add_argument("--n_heads",    type=int,   default=8)
    p.add_argument("--H_cycles",   type=int,   default=4)
    p.add_argument("--L_cycles",   type=int,   default=8)
    p.add_argument("--H_layers",   type=int,   default=8)
    p.add_argument("--L_layers",   type=int,   default=8)
    p.add_argument("--data",       type=str,   default="data/gsm8k_train_clean.json")
    p.add_argument("--save_dir",   type=str,   default="checkpoints/gsm8k_gpu")
    args = p.parse_args()
    train(args)
