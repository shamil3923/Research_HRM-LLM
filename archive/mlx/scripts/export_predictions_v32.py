"""
Export per-sample predictions from the v3.2 PyTorch checkpoint for the UI.

Mirrors the architecture defined in the training notebook (cell 6 — HRMForMath)
exactly. Captures z_H / z_L norms at every H- and L-cycle so the UI's
"H/L Reasoning Cycles" panel can render real bars instead of a placeholder.

Usage:
    python export_predictions_v32.py [--max N] [--device mps|cpu]

Reads:
    output/best_model2.pt
    data/gsm8k_val_split.json

Writes:
    ui/predictions.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Vocabs (must match notebook cell 6) ─────────────────────────────────────
OP_VOCAB = {"PAD": 0, "add": 1, "sub": 2, "mul": 3, "div": 4,
            "eq": 5, "const": 6, "var": 7, "final_answer": 8}
DIGIT_VOCAB = {"PAD": 0, "0": 1, "1": 2, "2": 3, "3": 4, "4": 5,
               "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "NEG": 11, "EOS": 12}
DIGIT_VOCAB_SIZE = 13
MAX_DIGITS = 8
IDX2DIG = {v: k for k, v in DIGIT_VOCAB.items()}
IDX2OP = {v: k for k, v in OP_VOCAB.items()}
NODE_VAL_DIM = 4


def encode_number(val: float) -> List[int]:
    n = int(round(val))
    d: List[int] = []
    if n < 0:
        d.append(DIGIT_VOCAB["NEG"]); n = abs(n)
    for ch in str(n):
        d.append(DIGIT_VOCAB[ch])
    d.append(DIGIT_VOCAB["EOS"])
    while len(d) < MAX_DIGITS:
        d.append(DIGIT_VOCAB["PAD"])
    return d[:MAX_DIGITS]


def decode_digits(tokens) -> int:
    neg, s = False, ""
    seen = False
    for t in tokens:
        lbl = IDX2DIG.get(int(t), "PAD")
        if lbl == "PAD": continue
        if lbl == "EOS": break
        if lbl == "NEG": neg = True; seen = True
        else: s += lbl; seen = True
    if not seen or not s: return -1
    return -int(s) if neg else int(s)


def parse_graph(trace, max_nodes=50):
    nids, nvals, ndigs = [], [], []
    v2i, v2v = {}, {}
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    for i, step in enumerate(trace.get("steps", [])):
        if i >= max_nodes - 1: break
        op = step.get("op", "PAD")
        nids.append(OP_VOCAB.get(op, 0))

        def r(a):
            if isinstance(a, (int, float)): return float(a), False
            if isinstance(a, str) and a in v2v: return v2v[a], True
            try: return float(a), False
            except Exception: return 0.0, False

        v1, is_ref1 = r(step.get("arg1", 0))
        v2, is_ref2 = r(step.get("arg2", 0))
        if   op == "add":              rv = v1 + v2
        elif op == "sub":              rv = v1 - v2
        elif op == "mul":              rv = v1 * v2
        elif op == "div" and v2 != 0:  rv = v1 / v2
        else:                          rv = v1
        nvals.append([
            float(np.sign(v1) * np.log1p(abs(v1))), float(is_ref1),
            float(np.sign(v2) * np.log1p(abs(v2))), float(is_ref2),
        ])
        ndigs.append(encode_number(rv))
        rk = step.get("result", "")
        if rk:
            v2i[rk] = i
            v2v[rk] = rv
        for k in ["arg1", "arg2"]:
            a = step.get(k, "")
            if isinstance(a, str) and a in v2i:
                adj[v2i[a], i] = 1.0
        adj[i, i] = 1.0
    nr = len(nids)
    if nr < max_nodes:
        fa_var = trace.get("final_answer", "")
        fa_val = v2v.get(fa_var, 0.0)
        nids.append(OP_VOCAB["final_answer"])
        nvals.append([
            float(np.sign(fa_val) * np.log1p(abs(fa_val))), 1.0, 0.0, 0.0,
        ])
        ndigs.append(encode_number(fa_val))
        fi = nr
        if fa_var in v2i:
            adj[v2i[fa_var], fi] = 1.0
        adj[fi, fi] = 1.0
        nr += 1
    while len(nids) < max_nodes:
        nids.append(0)
        nvals.append([0.0] * NODE_VAL_DIM)
        ndigs.append([DIGIT_VOCAB["PAD"]] * MAX_DIGITS)
    return nids, nvals, ndigs, adj, nr


# ─── Model — exact mirror of notebook cell 6 ─────────────────────────────────
class DenseGATLayer(nn.Module):
    def __init__(self, in_f, out_f, heads=4, concat=True, drop=0.1):
        super().__init__()
        self.heads, self.out_f, self.concat = heads, out_f, concat
        self.W = nn.Linear(in_f, heads * out_f, bias=False)
        self.as_ = nn.Linear(out_f, 1, bias=False)
        self.ad = nn.Linear(out_f, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x, adj):
        B, N, _ = x.shape
        xp = self.W(x).reshape(B, N, self.heads, self.out_f)
        s = self.as_(xp).squeeze(-1)
        d = self.ad(xp).squeeze(-1)
        e = F.leaky_relu(s.unsqueeze(2) + d.unsqueeze(1), 0.2)
        e = e.masked_fill(adj.unsqueeze(-1) == 0, -1e4)
        attn = self.drop(F.softmax(e, dim=2))
        h = torch.einsum("bnjh,bjhd->bnhd", attn, xp)
        return h.reshape(B, N, self.heads * self.out_f) if self.concat else h.mean(2)


class GraphAwareBridge(nn.Module):
    def __init__(self, vsz, d, vf=NODE_VAL_DIM, gh=128, gl=3, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vsz, d - vf)
        self.vp = nn.Linear(d, d)
        self.gats = nn.ModuleList()
        self.is_last = []
        ind = d
        for i in range(gl):
            out = gh; co = True
            if i == gl - 1:
                out = d; co = False
            self.gats.append(DenseGATLayer(ind, out, heads=heads, concat=co, drop=0.1))
            self.is_last.append(i == gl - 1)
            ind = out * (heads if co else 1)

    def forward(self, nids, nvals, adj):
        x = torch.cat([self.emb(nids), nvals], dim=-1)
        x = self.vp(x)
        pad = (nids == 0) & (nvals.abs().sum(-1) == 0)
        for layer, is_last in zip(self.gats, self.is_last):
            pm = pad.unsqueeze(2) | pad.unsqueeze(1)
            a2 = adj.clone(); a2[pm] = 0.0
            x = layer(x, a2)
            if not is_last:
                x = F.elu(x)
        return x


def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d, ex=2.0):
        super().__init__()
        i = int(round(ex * d * 2 / 3)); i = (i + 255) // 256 * 256
        self.gu = nn.Linear(d, i * 2, bias=False)
        self.dn = nn.Linear(i, d, bias=False)

    def forward(self, x):
        g, u = self.gu(x).chunk(2, dim=-1)
        return self.dn(F.silu(g) * u)


class HRMBlock(nn.Module):
    def __init__(self, d, h, ex=2.0):
        super().__init__()
        self.h = h; self.hd = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.op = nn.Linear(d, d, bias=False)
        self.mlp = SwiGLU(d, ex)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.hd)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        if mask is not None: s = s + mask
        out = (F.softmax(s, dim=-1) @ v).transpose(1, 2).reshape(B, N, D)
        x = rms_norm(x + self.op(out))
        x = rms_norm(x + self.mlp(x))
        return x


class HRMModule(nn.Module):
    def __init__(self, nl, d, h, ex=2.0):
        super().__init__()
        self.layers = nn.ModuleList([HRMBlock(d, h, ex) for _ in range(nl)])

    def forward(self, hid, inj, mask=None):
        hid = hid + inj
        for l in self.layers: hid = l(hid, mask)
        return hid


class HRMForMath(nn.Module):
    def __init__(self, vsz=9, d=256, heads=8, Hc=3, Lc=4, Hl=4, Ll=4, ex=2.0, slen=50):
        super().__init__()
        self.Hc, self.Lc = Hc, Lc
        self.d = d
        self.bridge = GraphAwareBridge(vsz, d, vf=NODE_VAL_DIM, gh=128, gl=3, heads=4)
        self.pos = nn.Embedding(slen, d)
        self.Hmod = HRMModule(Hl, d, heads, ex)
        self.Lmod = HRMModule(Ll, d, heads, ex)
        self.Hi = nn.Parameter(torch.randn(d) * 0.02)
        self.Li = nn.Parameter(torch.randn(d) * 0.02)
        self.dhead = nn.Linear(d, MAX_DIGITS * DIGIT_VOCAB_SIZE)
        self.qnorm = nn.LayerNorm(d)
        self.qhead = nn.Linear(d, 2)

    def encode_inputs(self, ni, nv, am):
        B, N = ni.shape
        xt = self.bridge(ni, nv, am)
        xt = xt + self.pos(torch.arange(N, device=ni.device).unsqueeze(0))
        pad = (ni == 0) & (nv.abs().sum(-1) == 0)
        amask = pad.float().unsqueeze(1).unsqueeze(1) * -1e4
        return xt, amask, B, N

    def init_carry(self, B, N, device):
        zH = self.Hi.unsqueeze(0).unsqueeze(0).expand(B, N, -1).contiguous()
        zL = self.Li.unsqueeze(0).unsqueeze(0).expand(B, N, -1).contiguous()
        return zH, zL

    @torch.no_grad()
    def step_with_trace(self, ni, nv, am, zH, zL):
        """Same as step() but returns per-cycle (z_H, z_L) norms for visualisation."""
        xt, amask, B, N = self.encode_inputs(ni, nv, am)
        h_states, l_states = [], []
        for h in range(self.Hc):
            for l in range(self.Lc):
                zL = self.Lmod(zL, zH + xt, amask)
                l_states.append({
                    "h_cycle": h,
                    "l_cycle": l,
                    "norm": round(float(zL.abs().mean().item()), 4),
                })
            zH = self.Hmod(zH, zL, amask)
            h_states.append({
                "h_cycle": h,
                "norm": round(float(zH.abs().mean().item()), 4),
            })
        dl = self.dhead(zH).reshape(B, N, MAX_DIGITS, DIGIT_VOCAB_SIZE)
        ql = self.qhead(self.qnorm(zH[:, 0]))
        return dl, ql[:, 0], ql[:, 1], zH, zL, h_states, l_states


# ─── Main export ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=200,
                    help="number of val samples to export (default 200)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--ckpt",   default="output/best_model2.pt")
    ap.add_argument("--val",    default="data/gsm8k_val_split.json")
    ap.add_argument("--out",    default="ui/predictions.json")
    ap.add_argument("--act-segments", type=int, default=4)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = os.path.join(ROOT, args.ckpt)
    val_path = os.path.join(ROOT, args.val)
    out_path = os.path.join(ROOT, args.out)

    print(f"device  : {device}")
    print(f"ckpt    : {ckpt_path}")
    print(f"val data: {val_path}")

    model = HRMForMath(
        vsz=len(OP_VOCAB), d=256, heads=8,
        Hc=3, Lc=4, Hl=4, Ll=4, slen=50,
    ).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN missing keys: {missing[:5]}{' …' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  WARN unexpected keys: {unexpected[:5]}{' …' if len(unexpected) > 5 else ''}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {n_params/1e6:.2f}M params")

    val = json.load(open(val_path))
    n = min(args.max, len(val))
    print(f"  exporting {n} of {len(val)} val samples")

    out_samples = []
    correct_count = 0

    for idx in range(n):
        item = val[idx]
        trace = item.get("trace", {})
        target = float(item.get("target", 0.0))

        nids, nvals, ndigs, adj, nr = parse_graph(trace, max_nodes=50)
        ni = torch.tensor([nids], dtype=torch.long, device=device)
        nv = torch.tensor([nvals], dtype=torch.float32, device=device)
        am = torch.tensor([adj], dtype=torch.float32, device=device)

        zH, zL = model.init_carry(1, ni.shape[1], device)
        # Run the configured number of ACT segments; capture H/L from the LAST segment
        # (those are the values that produced the final answer).
        h_states = l_states = None
        for seg in range(args.act_segments):
            dl, qh, qc, zH, zL, h_states, l_states = model.step_with_trace(ni, nv, am, zH, zL)
            if (qh > qc).item() and seg + 1 >= 2:   # respect act_min_steps = 2
                break

        # Decode final answer from the final-answer node (last real node)
        li = max(0, nr - 1)
        final_logits = dl[0, li]
        pred_tok = final_logits.argmax(-1).cpu().tolist()
        pred_int = decode_digits(pred_tok)
        true_int = int(round(target))
        true_tok = encode_number(target)

        # Per-node decoded predictions (for graph visualization)
        node_pred_tok = dl[0].argmax(-1).cpu().tolist()
        nodes = []
        for n_i in range(nr):
            op_name = IDX2OP.get(int(nids[n_i]), "PAD")
            v1 = round(float(nvals[n_i][0]), 3)
            v2 = round(float(nvals[n_i][2]), 3)
            node_pred_int = decode_digits(node_pred_tok[n_i])
            node_target_int = decode_digits(ndigs[n_i])
            emb_norm = round(float(torch.norm(zH[0, n_i]).item()), 3)
            nodes.append({
                "id": n_i,
                "op": op_name,
                "arg1_norm": v1,
                "arg2_norm": v2,
                "embedding_norm": emb_norm,
                "predicted": node_pred_int,
                "target": node_target_int,
                "correct": node_pred_int == node_target_int,
            })

        # Edges from adjacency
        edges = []
        for i_ in range(nr):
            for j_ in range(nr):
                if adj[i_, j_] > 0.5 and i_ != j_:
                    edges.append({"source": i_, "target": j_})

        # Top-5 digit distributions for the FINAL node
        probs = F.softmax(final_logits, dim=-1).cpu().tolist()
        digit_probs = []
        for d_ in range(MAX_DIGITS):
            dist = {IDX2DIG.get(v, "?"): round(probs[d_][v], 4) for v in range(DIGIT_VOCAB_SIZE)}
            digit_probs.append({
                "position": d_,
                "predicted": IDX2DIG.get(int(pred_tok[d_]), "?"),
                "distribution": dist,
            })

        correct = pred_int == true_int
        if correct:
            correct_count += 1

        out_samples.append({
            "sample_id": idx,
            "question": item.get("question", ""),
            "true_answer": true_int,
            "predicted_answer": pred_int,
            "computed_answer": pred_int,
            "correct": correct,
            "near_match": (pred_int != -1) and abs(pred_int - true_int) <= 1,
            "true_digits": [IDX2DIG.get(int(t), "?") for t in true_tok],
            "pred_digits": [IDX2DIG.get(int(t), "?") for t in pred_tok],
            "num_nodes": nr,
            "graph": {"nodes": nodes, "edges": edges},
            "h_states": h_states,
            "l_states": l_states,
            "digit_probs": digit_probs,
            "q_halt":     round(float(qh.item()), 4),
            "q_continue": round(float(qc.item()), 4),
            "halt_step":  seg + 1,
        })
        if (idx + 1) % 20 == 0 or idx + 1 == n:
            print(f"  [{idx+1:>4}/{n}]  acc so far: {correct_count}/{idx+1} = {100*correct_count/(idx+1):.1f}%")

    out = {
        "samples": out_samples,
        "total": len(out_samples),
        "accuracy": correct_count / max(1, len(out_samples)),
        "checkpoint": ckpt_path,
        "model_params": f"{n_params/1e6:.1f}M",
        "architecture": "HRMForMath (Bridge GAT + H/L modules + ACT)",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWrote {out_path}")
    print(f"  samples : {out['total']}")
    print(f"  acc     : {out['accuracy']*100:.2f}%")
    print(f"  params  : {out['model_params']}")


if __name__ == "__main__":
    main()
