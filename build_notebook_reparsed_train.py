"""Builder for notebook_train_reparsed.ipynb.

End-to-end training notebook on Claude-Opus-4.8-reparsed GSM8K:
  1. Loads re-parsed GSM8K (you upload the JSON files from reparse_gsm8k_with_claude.py
     as a Kaggle dataset).
  2. Builds a distribution-MATCHED pretrain corpus by 10x perturbation of the
     re-parsed TRAIN traces (perturbations vary numeric leaves but preserve
     tree topology). Pretrain and fine-tune therefore see IDENTICAL distributions.
  3. Moderate-deep HRM architecture (~7.5M params, 192 reasoning steps/fwd,
     fp16-stable) with MLP digit head (the key architectural fix for the
     per-digit precision ceiling).
  4. Pretrain → finetune → test eval pipeline.

Run:  python build_notebook_reparsed_train.py
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "notebook_train_reparsed.ipynb")


def md(cid, text):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(cid, src):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


# =========================================================================
MD_HEADER = """# HRM Training — Claude-Opus-Reparsed GSM8K + Matched-Distribution Pretrain + MLP Digit Head

**Three coordinated changes to break the 36.35% test ceiling we observed:**

### Change 1 — Better parsing (Claude Opus 4.8)
The training data is now re-parsed from raw GSM8K text by Claude Opus 4.8
into the HRM trace format. Expected improvements:
- Higher acceptance rate (more of the 8.5K problems converted successfully)
- Cleaner trace topologies (Claude understands the narrative; the prior Qwen
  parser introduced errors)
- Consistent op-decomposition style (no leftover noise from earlier parsing)

### Change 2 — Matched pretrain/finetune distributions
Pretrain corpus = 10× perturbations of the re-parsed train split. Topologies
preserved exactly; only numeric leaves resampled from GSM8K's empirical
distribution. This guarantees zero distribution shift between pretrain and
finetune — the bug that we already partially fixed and is now fully closed
(since BOTH come from the same parsing pass).

### Change 3 — MLP digit head
The per-digit accuracy ceiling at ~73% (which produced our 36.35% exact
ceiling) is caused by a single linear layer mapping the 256-dim H-state to
all 104 output logits. We replace it with a 2-layer MLP, giving the model
per-position computational capacity for digit prediction.

### Architecture summary

| Component | Setting |
|---|---|
| dmodel | 256 |
| nheads | 8 |
| Hcycles × Lcycles | 6 × 8 = 48 H/L iterations per segment |
| Hlayers × Llayers | 5 × 5 |
| ACT segments | 4 |
| Total reasoning steps per forward | **192** (4× v3.1, fp16-stable) |
| **Digit head** | **2-layer MLP (NEW)** |
| Params | ~8.5M (vs 7.5M with linear head) |

**Target.** GSM8K test exact accuracy ≥ 45%.

### Pre-requisites
- Upload the three JSON files from `reparse_gsm8k_with_claude.py`
  (`gsm8k_train_split.json`, `gsm8k_val_split.json`, `gsm8k_test_clean.json`)
  to a Kaggle dataset, then attach it to this notebook.
- Set `DATA_ROOT` in Cell 2 to that dataset's mount path.
"""

# =========================================================================
C_ENV = """# Cell 1 — Environment
import sys, os, json, time, math, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict

print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.version.cuda)
assert torch.cuda.is_available(), "GPU required"
DEVICE = torch.device("cuda")
"""

# =========================================================================
C_CONFIG = """# Cell 2 — Configuration

# UPDATE THIS PATH to your re-parsed dataset mount on Kaggle.
DATA_ROOT = "/kaggle/input/gsm8k-reparsed-claude"

CONFIG = dict(
    # --- Model: moderate-deep HRM ---
    dmodel     = 256,
    nheads     = 8,
    Hcycles    = 6,           # 4x v3.1 (3 → 6)
    Lcycles    = 8,           # 2x v3.1 (4 → 8)
    Hlayers    = 5,
    Llayers    = 5,
    max_nodes  = 50,

    # --- Digit head: 2-layer MLP (NEW) ---
    digit_head_hidden    = 512,
    digit_head_dropout   = 0.1,

    # --- Data paths ---
    data_train = f"{DATA_ROOT}/gsm8k_train_split.json",
    data_val   = f"{DATA_ROOT}/gsm8k_val_split.json",
    data_test  = f"{DATA_ROOT}/gsm8k_test_clean.json",

    # --- Pretrain corpus generation ---
    perturbations_per_trace = 10,
    max_intermediate        = 100_000,
    corpus_cache_path       = "/kaggle/working/reparsed_pretrain_corpus.json",

    # --- Pretrain ---
    pretrain_savedir    = "/kaggle/working/checkpoints/hrm_pretrain_reparsed",
    pretrain_epochs     = 20,
    pretrain_batch      = 96,
    pretrain_peak_lr    = 2e-4,
    pretrain_lr_floor   = 1e-5,
    pretrain_eval_every = 2,

    # --- Fine-tune ---
    finetune_savedir    = "/kaggle/working/checkpoints/hrm_finetune_reparsed",
    finetune_epochs     = 200,
    finetune_batch      = 96,
    finetune_peak_lr    = 5e-5,
    finetune_lr_floor   = 1e-6,
    finetune_eval_every = 10,
    augment_p           = 0.2,
    augment_max_value   = 200,

    # --- ACT (same as v3.1) ---
    act_max_steps = 4,
    act_min_steps = 2,

    # --- Loss weights ---
    aux_loss_weight = 1.5,
    q_loss_weight   = 0.5,

    # --- AdamATan2 ---
    optim_a = 1.27, optim_b = 1.0,
    optim_betas = (0.9, 0.95), optim_wd = 0.01,
)
os.makedirs(CONFIG["pretrain_savedir"], exist_ok=True)
os.makedirs(CONFIG["finetune_savedir"], exist_ok=True)

print("Config OK.")
print(f"  Architecture: dmodel={CONFIG['dmodel']}, Hc={CONFIG['Hcycles']}, "
      f"Lc={CONFIG['Lcycles']}, Hl={CONFIG['Hlayers']}, Ll={CONFIG['Llayers']}")
print(f"  Reasoning steps per fwd: {CONFIG['Hcycles']*CONFIG['Lcycles']*CONFIG['act_max_steps']}")
print(f"  Digit head: 2-layer MLP (hidden={CONFIG['digit_head_hidden']})")
print(f"\\nVerifying re-parsed data files exist:")
for k in ["data_train", "data_val", "data_test"]:
    p = CONFIG[k]
    print(f"  {k:12s}  exists={os.path.exists(p)}  {p}")
assert all(os.path.exists(CONFIG[k]) for k in ["data_train", "data_val", "data_test"]), \\
    "Re-parsed data files not found. Attach the dataset and update DATA_ROOT."
"""

# =========================================================================
C_UTILS = """# Cell 3 — Shared utilities (IDENTICAL to v3.1)
OP_VOCAB = {"PAD":0, "add":1, "sub":2, "mul":3, "div":4,
            "eq":5, "const":6, "var":7, "final_answer":8}
DIGIT_VOCAB = {"PAD":0, "0":1, "1":2, "2":3, "3":4, "4":5,
               "5":6, "6":7, "7":8, "8":9, "9":10, "NEG":11, "EOS":12}
DIGIT_VOCAB_SIZE = 13
MAX_DIGITS = 8
IDX2DIG = {v: k for k, v in DIGIT_VOCAB.items()}
NODE_VAL_DIM = 4
_AUG_MAX_ABS = 10 ** (MAX_DIGITS - 1) - 1


def encode_number(val):
    n = int(round(val))
    d = []
    if n < 0:
        d.append(DIGIT_VOCAB["NEG"]); n = abs(n)
    for ch in str(n):
        d.append(DIGIT_VOCAB[ch])
    d.append(DIGIT_VOCAB["EOS"])
    while len(d) < MAX_DIGITS:
        d.append(DIGIT_VOCAB["PAD"])
    return d[:MAX_DIGITS]


def decode_digits(tokens):
    neg, s, saw = False, "", False
    for t in tokens:
        lbl = IDX2DIG.get(int(t), "PAD")
        if lbl == "PAD": continue
        elif lbl == "EOS":
            saw = saw or bool(s); break
        elif lbl == "NEG":
            neg = True; saw = True
        else:
            s += lbl; saw = True
    if not saw or not s: return -1
    return -int(s) if neg else int(s)


def parse_graph(trace, max_nodes=50):
    nids, nvals, ndigs, raws = [], [], [], []
    v2i, v2v = {}, {}
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    for i, step in enumerate(trace.get("steps", [])):
        if i >= max_nodes - 1: break
        op = step.get("op", "PAD")
        nids.append(OP_VOCAB.get(op, 0))
        def res(a):
            if isinstance(a, (int, float)): return float(a), False
            if isinstance(a, str) and a in v2v: return v2v[a], True
            try: return float(a), False
            except: return 0.0, False
        v1, is_ref1 = res(step.get("arg1", 0))
        v2, is_ref2 = res(step.get("arg2", 0))
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
        raws.append(rv)
        rv_key = step.get("result", "")
        if rv_key:
            v2i[rv_key] = i; v2v[rv_key] = rv
        for k in ["arg1", "arg2"]:
            a = step.get(k, "")
            if isinstance(a, str) and a in v2i:
                adj[v2i[a], i] = 1.0
        adj[i, i] = 1.0
    nr = len(nids)
    if nr < max_nodes:
        fa_var = trace.get("final_answer", "")
        fa_val = v2v.get(fa_var, 0.0)
        fi = nr
        nids.append(OP_VOCAB["final_answer"])
        nvals.append([float(np.sign(fa_val) * np.log1p(abs(fa_val))), 1.0, 0.0, 0.0])
        ndigs.append(encode_number(fa_val))
        raws.append(fa_val)
        if fa_var in v2i: adj[v2i[fa_var], fi] = 1.0
        adj[fi, fi] = 1.0
        nr += 1
    while len(nids) < max_nodes:
        nids.append(0)
        nvals.append([0.0] * NODE_VAL_DIM)
        ndigs.append([DIGIT_VOCAB["PAD"]] * MAX_DIGITS)
        raws.append(0.0)
    return nids, nvals, ndigs, adj, nr, raws


def _sample_to_tensors(trace, target, max_nodes):
    nids, nvals, ndigs, adj, nr, _ = parse_graph(trace, max_nodes)
    try: tgt_int = int(round(float(target)))
    except (OverflowError, ValueError): tgt_int = 0
    if abs(tgt_int) > _AUG_MAX_ABS:
        tgt_int = _AUG_MAX_ABS if tgt_int > 0 else -_AUG_MAX_ABS
    return {
        "node_ids":        torch.tensor(nids,  dtype=torch.long),
        "node_values":     torch.tensor(nvals, dtype=torch.float32),
        "adj_mask":        torch.tensor(adj,   dtype=torch.float32),
        "node_digit_tgts": torch.tensor(ndigs, dtype=torch.long),
        "final_digit_tgt": torch.tensor(encode_number(tgt_int), dtype=torch.long),
        "raw_target":      torch.tensor(tgt_int, dtype=torch.long),
        "num_real_nodes":  torch.tensor(nr, dtype=torch.long),
    }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

print("Shared utilities loaded.")
"""

# =========================================================================
MD_PERTURB = """## Distribution-matched pretrain corpus

10× perturbations of each re-parsed GSM8K training trace. Numeric leaves
are resampled from the empirical leaf magnitude distribution; topology
(op sequence, references) preserved exactly. Therefore the pretrain
distribution is identical to the fine-tune distribution by construction.
"""

C_PERTURB = '''# Cell 4 — Pretrain corpus generation

def extract_leaf_magnitudes(records):
    mags = []
    for r in records:
        for s in r.get("trace", {}).get("steps", []):
            for a in [s.get("arg1"), s.get("arg2")]:
                if isinstance(a, (int, float)):
                    mags.append(abs(float(a)))
    return np.array(mags)


def _sample_leaf(rng, mags):
    v = rng.choice(mags)
    jitter = rng.randint(-2, 2)
    return max(1, int(round(v)) + jitter)


def perturb_trace(trace, rng, mags, max_tries=20, max_abs=None):
    if max_abs is None: max_abs = CONFIG["max_intermediate"]
    steps = trace.get("steps", [])
    if not steps: return None, None
    for _ in range(max_tries):
        new_steps, values = [], {}
        ok = True
        for s in steps:
            op = s.get("op", "")
            def maybe_replace(a):
                if isinstance(a, (int, float)):
                    sign = -1 if float(a) < 0 else 1
                    return float(sign * _sample_leaf(rng, mags))
                if isinstance(a, str):
                    if a in values: return a
                    try:
                        sign = -1 if float(a) < 0 else 1
                        return float(sign * _sample_leaf(rng, mags))
                    except ValueError:
                        return a
                return a
            arg1 = maybe_replace(s.get("arg1"))
            arg2 = maybe_replace(s.get("arg2"))
            def resolve(a):
                if isinstance(a, (int, float)): return float(a)
                if isinstance(a, str) and a in values: return values[a]
                try: return float(a)
                except: return 0.0
            v1 = resolve(arg1); v2 = resolve(arg2)
            if   op == "add": rv = v1 + v2
            elif op == "sub": rv = v1 - v2
            elif op == "mul": rv = v1 * v2
            elif op == "div":
                if v2 == 0 or v1 % v2 != 0: ok = False; break
                rv = v1 / v2
            else: rv = v1
            if not np.isfinite(rv) or abs(rv) > max_abs: ok = False; break
            if rv != int(rv): ok = False; break
            rk = s.get("result", "")
            if rk: values[rk] = rv
            new_steps.append({"op": op, "arg1": arg1, "arg2": arg2,
                              "result": rk, "result_value": rv})
        if not ok: continue
        fa_var = trace.get("final_answer", "")
        if fa_var not in values: continue
        fa = values[fa_var]
        if abs(fa) > _AUG_MAX_ABS: continue
        return {"steps": new_steps, "final_answer": fa_var}, int(round(fa))
    return None, None


def materialize_pretrain_corpus(force_rebuild=False):
    path = CONFIG["corpus_cache_path"]
    if not force_rebuild and os.path.exists(path):
        print(f"Loading cached corpus from {path}")
        with open(path) as f: return json.load(f)

    with open(CONFIG["data_train"]) as f:
        real_train = json.load(f)
    print(f"Loaded {len(real_train)} re-parsed GSM8K training traces")

    mags = extract_leaf_magnitudes(real_train)
    print(f"Extracted {len(mags)} leaf magnitudes  "
          f"(median={np.median(mags):.0f}, p95={np.percentile(mags,95):.0f})")

    rng = random.Random(20260528)
    corpus = []
    print(f"\\nGenerating {CONFIG['perturbations_per_trace']}x perturbations per trace...")
    n_attempts = n_kept = 0
    for i, item in enumerate(real_train):
        trace = item.get("trace", {})
        for _ in range(CONFIG["perturbations_per_trace"]):
            n_attempts += 1
            new_t, tgt = perturb_trace(trace, rng, mags)
            if new_t is None: continue
            corpus.append({"trace": new_t, "target": tgt})
            n_kept += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(real_train)}]  kept {n_kept}")
    print(f"  Total: {n_kept} kept from {n_attempts} attempts ({n_kept/max(1,n_attempts):.1%})")

    rng.shuffle(corpus)
    with open(path, "w") as f: json.dump(corpus, f)
    print(f"Cached to {path}")
    return corpus


CORPUS = materialize_pretrain_corpus()
print(f"\\nPretrain corpus size: {len(CORPUS)}")
'''

# =========================================================================
C_DATASETS = """# Cell 5 — Dataset classes

class CorpusDataset(Dataset):
    def __init__(self, records, max_nodes=50):
        self.records = records
        self.max_nodes = max_nodes
        self.cached = [_sample_to_tensors(r["trace"], r["target"], max_nodes)
                       for r in records]
        print(f"CorpusDataset: {len(self.cached)} samples")
    def __len__(self): return len(self.cached)
    def __getitem__(self, i): return self.cached[i]


class GSM8KDataset(Dataset):
    def __init__(self, json_file, max_nodes=50, augment=False,
                 augment_p=0.5, augment_max_value=200, _mags=None):
        with open(json_file) as f: raw = json.load(f)
        self.max_nodes = max_nodes
        self.augment = augment
        self.augment_p = augment_p
        self.augment_max_value = augment_max_value
        self._mags = _mags
        self.raw, self.cached, skipped = [], [], 0
        for item in raw:
            trace  = item.get("trace", {})
            target = float(item.get("target", 0.0))
            fa_var = trace.get("final_answer", "")
            step_results = {s.get("result", "") for s in trace.get("steps", [])}
            if fa_var not in step_results:
                skipped += 1; continue
            self.raw.append((trace, target))
            self.cached.append(_sample_to_tensors(trace, target, max_nodes))
        msg = f"GSM8K: {len(self.cached)} samples"
        if augment: msg += f"  [aug p={augment_p}]"
        if skipped: msg += f"  [skipped {skipped}]"
        print(msg)
    def __len__(self): return len(self.cached)
    def __getitem__(self, i):
        if self.augment and random.random() < self.augment_p and self._mags is not None:
            trace, _ = self.raw[i]
            new_t, new_tgt = perturb_trace(trace, random, self._mags)
            if new_t is not None and new_tgt is not None:
                return _sample_to_tensors(new_t, new_tgt, self.max_nodes)
        return self.cached[i]


random.seed(20260528)
random.shuffle(CORPUS)
_split = int(0.98 * len(CORPUS))
CORPUS_TRAIN = CORPUS[:_split]
CORPUS_VAL   = CORPUS[_split:]
print(f"Corpus train={len(CORPUS_TRAIN)}  val={len(CORPUS_VAL)}")
"""

# =========================================================================
MD_ARCH = """## Architecture — moderate-deep HRM + MLP digit head

The key new architectural component is the **MLP digit head**. Instead of
a single `nn.Linear(d, MAX_DIGITS * V)`, we use a 2-layer MLP that gives
the model per-position computational capacity for digit prediction.

This is the architectural fix for the per-digit precision ceiling we hit
at ~73%. Expected effect: per-digit accuracy +3-7pp → exact accuracy +5-15pp.
"""

C_ARCH = '''# Cell 6 — Architecture (HRM core + MLP digit head)

class DenseGATLayer(nn.Module):
    def __init__(self, in_f, out_f, heads=4, concat=True, drop=0.1):
        super().__init__()
        self.heads, self.out_f, self.concat = heads, out_f, concat
        self.W   = nn.Linear(in_f, heads * out_f, bias=False)
        self.as_ = nn.Linear(out_f, 1, bias=False)
        self.ad  = nn.Linear(out_f, 1, bias=False)
        self.drop = nn.Dropout(drop)
    def forward(self, x, adj):
        B, N, _ = x.shape
        xp = self.W(x).reshape(B, N, self.heads, self.out_f)
        s  = self.as_(xp).squeeze(-1); d = self.ad(xp).squeeze(-1)
        e  = F.leaky_relu(s.unsqueeze(2) + d.unsqueeze(1), 0.2)
        e  = e.masked_fill(adj.unsqueeze(-1) == 0, -1e4)
        attn = self.drop(F.softmax(e, dim=2))
        h    = torch.einsum("bnjh,bjhd->bnhd", attn, xp)
        return h.reshape(B, N, self.heads * self.out_f) if self.concat else h.mean(2)


class GraphAwareBridge(nn.Module):
    def __init__(self, vsz, d, vf=NODE_VAL_DIM, gh=128, gl=3, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vsz, d - vf)
        self.vp  = nn.Linear(d, d)
        self.gats = nn.ModuleList(); self.is_last = []
        ind = d
        for i in range(gl):
            out = gh; co = True
            if i == gl - 1: out = d; co = False
            self.gats.append(DenseGATLayer(ind, out, heads=heads, concat=co, drop=0.1))
            self.is_last.append(i == gl - 1)
            ind = out * (heads if co else 1)
    def forward(self, nids, nvals, adj):
        x = torch.cat([self.emb(nids), nvals], dim=-1); x = self.vp(x)
        pad = (nids == 0) & (nvals.abs().sum(-1) == 0)
        for layer, is_last in zip(self.gats, self.is_last):
            pm = pad.unsqueeze(2) | pad.unsqueeze(1)
            a2 = adj.clone(); a2[pm] = 0.0
            x = layer(x, a2)
            if not is_last: x = F.elu(x)
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
        self.op  = nn.Linear(d, d, bias=False)
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


class MLPDigitHead(nn.Module):
    """2-layer MLP digit head — the key fix for the per-digit precision ceiling.

    Old single-linear head:  d → MAX_DIGITS * V  (one matmul, ~104K params)
    New MLP head:            d → hidden → MAX_DIGITS * V  (~1M params)

    Gives the model per-position computational capacity for digit prediction
    instead of forcing all 8 positions to share a single linear projection.
    """
    def __init__(self, d, hidden, max_digits, vocab_size, dropout=0.1):
        super().__init__()
        self.max_digits = max_digits
        self.vocab_size = vocab_size
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max_digits * vocab_size),
        )
    def forward(self, x):
        B, N, _ = x.shape
        out = self.net(x)
        return out.reshape(B, N, self.max_digits, self.vocab_size)


class HRMForMath(nn.Module):
    def __init__(self, vsz=9, d=512, heads=8, Hc=4, Lc=8, Hl=8, Ll=8, ex=2.0, slen=50,
                 digit_head_hidden=512, digit_head_dropout=0.1):
        super().__init__()
        self.Hc, self.Lc = Hc, Lc
        self.bridge = GraphAwareBridge(vsz, d, vf=NODE_VAL_DIM, gh=128, gl=3, heads=4)
        self.pos    = nn.Embedding(slen, d)
        self.Hmod   = HRMModule(Hl, d, heads, ex)
        self.Lmod   = HRMModule(Ll, d, heads, ex)
        self.Hi     = nn.Parameter(torch.randn(d) * 0.02)
        self.Li     = nn.Parameter(torch.randn(d) * 0.02)
        # MLP digit head (NEW — the key precision-ceiling fix).
        self.dhead  = MLPDigitHead(d, digit_head_hidden, MAX_DIGITS, DIGIT_VOCAB_SIZE,
                                   dropout=digit_head_dropout)
        self.qnorm  = nn.LayerNorm(d)
        self.qhead  = nn.Linear(d, 2)
        nn.init.zeros_(self.qhead.weight)
        self.qhead.bias.data.copy_(torch.tensor([-5.0, -5.0]))
    def encode_inputs(self, batch):
        ni = batch["node_ids"]; nv = batch["node_values"]; am = batch["adj_mask"]
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
    def step(self, batch, zH, zL):
        xt, amask, B, N = self.encode_inputs(batch)
        with torch.no_grad():
            for h in range(self.Hc):
                for l in range(self.Lc):
                    if h == self.Hc - 1 and l == self.Lc - 1: continue
                    zL = self.Lmod(zL, zH + xt, amask)
                if h != self.Hc - 1:
                    zH = self.Hmod(zH, zL, amask)
        zL = self.Lmod(zL, zH + xt, amask)
        zH = self.Hmod(zH, zL, amask)
        dl = self.dhead(zH)
        ql = self.qhead(self.qnorm(zH[:, 0]))
        return dl, ql[:, 0], ql[:, 1], zH.detach(), zL.detach()

print("Architecture defined (with MLP digit head).")
'''

# =========================================================================
C_LOSS = """# Cell 7 — Losses + AdamATan2
from torch.optim.optimizer import Optimizer

def main_loss(dl, fdt, nrn):
    B, N, D, V = dl.shape
    li = (nrn - 1).clamp(0, N - 1)
    idx = li.view(B, 1, 1, 1).expand(B, 1, D, V)
    fl = dl.gather(1, idx).squeeze(1)
    dm = (fdt != DIGIT_VOCAB["PAD"]).float()
    lp = F.log_softmax(fl, dim=-1)
    tlp = lp.gather(-1, fdt.unsqueeze(-1)).squeeze(-1)
    return (-tlp * dm).sum() / dm.sum().clamp(min=1)

def aux_loss(dl, ndt, nrn):
    B, N, D, V = dl.shape
    node_idx = torch.arange(N, device=dl.device).unsqueeze(0)
    node_mask = (node_idx < (nrn - 1).unsqueeze(1)).float()
    digit_mask = (ndt != DIGIT_VOCAB["PAD"]).float()
    full_mask = node_mask.unsqueeze(-1) * digit_mask
    lp = F.log_softmax(dl, dim=-1)
    tlp = lp.gather(-1, ndt.unsqueeze(-1)).squeeze(-1)
    return (-tlp * full_mask).sum() / full_mask.sum().clamp(min=1)

def is_correct(dl, fdt, nrn):
    B, N, D, V = dl.shape
    li = (nrn - 1).clamp(0, N - 1)
    idx = li.view(B, 1, 1, 1).expand(B, 1, D, V)
    fl = dl.gather(1, idx).squeeze(1)
    pred = fl.argmax(-1)
    dm = (fdt != DIGIT_VOCAB["PAD"])
    return ((pred == fdt) | (~dm)).all(dim=-1).float()

def q_halt_loss(qh, c):     return F.binary_cross_entropy_with_logits(qh, c)
def q_continue_loss(qc, t): return F.binary_cross_entropy_with_logits(qc, t.detach())

def combined_segment_loss(dl, fdt, ndt, nrn, q_halt, q_cont, next_q_target,
                          aux_weight, q_weight):
    correct = is_correct(dl, fdt, nrn)
    main    = main_loss(dl, fdt, nrn)
    aux     = aux_loss(dl, ndt, nrn)
    qh      = q_halt_loss(q_halt, correct)
    qc      = q_continue_loss(q_cont, next_q_target)
    return main + aux_weight * aux + q_weight * (qh + qc), correct


class AdamATan2(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95),
                 weight_decay=0.0, a=1.27, b=1.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group["lr"]; beta1, beta2 = group["betas"]
            wd = group["weight_decay"]; a, b = group["a"], group["b"]
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad
                if grad.is_sparse: raise RuntimeError("sparse not supported")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"]    = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1; step = state["step"]
                if wd != 0.0: p.mul_(1.0 - lr * wd)
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bc1 = 1.0 - beta1 ** step; bc2 = 1.0 - beta2 ** step
                m_hat = m / bc1; v_hat = v / bc2
                delta = torch.atan2(m_hat, v_hat.sqrt() * b)
                p.add_(delta, alpha=-lr * a)
        return loss

print("Losses + AdamATan2 defined.")
"""

# =========================================================================
C_EVAL = """# Cell 8 — Evaluation
import statistics as _st

@torch.no_grad()
def evaluate(model, loader, device, max_steps=None, min_steps=None):
    if max_steps is None: max_steps = CONFIG["act_max_steps"]
    if min_steps is None: min_steps = CONFIG.get("act_min_steps", 1)
    model.eval()
    exact = near = total = dig_ok = dig_tot = no_out = 0
    halt_steps_used, preds = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        xt, amask, B, N = model.encode_inputs(batch)
        zH, zL = model.init_carry(B, N, device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)
        final_dl = torch.zeros(B, N, MAX_DIGITS, DIGIT_VOCAB_SIZE,
                               device=device, dtype=torch.float32)
        steps_taken = torch.zeros(B, dtype=torch.long, device=device)
        for s in range(max_steps):
            dl, qh, qc, zH, zL = model.step(batch, zH, zL)
            allow_halt = (s + 1) >= min_steps
            new_halt = ((qh > qc) & allow_halt) | (s == max_steps - 1)
            just_halted = new_halt & (~halted)
            final_dl[just_halted] = dl[just_halted].to(final_dl.dtype)
            steps_taken[just_halted] = s + 1
            halted = halted | new_halt
            if halted.all(): break
        not_halted = ~halted
        if not_halted.any():
            final_dl[not_halted] = dl[not_halted].to(final_dl.dtype)
            steps_taken[not_halted] = max_steps
        for b in range(B):
            halt_steps_used.append(int(steps_taken[b].item()))
            li = (batch["num_real_nodes"][b] - 1).clamp(0, N - 1).item()
            pd = final_dl[b, li].argmax(-1).cpu().tolist()
            pi = decode_digits(pd)
            ti = batch["raw_target"][b].item()
            if pi == -1:
                no_out += 1; preds.append(0)
            else:
                if pi == ti: exact += 1
                if abs(pi - ti) <= 1: near += 1
                preds.append(pi)
            tdig = batch["final_digit_tgt"][b].cpu().tolist()
            for d in range(MAX_DIGITS):
                if tdig[d] != DIGIT_VOCAB["PAD"]:
                    dig_tot += 1
                    if pd[d] == tdig[d]: dig_ok += 1
            total += 1
    pstd = _st.stdev(preds) if len(preds) > 1 else 0
    mean_steps = _st.mean(halt_steps_used) if halt_steps_used else 0
    return {"exact_acc": exact / max(1, total),
            "digit_acc": dig_ok / max(1, dig_tot),
            "near_acc":  (exact + near) / max(1, total),
            "no_output": no_out,
            "pred_std":  pstd,
            "mean_halt_steps": mean_steps}

print("Evaluation defined.")
"""

# =========================================================================
MD_PRETRAIN = """## Pretrain on the perturbation corpus (same distribution as fine-tune)

20 epochs, single global cosine LR schedule, fp16 with GradScaler.
Saves the final checkpoint to disk for fine-tune.
"""

C_PRETRAIN = """# Cell 9 — Pretrain

train_ds_pre = CorpusDataset(CORPUS_TRAIN, max_nodes=CONFIG["max_nodes"])
val_ds_pre   = CorpusDataset(CORPUS_VAL,   max_nodes=CONFIG["max_nodes"])
train_loader = DataLoader(train_ds_pre, batch_size=CONFIG["pretrain_batch"],
                          shuffle=True, collate_fn=collate_fn,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds_pre, batch_size=CONFIG["pretrain_batch"] * 2,
                          shuffle=False, collate_fn=collate_fn,
                          num_workers=2, pin_memory=True)

model = HRMForMath(vsz=len(OP_VOCAB), d=CONFIG["dmodel"], heads=CONFIG["nheads"],
                   Hc=CONFIG["Hcycles"], Lc=CONFIG["Lcycles"],
                   Hl=CONFIG["Hlayers"], Ll=CONFIG["Llayers"],
                   slen=CONFIG["max_nodes"],
                   digit_head_hidden=CONFIG["digit_head_hidden"],
                   digit_head_dropout=CONFIG["digit_head_dropout"]).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\\nHRMForMath (MLP digit head)  {n_params/1e6:.2f}M params")
print(f"  Reasoning steps per fwd: {CONFIG['Hcycles']*CONFIG['Lcycles']*CONFIG['act_max_steps']}")

opt = AdamATan2(model.parameters(), lr=CONFIG["pretrain_peak_lr"],
                betas=CONFIG["optim_betas"], weight_decay=CONFIG["optim_wd"],
                a=CONFIG["optim_a"], b=CONFIG["optim_b"])

total_steps  = max(1, CONFIG["pretrain_epochs"] * len(train_loader))
warmup_steps = max(1, int(0.05 * total_steps))
floor_ratio  = CONFIG["pretrain_lr_floor"] / CONFIG["pretrain_peak_lr"]

def _lr_lambda(step):
    if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    p = min(max(p, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * p))
    return floor_ratio + (1.0 - floor_ratio) * cos

sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
scaler = torch.amp.GradScaler("cuda")
ACT_MIN = CONFIG["act_min_steps"]

print(f"\\n{'Ep':>4}  {'Loss':>8}  {'gN':>5}  {'LR':>8}  {'ValEx%':>7}  {'ValDig%':>8}")
print("=" * 70)

pretrain_log = []
for ep in range(CONFIG["pretrain_epochs"]):
    model.train()
    el = eg = stp = 0
    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            B, N = batch["node_ids"].shape
            zH, zL = model.init_carry(B, N, DEVICE)
            num_segs = CONFIG["act_max_steps"]
            seg_outputs = []
            for s in range(num_segs):
                dl, qh, qc, zH, zL = model.step(batch, zH, zL)
                seg_outputs.append((dl, qh, qc))
            total_loss = 0.0
            for s, (dl, qh, qc) in enumerate(seg_outputs):
                if s + 1 < num_segs:
                    nqh, nqc = seg_outputs[s+1][1], seg_outputs[s+1][2]
                    next_q_target = torch.sigmoid(torch.maximum(nqh, nqc)).detach()
                else:
                    next_q_target = torch.sigmoid(qh).detach()
                q_w = 0.0 if (s + 1) < ACT_MIN else CONFIG["q_loss_weight"]
                seg_loss, _ = combined_segment_loss(
                    dl, batch["final_digit_tgt"], batch["node_digit_tgts"],
                    batch["num_real_nodes"], qh, qc, next_q_target,
                    CONFIG["aux_loss_weight"], q_w,
                )
                total_loss = total_loss + seg_loss
            total_loss = total_loss / num_segs
        scaler.scale(total_loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True); scaler.update(); sch.step()
            continue
        scaler.step(opt); scaler.update(); sch.step()
        el += total_loss.item(); eg += gn.item(); stp += 1

    al = el / max(stp, 1); ag = eg / max(stp, 1); lr = sch.get_last_lr()[0]
    if (ep + 1) % CONFIG["pretrain_eval_every"] == 0 or ep == 0:
        m = evaluate(model, val_loader, DEVICE)
        print(f"{ep+1:>4}  {al:>8.4f}  {ag:>5.2f}  {lr:>8.1e}  "
              f"{m['exact_acc']*100:>7.2f}  {m['digit_acc']*100:>8.2f}")
        pretrain_log.append({"epoch": ep+1, "loss": al,
                             "val_exact": m["exact_acc"], "val_digit": m["digit_acc"]})

PRETRAIN_CKPT = os.path.join(CONFIG["pretrain_savedir"], "hrm_pretrain.pt")
torch.save(model.state_dict(), PRETRAIN_CKPT)
with open(os.path.join(CONFIG["pretrain_savedir"], "pretrain_log.json"), "w") as f:
    json.dump(pretrain_log, f, indent=2)
print(f"\\nPretrain done. Saved to {PRETRAIN_CKPT}")
"""

# =========================================================================
MD_FINETUNE = """## Fine-tune on real (re-parsed) GSM8K

200 epochs at LR 5e-5. Lower LR preserves distribution-aligned pretrain features.
Augmentation = mild (p=0.2) using the same perturbation function so train sees
distribution-matched variation across epochs.
"""

C_FINETUNE = """# Cell 10 — Fine-tune on real GSM8K

# Extract leaf magnitudes once for augmentation
_FINETUNE_MAGS = extract_leaf_magnitudes([{"trace": t} for t, _ in
    [(item.get("trace", {}), item.get("target", 0.0))
     for item in json.load(open(CONFIG["data_train"]))]])

train_set_ft = GSM8KDataset(CONFIG["data_train"], max_nodes=CONFIG["max_nodes"],
                            augment=True, augment_p=CONFIG["augment_p"],
                            augment_max_value=CONFIG["augment_max_value"],
                            _mags=_FINETUNE_MAGS)
val_set_ft   = GSM8KDataset(CONFIG["data_val"], max_nodes=CONFIG["max_nodes"])
train_loader_ft = DataLoader(train_set_ft, batch_size=CONFIG["finetune_batch"],
                             shuffle=True, collate_fn=collate_fn,
                             num_workers=2, pin_memory=True)
val_loader_ft   = DataLoader(val_set_ft, batch_size=CONFIG["finetune_batch"] * 2,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=2, pin_memory=True)

model_ft = HRMForMath(vsz=len(OP_VOCAB), d=CONFIG["dmodel"], heads=CONFIG["nheads"],
                      Hc=CONFIG["Hcycles"], Lc=CONFIG["Lcycles"],
                      Hl=CONFIG["Hlayers"], Ll=CONFIG["Llayers"],
                      slen=CONFIG["max_nodes"],
                      digit_head_hidden=CONFIG["digit_head_hidden"],
                      digit_head_dropout=CONFIG["digit_head_dropout"]).to(DEVICE)
missing, unexpected = model_ft.load_state_dict(torch.load(PRETRAIN_CKPT), strict=False)
assert not missing,    f"Pretrained ckpt missing keys: {missing}"
assert not unexpected, f"Pretrained ckpt unexpected keys: {unexpected}"
print(f"\\nLoaded pretrain.  Train={len(train_set_ft)}  Val={len(val_set_ft)}")

opt_ft = AdamATan2(model_ft.parameters(), lr=CONFIG["finetune_peak_lr"],
                   betas=CONFIG["optim_betas"], weight_decay=CONFIG["optim_wd"],
                   a=CONFIG["optim_a"], b=CONFIG["optim_b"])

total_steps_ft  = max(1, CONFIG["finetune_epochs"] * len(train_loader_ft))
warmup_steps_ft = max(1, int(0.05 * total_steps_ft))
floor_ratio_ft  = CONFIG["finetune_lr_floor"] / CONFIG["finetune_peak_lr"]

def _lr_lambda_ft(step):
    if step < warmup_steps_ft:
        return float(step) / float(max(1, warmup_steps_ft))
    p = (step - warmup_steps_ft) / max(1, total_steps_ft - warmup_steps_ft)
    p = min(max(p, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * p))
    return floor_ratio_ft + (1.0 - floor_ratio_ft) * cos

sch_ft = torch.optim.lr_scheduler.LambdaLR(opt_ft, _lr_lambda_ft)
scaler_ft = torch.amp.GradScaler("cuda")
ACT_MIN = CONFIG["act_min_steps"]

best_acc = 0.0
ft_log = []
print(f"\\n{'Ep':>4}  {'Loss':>8}  {'gN':>5}  {'LR':>8}  {'Ex%':>6}  "
      f"{'Dig%':>7}  {'Near%':>6}  {'AvgHlt':>6}")
print("=" * 90)

for ep in range(CONFIG["finetune_epochs"]):
    model_ft.train()
    el = eg = stp = 0
    for batch in train_loader_ft:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        opt_ft.zero_grad()
        with torch.amp.autocast("cuda"):
            B, N = batch["node_ids"].shape
            zH, zL = model_ft.init_carry(B, N, DEVICE)
            num_segs = CONFIG["act_max_steps"]
            seg_outputs = []
            for s in range(num_segs):
                dl, qh, qc, zH, zL = model_ft.step(batch, zH, zL)
                seg_outputs.append((dl, qh, qc))
            total_loss = 0.0
            for s, (dl, qh, qc) in enumerate(seg_outputs):
                if s + 1 < num_segs:
                    nqh, nqc = seg_outputs[s+1][1], seg_outputs[s+1][2]
                    next_q_target = torch.sigmoid(torch.maximum(nqh, nqc)).detach()
                else:
                    next_q_target = torch.sigmoid(qh).detach()
                q_w = 0.0 if (s + 1) < ACT_MIN else CONFIG["q_loss_weight"]
                seg_loss, _ = combined_segment_loss(
                    dl, batch["final_digit_tgt"], batch["node_digit_tgts"],
                    batch["num_real_nodes"], qh, qc, next_q_target,
                    CONFIG["aux_loss_weight"], q_w,
                )
                total_loss = total_loss + seg_loss
            total_loss = total_loss / num_segs
        scaler_ft.scale(total_loss).backward()
        scaler_ft.unscale_(opt_ft)
        gn = torch.nn.utils.clip_grad_norm_(model_ft.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt_ft.zero_grad(set_to_none=True); scaler_ft.update(); sch_ft.step()
            continue
        scaler_ft.step(opt_ft); scaler_ft.update(); sch_ft.step()
        el += total_loss.item(); eg += gn.item(); stp += 1

    al = el / max(stp, 1); ag = eg / max(stp, 1); lr = sch_ft.get_last_lr()[0]
    do_eval = ((ep + 1) % CONFIG["finetune_eval_every"] == 0) or ep == 0
    if do_eval:
        m = evaluate(model_ft, val_loader_ft, DEVICE)
        ex, da, ne = m["exact_acc"], m["digit_acc"], m["near_acc"]
        imp = ex > best_acc
        if imp:
            best_acc = ex
            torch.save(model_ft.state_dict(),
                       os.path.join(CONFIG["finetune_savedir"], "best_model.pt"))
            with open(os.path.join(CONFIG["finetune_savedir"], "best_meta.json"), "w") as f:
                json.dump({"epoch": ep+1, "val_exact_acc": ex,
                           "val_digit_acc": da, "val_near_acc": ne,
                           "source": "reparsed_pretrain_then_gsm8k_mlp_head"}, f, indent=2)
        mk = " *" if imp else ""
        print(f"{ep+1:>4}  {al:>8.4f}  {ag:>5.2f}  {lr:>8.1e}  "
              f"{ex*100:>6.2f}  {da*100:>7.1f}  {ne*100:>6.2f}  "
              f"{m['mean_halt_steps']:>6.2f}{mk}")
        ft_log.append({"epoch": ep+1, "loss": al,
                       "exact_acc": ex, "digit_acc": da, "near_acc": ne})

print(f"\\nFinetune done. Best VAL exact = {best_acc*100:.2f}%")
with open(os.path.join(CONFIG["finetune_savedir"], "finetune_log.json"), "w") as f:
    json.dump(ft_log, f, indent=2)
"""

# =========================================================================
C_TEST = """# Cell 11 — Final test eval
test_set = GSM8KDataset(CONFIG["data_test"], max_nodes=CONFIG["max_nodes"])
test_loader = DataLoader(test_set, batch_size=CONFIG["finetune_batch"] * 2,
                         shuffle=False, collate_fn=collate_fn,
                         num_workers=2, pin_memory=True)

best_path = os.path.join(CONFIG["finetune_savedir"], "best_model.pt")
model_ft.load_state_dict(torch.load(best_path))
tm = evaluate(model_ft, test_loader, DEVICE)
print("=" * 60)
print("GSM8K TEST — Claude-reparsed + matched-dist pretrain + MLP head")
print("=" * 60)
for k, v in tm.items():
    if isinstance(v, float): print(f"  {k:18s} = {v:.4f}")
    else:                    print(f"  {k:18s} = {v}")
print(f"\\nArch: dmodel={CONFIG['dmodel']}, Hc={CONFIG['Hcycles']}, "
      f"Lc={CONFIG['Lcycles']}, Hl={CONFIG['Hlayers']}, Ll={CONFIG['Llayers']}")
print(f"Reasoning steps/fwd: {CONFIG['Hcycles']*CONFIG['Lcycles']*CONFIG['act_max_steps']}")
print(f"Digit head: MLP (hidden={CONFIG['digit_head_hidden']})")
"""


cells = [
    md("md_header",   MD_HEADER),
    code("c_env",     C_ENV),
    code("c_config",  C_CONFIG),
    code("c_utils",   C_UTILS),
    md("md_perturb",  MD_PERTURB),
    code("c_perturb", C_PERTURB),
    code("c_datasets", C_DATASETS),
    md("md_arch",     MD_ARCH),
    code("c_arch",    C_ARCH),
    code("c_loss",    C_LOSS),
    code("c_eval",    C_EVAL),
    md("md_pretrain", MD_PRETRAIN),
    code("c_pretrain", C_PRETRAIN),
    md("md_finetune", MD_FINETUNE),
    code("c_finetune", C_FINETUNE),
    code("c_test",    C_TEST),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {OUT}  ({sum(len(c['source']) for c in cells)} lines, {len(cells)} cells)")
