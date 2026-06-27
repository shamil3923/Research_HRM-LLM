"""Builder for notebook_pretrain_distmatched_deep.ipynb.

The consolidated experiment combining:
  1. Distribution-matched pretrain data:
     - Phase 1: 10x perturbations of real GSM8K traces (~47K)
     - Phase 2: pure-synthetic samples drawn from extracted op-sequence
                distributions (~20K, optional)
     - Phase 3: SVAMP + MAWPS + ASDiv traces parsed via Python AST,
                plus 5x perturbations each (~34K)
     - Total target: ~100K word-problem-distributional examples
  2. Moderate-deep HRM (~7.5M params):
     - dmodel=256, Hcycles=6, Lcycles=8, Hlayers=5, Llayers=5
     - 192 reasoning steps per forward (4x v3.1's 48, fp16-safe)
  3. Pretrain → fine-tune on real GSM8K

Run:  python build_pretrain_distmatched_deep_notebook.py
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__),
                   "notebook_pretrain_distmatched_deep.ipynb")


def md(cid, text):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(cid, src):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


# =========================================================================
MD_HEADER = """# HRM Pretrain — Distribution-Matched (GSM8K + SVAMP/MAWPS/ASDiv) + Deep Architecture

**Combined hypothesis.** Both **distribution mismatch** and **insufficient
reasoning depth** contribute to the ~30% GSM8K ceiling we observed across
multiple prior experiments. This notebook addresses both:

### Change 1 — Pretrain on word-problem-distributional data

Four sources, all word-problem-derived, parsed into HRM's trace format:

| Source | Volume (after augmentation) | Why |
|---|---|---|
| GSM8K train traces (10× perturbation) | ~47K | Same topology as fine-tune target |
| Pure synth from GSM8K op-sequence dist | ~20K | Variety beyond 4,682 base topologies |
| SVAMP + MAWPS + ASDiv (real) + 5× perturbation each | ~34K | Independent word-problem corpora, similar style |
| **Total target** | **~100K** | All distributionally aligned with GSM8K |

### Change 2 — Moderate-deep HRM (~7.5M params)

| Param | v3.1 | This notebook |
|---|---|---|
| dmodel | 256 | **256** (unchanged — width not the bottleneck) |
| Hcycles | 3 | **6** |
| Lcycles | 4 | **8** |
| Hlayers | 4 | **5** |
| Llayers | 4 | **5** |
| Reasoning steps/fwd | 48 | **192** (4×) |
| fp16 stable? | yes | **yes** (below the 320 crash threshold we hit earlier) |

### Setup

You need these Kaggle datasets attached (right sidebar → + Add Input):
- `shamilmrm/gsm8k-dataset-and-optimizer` (your existing GSM8K parsed data)
- SVAMP / MAWPS / ASDiv — upload as a single dataset, or attach individually
  if available on Kaggle. The notebook gracefully skips any source that
  isn't found.

**Target.** GSM8K test exact accuracy **≥ 45%** (vs prior 30% ceiling).
"""

# =========================================================================
C_ENV = """# Cell 1 — Environment
import sys, os, json, time, math, random, copy, re
import ast as _ast
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
DATA_ROOT = "/kaggle/input/datasets/shamilmrm/gsm8k-dataset-and-optimizer"

# Update these paths to wherever you attached the extra datasets.
# Set to None or non-existent path to skip a source.
EXTRA_DATASETS = {
    "svamp": "/kaggle/input/svamp/SVAMP.json",
    "mawps": "/kaggle/input/mawps/mawps.json",
    "asdiv": "/kaggle/input/asdiv/asdiv.json",
}

CONFIG = dict(
    # --- Model: MODERATE-DEEP HRM (~7.5M params) ---
    dmodel     = 256,
    nheads     = 8,
    Hcycles    = 6,      # was 3 in v3.1 (4x deeper reasoning)
    Lcycles    = 8,      # was 4 in v3.1
    Hlayers    = 5,      # was 4 in v3.1
    Llayers    = 5,      # was 4 in v3.1
    max_nodes  = 50,

    # --- Data paths ---
    data_train = f"{DATA_ROOT}/gsm8k_train_split.json",
    data_val   = f"{DATA_ROOT}/gsm8k_val_split.json",
    data_test  = f"{DATA_ROOT}/gsm8k_test_clean.json",

    # --- Distribution-matched corpus generation ---
    perturbations_per_gsm8k = 10,
    perturbations_per_extra = 5,
    n_pure_synthetic        = 20_000,
    max_intermediate        = 100_000,
    corpus_cache_path       = "/kaggle/working/distmatched_deep_corpus.json",

    # --- Pretrain ---
    pretrain_savedir    = "/kaggle/working/checkpoints/hrm_pretrain_distmatched_deep",
    pretrain_epochs     = 20,
    pretrain_batch      = 96,           # deeper model — smaller batch fits T4 16GB
    pretrain_peak_lr    = 2e-4,
    pretrain_lr_floor   = 1e-5,
    pretrain_eval_every = 2,

    # --- Fine-tune on real GSM8K ---
    finetune_savedir    = "/kaggle/working/checkpoints/hrm_finetune_distmatched_deep",
    finetune_epochs     = 150,
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
print(f"  Reasoning steps per forward: {CONFIG['Hcycles']*CONFIG['Lcycles']*CONFIG['act_max_steps']}")
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
MD_ANALYZER = """## GSM8K Distribution Analyzer

Extract empirical distributions from real GSM8K training traces so we
can generate distribution-matched synthetic data.
"""

C_ANALYZER = '''# Cell 4 — Analyze GSM8K training distribution

def analyze_gsm8k_distribution(json_path):
    with open(json_path) as f: data = json.load(f)
    op_counts, op_seqs, leaf_mags = [], [], []
    for item in data:
        steps = item.get("trace", {}).get("steps", [])
        if not steps: continue
        op_counts.append(len(steps))
        seq = []
        var_set = set()
        for s in steps:
            seq.append(s.get("op", ""))
            for arg_key in ["arg1", "arg2"]:
                a = s.get(arg_key, "")
                if isinstance(a, str) and a in var_set:
                    continue
                if isinstance(a, (int, float)):
                    leaf_mags.append(abs(float(a)))
                elif isinstance(a, str):
                    try: leaf_mags.append(abs(float(a)))
                    except: pass
            rk = s.get("result", "")
            if rk: var_set.add(rk)
        op_seqs.append(tuple(seq))
    return {
        "n_traces": len(op_counts),
        "op_count_hist": Counter(op_counts),
        "op_seq_hist":   Counter(op_seqs),
        "leaf_magnitudes": leaf_mags,
    }


print("Analyzing GSM8K training distribution...")
GSM_STATS = analyze_gsm8k_distribution(CONFIG["data_train"])

print(f"\\nTotal traces: {GSM_STATS['n_traces']}")
print(f"\\nStep-count distribution:")
total = sum(GSM_STATS["op_count_hist"].values())
for k in sorted(GSM_STATS["op_count_hist"].keys()):
    v = GSM_STATS["op_count_hist"][k]
    bar = "█" * int(v * 40 / max(GSM_STATS["op_count_hist"].values()))
    print(f"  {k:>2} ops: {v:>5}  ({v/total*100:5.1f}%)  {bar}")

all_ops = Counter()
for seq in GSM_STATS["op_seq_hist"].elements():
    for op in seq: all_ops[op] += 1
print(f"\\nOp frequency:")
tot_ops = sum(all_ops.values())
for op, c in sorted(all_ops.items(), key=lambda x: -x[1]):
    print(f"  {op:>10s}: {c:>6}  ({c/tot_ops*100:5.1f}%)")

mags = np.array(GSM_STATS["leaf_magnitudes"])
print(f"\\nLeaf magnitudes (n={len(mags)}):  median={np.median(mags):.0f}  "
      f"p75={np.percentile(mags, 75):.0f}  p95={np.percentile(mags, 95):.0f}  "
      f"max={mags.max():.0f}")
LEAF_MAG_SAMPLER = mags.copy()
'''

# =========================================================================
MD_EXTRA = """## Load SVAMP / MAWPS / ASDiv via AST parsing

Each of these datasets ships explicit equation strings (e.g. `"10 - 4"`
or `"X = (5+3)*2"`). We parse them with Python's AST module and walk
the tree in post-order, emitting HRM-format trace steps. Same technique
as the DeepMind Math notebook from earlier.

If any source isn't attached, the loader skips it without error.
"""

C_EXTRA = '''# Cell 5 — Parse SVAMP / MAWPS / ASDiv via AST

_BINOP = {_ast.Add: "add", _ast.Sub: "sub",
          _ast.Mult: "mul", _ast.Div: "div", _ast.FloorDiv: "div"}


def expr_to_trace(expr_str, max_steps=20):
    try:
        tree = _ast.parse(expr_str.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None, None
    steps, counter = [], [0]
    def walk(node):
        if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
            v = node.value
            if isinstance(v, float) and v != int(v): return None, None
            iv = int(v)
            if abs(iv) > _AUG_MAX_ABS: return None, None
            return float(iv), iv
        if isinstance(node, _ast.UnaryOp):
            if isinstance(node.op, _ast.UAdd): return walk(node.operand)
            if isinstance(node.op, _ast.USub):
                ir, iv = walk(node.operand)
                if iv is None: return None, None
                if isinstance(ir, float):
                    return float(-iv), -iv
                rk = f"v{counter[0]}"; counter[0] += 1
                steps.append({"op": "sub", "arg1": 0.0, "arg2": ir,
                              "result": rk, "result_value": float(-iv)})
                return rk, -iv
            return None, None
        if isinstance(node, _ast.BinOp):
            op_name = _BINOP.get(type(node.op))
            if op_name is None: return None, None
            la, lv = walk(node.left)
            if lv is None: return None, None
            ra, rv = walk(node.right)
            if rv is None: return None, None
            if   op_name == "add": res = lv + rv
            elif op_name == "sub": res = lv - rv
            elif op_name == "mul": res = lv * rv
            else:
                if rv == 0 or lv % rv != 0: return None, None
                res = lv // rv
            if abs(res) > _AUG_MAX_ABS: return None, None
            if len(steps) >= max_steps: return None, None
            rk = f"v{counter[0]}"; counter[0] += 1
            steps.append({"op": op_name, "arg1": la, "arg2": ra,
                          "result": rk, "result_value": float(res)})
            return rk, res
        return None, None
    root_r, root_v = walk(tree)
    if root_v is None or not steps: return None, None
    if not isinstance(root_r, str) or root_r != steps[-1]["result"]:
        return None, None
    return {"steps": steps, "final_answer": steps[-1]["result"]}, int(root_v)


def _to_int(x):
    if isinstance(x, (int, float)) and float(x) == int(x):
        return int(x)
    s = str(x).strip()
    m = re.match(r"-?\\d+", s)
    if m:
        try: return int(m.group())
        except: return None
    try:
        v = float(s)
        if v == int(v): return int(v)
    except: pass
    return None


def load_svamp(path):
    with open(path) as f: data = json.load(f)
    out = []
    for item in data:
        eq  = item.get("Equation", "")
        ans = item.get("Answer", None)
        if not eq or ans is None: continue
        trace, computed = expr_to_trace(eq)
        if trace is None: continue
        ans_int = _to_int(ans)
        if ans_int is None or computed != ans_int: continue
        out.append({"trace": trace, "target": ans_int, "source": "svamp"})
    return out


def load_mawps(path):
    """MAWPS often has lEquations like 'X = 10 - 4' or just '10 - 4'."""
    with open(path) as f: data = json.load(f)
    out = []
    for item in data:
        eqs  = item.get("lEquations") or item.get("Equation") or []
        sols = item.get("lSolutions") or [item.get("Answer")]
        if isinstance(eqs, str): eqs = [eqs]
        if not eqs or not sols or sols[0] is None: continue
        eq = eqs[0]
        eq = re.sub(r"^[a-zA-Z_]+\\s*=\\s*", "", str(eq)).strip()
        trace, computed = expr_to_trace(eq)
        if trace is None: continue
        ans_int = _to_int(sols[0])
        if ans_int is None or computed != ans_int: continue
        out.append({"trace": trace, "target": ans_int, "source": "mawps"})
    return out


def load_asdiv(path):
    with open(path) as f: data = json.load(f)
    if isinstance(data, dict): data = data.get("Problems", data.get("data", []))
    out = []
    for item in data:
        formula = item.get("Formula") or item.get("formula") or ""
        ans     = item.get("Answer")  or item.get("answer")
        if not formula or ans is None: continue
        formula = str(formula).split("=")[0].strip()
        trace, computed = expr_to_trace(formula)
        if trace is None: continue
        ans_int = _to_int(ans)
        if ans_int is None or computed != ans_int: continue
        out.append({"trace": trace, "target": ans_int, "source": "asdiv"})
    return out


EXTRA_TRACES = []
for name, path in EXTRA_DATASETS.items():
    if not path or not os.path.exists(path):
        print(f"  [skip] {name}: {path} not found")
        continue
    try:
        if   name == "svamp": parsed = load_svamp(path)
        elif name == "mawps": parsed = load_mawps(path)
        elif name == "asdiv": parsed = load_asdiv(path)
        else: parsed = []
        EXTRA_TRACES.extend(parsed)
        print(f"  [loaded] {name}: {len(parsed)} traces from {path}")
    except Exception as e:
        print(f"  [error] {name}: {e}")

print(f"\\nTotal extra dataset traces: {len(EXTRA_TRACES)}")
'''

# =========================================================================
MD_GEN = """## Build the consolidated pretrain corpus

Three phases:

1. **Phase 1**: perturb each real GSM8K trace 10× by replacing leaves
   with values from `LEAF_MAG_SAMPLER`. Reject if any intermediate is
   non-integer or overflows. Topology preserved.

2. **Phase 2**: pure-synthetic from op-sequences in `GSM_STATS`
   (op_seq_hist), filled with fresh leaves and 30% reference probability.

3. **Phase 3**: include each SVAMP/MAWPS/ASDiv trace plus 5× perturbations.

Caches the final corpus to disk so re-runs are instant.
"""

C_GEN = '''# Cell 6 — Generate consolidated distribution-matched corpus

def _sample_leaf(rng):
    v = rng.choice(LEAF_MAG_SAMPLER)
    jitter = rng.randint(-2, 2)
    return max(1, int(round(v)) + jitter)


def perturb_real_trace(real_trace, rng, max_tries=20, max_abs=None):
    if max_abs is None: max_abs = CONFIG["max_intermediate"]
    steps = real_trace.get("steps", [])
    if not steps: return None, None
    for _ in range(max_tries):
        new_steps = []
        values = {}
        ok = True
        for s in steps:
            op = s.get("op", "")
            def maybe_replace(a):
                if isinstance(a, (int, float)):
                    sign = -1 if float(a) < 0 else 1
                    return float(sign * _sample_leaf(rng))
                if isinstance(a, str):
                    if a in values: return a
                    try:
                        sign = -1 if float(a) < 0 else 1
                        return float(sign * _sample_leaf(rng))
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
        fa_var = real_trace.get("final_answer", "")
        if fa_var not in values: continue
        fa = values[fa_var]
        if abs(fa) > _AUG_MAX_ABS: continue
        return {"steps": new_steps, "final_answer": fa_var}, int(round(fa))
    return None, None


def _synthesize_from_op_seq(op_seq, rng, max_tries=10):
    max_abs = CONFIG["max_intermediate"]
    for _ in range(max_tries):
        steps, values = [], {}
        ok = True
        for i, op in enumerate(op_seq):
            def make_arg():
                if values and rng.random() < 0.30:
                    return rng.choice(list(values.keys()))
                sign = 1 if rng.random() < 0.95 else -1
                return float(sign * _sample_leaf(rng))
            arg1 = make_arg(); arg2 = make_arg()
            def resolve(a):
                if isinstance(a, (int, float)): return float(a)
                if isinstance(a, str) and a in values: return values[a]
                return 0.0
            v1 = resolve(arg1); v2 = resolve(arg2)
            if   op == "add": rv = v1 + v2
            elif op == "sub": rv = v1 - v2
            elif op == "mul": rv = v1 * v2
            elif op == "div":
                if v2 == 0 or v1 % v2 != 0: ok = False; break
                rv = v1 / v2
            else: rv = v1
            if not np.isfinite(rv) or abs(rv) > max_abs or rv != int(rv):
                ok = False; break
            rk = f"v{i}"
            values[rk] = rv
            steps.append({"op": op, "arg1": arg1, "arg2": arg2,
                          "result": rk, "result_value": rv})
        if not ok or not steps: continue
        fa_var = steps[-1]["result"]
        return {"steps": steps, "final_answer": fa_var}, int(values[fa_var])
    return None, None


def materialize_corpus(force_rebuild=False):
    path = CONFIG["corpus_cache_path"]
    if not force_rebuild and os.path.exists(path):
        print(f"Loading cached corpus from {path}")
        with open(path) as f: return json.load(f)

    with open(CONFIG["data_train"]) as f:
        real_gsm8k = json.load(f)
    print(f"Loaded {len(real_gsm8k)} real GSM8K traces")

    rng = random.Random(20260527)
    corpus = []

    # Phase 1 — perturb GSM8K
    print(f"\\nPhase 1: {CONFIG['perturbations_per_gsm8k']}x perturbations of GSM8K traces...")
    n_p1 = n_p1_attempts = 0
    for i, item in enumerate(real_gsm8k):
        real_trace = item.get("trace", {})
        for _ in range(CONFIG["perturbations_per_gsm8k"]):
            n_p1_attempts += 1
            new_t, tgt = perturb_real_trace(real_trace, rng)
            if new_t is None: continue
            corpus.append({"trace": new_t, "target": tgt, "source": "gsm8k_perturbed"})
            n_p1 += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(real_gsm8k)}]  kept {n_p1}")
    print(f"  Phase 1 done: {n_p1} kept from {n_p1_attempts} attempts ({n_p1/max(1,n_p1_attempts):.1%})")

    # Phase 2 — pure synth from extracted op sequences
    n_p2 = 0
    if CONFIG["n_pure_synthetic"] > 0:
        print(f"\\nPhase 2: pure synth from extracted op-sequences (target {CONFIG['n_pure_synthetic']})...")
        op_seq_list = list(GSM_STATS["op_seq_hist"].elements())
        n_p2_attempts = 0
        target = CONFIG["n_pure_synthetic"]
        while n_p2 < target and n_p2_attempts < 5 * target:
            n_p2_attempts += 1
            seq = rng.choice(op_seq_list)
            new_t, tgt = _synthesize_from_op_seq(seq, rng)
            if new_t is None: continue
            corpus.append({"trace": new_t, "target": tgt, "source": "pure_synth"})
            n_p2 += 1
            if n_p2 % 5000 == 0:
                print(f"  [{n_p2}/{target}]")
        print(f"  Phase 2 done: {n_p2} kept")

    # Phase 3 — SVAMP/MAWPS/ASDiv + perturbations
    n_p3 = 0
    if EXTRA_TRACES:
        print(f"\\nPhase 3: integrating {len(EXTRA_TRACES)} extra dataset traces + "
              f"{CONFIG['perturbations_per_extra']}x perturbations each...")
        for ex in EXTRA_TRACES:
            corpus.append({"trace": ex["trace"], "target": ex["target"],
                           "source": ex.get("source", "extra")})
            n_p3 += 1
            for _ in range(CONFIG["perturbations_per_extra"]):
                new_t, tgt = perturb_real_trace(ex["trace"], rng)
                if new_t is not None:
                    corpus.append({"trace": new_t, "target": tgt,
                                   "source": f"{ex.get('source')}_perturbed"})
                    n_p3 += 1
        print(f"  Phase 3 done: {n_p3} added")

    rng.shuffle(corpus)
    print(f"\\nTotal corpus: {len(corpus)}  (phase1={n_p1}, phase2={n_p2}, phase3={n_p3})")

    with open(path, "w") as f: json.dump(corpus, f)
    print(f"Cached to {path}")
    return corpus


CORPUS = materialize_corpus()
print(f"\\nFinal corpus size: {len(CORPUS)}")
'''

# =========================================================================
C_VERIFY = '''# Cell 7 — Distribution verification

def report_distribution(records, label):
    op_counts, op_freq, leaf_mags = [], Counter(), []
    for r in records:
        steps = r.get("trace", {}).get("steps", [])
        if not steps: continue
        op_counts.append(len(steps))
        for s in steps:
            op_freq[s.get("op", "")] += 1
            for a in [s.get("arg1"), s.get("arg2")]:
                if isinstance(a, (int, float)):
                    leaf_mags.append(abs(float(a)))
                elif isinstance(a, str):
                    try: leaf_mags.append(abs(float(a)))
                    except: pass
    print(f"\\n{label} (n={len(records)}):")
    c = Counter(op_counts)
    total = sum(c.values())
    print(f"  Op counts (top 6):")
    for k in sorted(c.keys())[:6]:
        print(f"    {k:>2}: {c[k]/total*100:5.1f}%")
    print(f"  Op frequency:")
    tot_ops = sum(op_freq.values())
    for op in ["add", "sub", "mul", "div"]:
        print(f"    {op:>5}: {op_freq[op]/tot_ops*100:5.1f}%")
    mags = np.array(leaf_mags)
    print(f"  Leaf mag: median={np.median(mags):.0f}  p75={np.percentile(mags,75):.0f}  "
          f"p95={np.percentile(mags,95):.0f}  max={mags.max():.0f}")


with open(CONFIG["data_train"]) as f:
    _real = json.load(f)
report_distribution(_real,  "REAL GSM8K train")
report_distribution(CORPUS, "GENERATED corpus")
'''

# =========================================================================
C_DATASETS = """# Cell 8 — Dataset classes

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
                 augment_p=0.5, augment_max_value=200):
        with open(json_file) as f: raw = json.load(f)
        self.max_nodes = max_nodes
        self.augment = augment
        self.augment_p = augment_p
        self.augment_max_value = augment_max_value
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
        if self.augment and random.random() < self.augment_p:
            trace, _ = self.raw[i]
            new_t, new_tgt = perturb_real_trace(trace, random)
            if new_t is not None and new_tgt is not None:
                return _sample_to_tensors(new_t, new_tgt, self.max_nodes)
        return self.cached[i]


random.seed(20260527)
random.shuffle(CORPUS)
_split = int(0.98 * len(CORPUS))
CORPUS_TRAIN = CORPUS[:_split]
CORPUS_VAL   = CORPUS[_split:]
print(f"Corpus train={len(CORPUS_TRAIN)}  val={len(CORPUS_VAL)}")
"""

# =========================================================================
C_ARCH = """# Cell 9 — Architecture (IDENTICAL to v3.1, works with deeper Hc/Lc/Hl/Ll)

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


class HRMForMath(nn.Module):
    def __init__(self, vsz=9, d=512, heads=8, Hc=4, Lc=8, Hl=8, Ll=8, ex=2.0, slen=50):
        super().__init__()
        self.Hc, self.Lc = Hc, Lc
        self.bridge = GraphAwareBridge(vsz, d, vf=NODE_VAL_DIM, gh=128, gl=3, heads=4)
        self.pos    = nn.Embedding(slen, d)
        self.Hmod   = HRMModule(Hl, d, heads, ex)
        self.Lmod   = HRMModule(Ll, d, heads, ex)
        self.Hi     = nn.Parameter(torch.randn(d) * 0.02)
        self.Li     = nn.Parameter(torch.randn(d) * 0.02)
        self.dhead  = nn.Linear(d, MAX_DIGITS * DIGIT_VOCAB_SIZE)
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
        dl = self.dhead(zH).reshape(B, N, MAX_DIGITS, DIGIT_VOCAB_SIZE)
        ql = self.qhead(self.qnorm(zH[:, 0]))
        return dl, ql[:, 0], ql[:, 1], zH.detach(), zL.detach()

print("Architecture defined.")
"""

# =========================================================================
C_LOSS = """# Cell 10 — Losses + AdamATan2
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
C_EVAL = """# Cell 11 — Evaluation
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
MD_PRETRAIN = """## Pretrain on the consolidated corpus

Moderate-deep HRM (~7.5M params, 192 reasoning steps/fwd) trains on
~100K word-problem-distributional examples. 20 epochs, single global
cosine schedule.
"""

C_PRETRAIN = """# Cell 12 — Pretrain

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
                   slen=CONFIG["max_nodes"]).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\\nHRMForMath  {n_params/1e6:.2f}M params")
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
MD_FINETUNE = """## Fine-tune on real GSM8K

Lower LR (5e-5) to preserve distribution-aligned pretrain features.
150 epochs, eval every 10.
"""

C_FINETUNE = """# Cell 13 — Fine-tune on real GSM8K

train_set_ft = GSM8KDataset(CONFIG["data_train"], max_nodes=CONFIG["max_nodes"],
                            augment=True, augment_p=CONFIG["augment_p"],
                            augment_max_value=CONFIG["augment_max_value"])
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
                      slen=CONFIG["max_nodes"]).to(DEVICE)
missing, unexpected = model_ft.load_state_dict(torch.load(PRETRAIN_CKPT), strict=False)
assert not missing,    f"Pretrained ckpt missing keys: {missing}"
assert not unexpected, f"Pretrained ckpt unexpected keys: {unexpected}"
print(f"\\nLoaded distribution-matched deep pretrain.  "
      f"Train={len(train_set_ft)}  Val={len(val_set_ft)}")

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
                           "source": "distmatched_deep_pretrain_then_gsm8k"}, f, indent=2)
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
C_TEST = """# Cell 14 — Final test eval
test_set = GSM8KDataset(CONFIG["data_test"], max_nodes=CONFIG["max_nodes"])
test_loader = DataLoader(test_set, batch_size=CONFIG["finetune_batch"] * 2,
                         shuffle=False, collate_fn=collate_fn,
                         num_workers=2, pin_memory=True)

best_path = os.path.join(CONFIG["finetune_savedir"], "best_model.pt")
model_ft.load_state_dict(torch.load(best_path))
tm = evaluate(model_ft, test_loader, DEVICE)
print("=" * 60)
print("GSM8K TEST — distribution-matched deep pretrain → finetune")
print("=" * 60)
for k, v in tm.items():
    if isinstance(v, float): print(f"  {k:18s} = {v:.4f}")
    else:                    print(f"  {k:18s} = {v}")
print(f"\\nArch: dmodel={CONFIG['dmodel']}, Hc={CONFIG['Hcycles']}, "
      f"Lc={CONFIG['Lcycles']}, Hl={CONFIG['Hlayers']}, Ll={CONFIG['Llayers']}")
print(f"Reasoning steps/fwd: {CONFIG['Hcycles']*CONFIG['Lcycles']*CONFIG['act_max_steps']}")
"""


cells = [
    md("md_header",    MD_HEADER),
    code("c_env",      C_ENV),
    code("c_config",   C_CONFIG),
    code("c_utils",    C_UTILS),
    md("md_analyzer",  MD_ANALYZER),
    code("c_analyzer", C_ANALYZER),
    md("md_extra",     MD_EXTRA),
    code("c_extra",    C_EXTRA),
    md("md_gen",       MD_GEN),
    code("c_gen",      C_GEN),
    code("c_verify",   C_VERIFY),
    code("c_datasets", C_DATASETS),
    code("c_arch",     C_ARCH),
    code("c_loss",     C_LOSS),
    code("c_eval",     C_EVAL),
    md("md_pretrain",  MD_PRETRAIN),
    code("c_pretrain", C_PRETRAIN),
    md("md_finetune",  MD_FINETUNE),
    code("c_finetune", C_FINETUNE),
    code("c_test",     C_TEST),
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
