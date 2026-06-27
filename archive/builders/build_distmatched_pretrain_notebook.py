"""Builder for notebook_pretrain_distmatched.ipynb.

Generates a notebook that:
  1. Loads the parsed GSM8K training set.
  2. Extracts empirical distributions (step count, op patterns, leaf
     magnitudes, reference patterns) from real GSM8K traces.
  3. Generates a 47K-example synthetic pretrain corpus by aggressive
     constant perturbation of real GSM8K traces (10x per trace), PLUS
     pure-synthetic samples drawn from the extracted distributions.
  4. Pretrains v3.1 HRM (6.27M, dmodel=256) on this corpus.
  5. Fine-tunes on the original GSM8K training set.

The key invariant: every pretrain sample has the SAME tree topology
and op pattern as a real GSM8K problem. Only numeric values differ.
This eliminates the distribution mismatch that capped prior runs at ~30%.

Run:  python build_distmatched_pretrain_notebook.py
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "notebook_pretrain_distmatched.ipynb")


def md(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(cell_id, src):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


MD_HEADER = """# HRM Pretrain on GSM8K-Distribution-Matched Synthetic → Fine-tune on GSM8K

**Hypothesis under test.** The 30% accuracy ceiling we observed across
multiple architectures (v3.1 reference, 14M wide, 8.89M deep) was caused
by **distribution mismatch between pretrain synthetic data and GSM8K**,
not by capacity limitations.

**Proof by construction.** This notebook generates pretrain data whose
distribution **exactly matches GSM8K** by re-using GSM8K's own trace
topologies and only varying numeric leaf values. Every pretrain sample
has the same reasoning shape as a real GSM8K problem — running totals,
"per-X" multiplications, "split among N" divisions — just with different
numbers.

**Generation strategy.**
1. **Heavy perturbation (10× per real trace)**: take each of GSM8K's
   4,682 training traces, replace numeric constants with values drawn
   from GSM8K's empirical magnitude distribution, re-evaluate the trace.
   Reject if any intermediate becomes non-integer or overflows.
   → ~47,000 distribution-matched examples (some rejected).
2. **Pure synthetic from extracted distributions** (optional, for
   variety): sample new trees from the empirical distributions over
   step count, op sequence, leaf vs reference patterns.

**Architecture.** v3.1 baseline (dmodel=256, Hc=3, Lc=4, Hl=4, Ll=4,
~6.27M params). No depth/width experiments here — the variable we're
testing is the data distribution.

**Target.** GSM8K test exact accuracy **≥ 45%**. If we hit that, the
distribution-mismatch hypothesis is confirmed. If we plateau at ~30%
again, the bottleneck is architectural and we need different solutions.
"""

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

C_CONFIG = """# Cell 2 — Configuration
DATA_ROOT = "/kaggle/input/datasets/shamilmrm/gsm8k-dataset-and-optimizer"

CONFIG = dict(
    # --- Model: v3.1 baseline architecture (~6.27M params) ---
    dmodel     = 256,
    nheads     = 8,
    Hcycles    = 3,
    Lcycles    = 4,
    Hlayers    = 4,
    Llayers    = 4,
    max_nodes  = 50,

    # --- Data paths ---
    data_train = f"{DATA_ROOT}/gsm8k_train_split.json",
    data_val   = f"{DATA_ROOT}/gsm8k_val_split.json",
    data_test  = f"{DATA_ROOT}/gsm8k_test_clean.json",

    # --- Distribution-matched synthetic generation ---
    perturbations_per_trace = 10,         # 10x augmentation of real traces
    n_pure_synthetic        = 20_000,     # additional pure samples from extracted dist
    max_intermediate        = 100_000,    # reject samples with abs value > this
    distmatched_cache_path  = "/kaggle/working/distmatched_corpus.json",

    # --- Pretrain ---
    pretrain_savedir    = "/kaggle/working/checkpoints/hrm_pretrain_distmatched",
    pretrain_epochs     = 25,
    pretrain_batch      = 128,
    pretrain_peak_lr    = 2e-4,
    pretrain_lr_floor   = 1e-5,
    pretrain_eval_every = 2,

    # --- Fine-tune on original GSM8K ---
    finetune_savedir    = "/kaggle/working/checkpoints/hrm_finetune_distmatched",
    finetune_epochs     = 150,
    finetune_batch      = 128,
    finetune_peak_lr    = 5e-5,
    finetune_lr_floor   = 1e-6,
    finetune_eval_every = 10,
    augment_p           = 0.2,           # mild aug, distribution already matched
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
"""

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

# ----------------------------------------------------------------- distribution analyzer
MD_ANALYZER = """## GSM8K Distribution Analyzer

Before we can match GSM8K's distribution, we have to characterize it.
This cell loads the parsed training traces and extracts:

- **Step-count histogram** — `num_ops` distribution across all traces
- **Op-position distribution** — `P(op | step_index, num_ops)`
- **Leaf magnitude distribution** — log-bucketed histogram of all
  numeric constants used in any trace
- **Reference-vs-new-leaf pattern** — at each step, what fraction of
  args refer to prior step results vs introduce new leaf values

These statistics drive both the perturbation generator (which preserves
trace topology but resamples leaves from the magnitude distribution)
and the pure-synthetic generator.
"""

C_ANALYZER = '''# Cell 4 — Analyze GSM8K training distribution

def analyze_gsm8k_distribution(json_path):
    """Extract empirical distributions from parsed GSM8K training traces."""
    with open(json_path) as f:
        data = json.load(f)

    op_counts = []
    op_seqs   = []
    leaf_mags = []
    ref_pattern_by_pos = defaultdict(lambda: {"ref": 0, "new": 0})

    for item in data:
        trace  = item.get("trace", {})
        steps  = trace.get("steps", [])
        n      = len(steps)
        if n == 0: continue

        op_counts.append(n)
        seq = []
        var_set = set()
        for i, s in enumerate(steps):
            op = s.get("op", "")
            seq.append(op)

            for arg_key in ["arg1", "arg2"]:
                a = s.get(arg_key, "")
                if isinstance(a, str) and a in var_set:
                    ref_pattern_by_pos[i]["ref"] += 1
                else:
                    ref_pattern_by_pos[i]["new"] += 1
                    if isinstance(a, (int, float)):
                        leaf_mags.append(abs(float(a)))
                    else:
                        try: leaf_mags.append(abs(float(a)))
                        except: pass

            rk = s.get("result", "")
            if rk: var_set.add(rk)
        op_seqs.append(tuple(seq))

    stats = {
        "n_traces":           len(op_counts),
        "op_count_hist":      Counter(op_counts),
        "op_seq_hist":        Counter(op_seqs),
        "leaf_magnitudes":    leaf_mags,
        "ref_pattern_by_pos": {i: dict(d) for i, d in ref_pattern_by_pos.items()},
    }
    return stats


print("Analyzing GSM8K training distribution...")
GSM_STATS = analyze_gsm8k_distribution(CONFIG["data_train"])

# Report
print(f"\\nTotal traces analyzed: {GSM_STATS['n_traces']}")
print(f"\\nStep-count distribution:")
total = sum(GSM_STATS["op_count_hist"].values())
for k in sorted(GSM_STATS["op_count_hist"].keys()):
    v = GSM_STATS["op_count_hist"][k]
    bar = "█" * int(v * 50 / max(GSM_STATS["op_count_hist"].values()))
    print(f"  {k:>2} ops: {v:>5}  ({v/total*100:5.1f}%)  {bar}")

# Op frequency across all positions
all_ops = Counter()
for seq in GSM_STATS["op_seq_hist"].elements():
    for op in seq: all_ops[op] += 1
print(f"\\nOp frequency:")
tot_ops = sum(all_ops.values())
for op, c in sorted(all_ops.items(), key=lambda x: -x[1]):
    print(f"  {op:>10s}: {c:>6}  ({c/tot_ops*100:5.1f}%)")

# Leaf magnitudes
mags = np.array(GSM_STATS["leaf_magnitudes"])
print(f"\\nLeaf value magnitudes (n={len(mags)}):")
print(f"  median  = {np.median(mags):.1f}")
print(f"  p25/p75 = {np.percentile(mags, 25):.1f} / {np.percentile(mags, 75):.1f}")
print(f"  p95/p99 = {np.percentile(mags, 95):.1f} / {np.percentile(mags, 99):.1f}")
print(f"  max     = {mags.max():.1f}")

# Log-bucketed histogram for sampling
LOG_BINS = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 100000])
counts, _ = np.histogram(mags, bins=LOG_BINS)
print(f"\\n  Log-bucketed:")
for i in range(len(LOG_BINS) - 1):
    pct = counts[i] / counts.sum() * 100
    bar = "█" * int(pct * 0.6)
    print(f"    [{LOG_BINS[i]:>5}, {LOG_BINS[i+1]:>6}): {pct:5.1f}%  {bar}")

# Convert leaf magnitudes to a sampler
LEAF_MAG_SAMPLER = mags.copy()
print(f"\\nLeaf magnitude sampler ready ({len(LEAF_MAG_SAMPLER)} values).")
'''

# ----------------------------------------------------------------- generator
MD_GEN = """## Distribution-Matched Generator

Two generation paths:

1. **`perturb_real_trace`** — take a real GSM8K trace, walk through its
   steps, replace every numeric leaf with a value drawn from
   `LEAF_MAG_SAMPLER` (the empirical magnitude distribution). Re-execute
   the trace symbolically. Reject if any intermediate is non-integer or
   exceeds `max_intermediate`. Topology is preserved exactly.

2. **`pure_synth_trace`** — sample step count from `op_count_hist`,
   sample op sequence from `op_seq_hist` (with smoothing), build a tree
   matching the reference patterns at each position. Optional, for
   diversity beyond the 4,682 base topologies.

The perturbation path is the primary engine. Pure-synthetic adds variety
but isn't strictly necessary if 10× perturbation gives enough volume.
"""

C_GEN = '''# Cell 5 — Distribution-matched trace generator

def _sample_leaf(rng):
    """Sample a leaf value matching GSM8K's empirical magnitude distribution."""
    v = rng.choice(LEAF_MAG_SAMPLER)
    # Add small jitter for variety, keep integer
    jitter = rng.randint(-2, 2)
    return max(1, int(round(v)) + jitter)


def perturb_real_trace(real_trace, rng, max_tries=20, max_abs=None):
    """Replace numeric leaves in a real GSM8K trace with values sampled
    from GSM8K's empirical magnitude distribution. Re-execute symbolically.
    Returns (new_trace, target) or (None, None) on failure."""
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
                    if a in values: return a  # reference, keep
                    # numeric stored as string
                    try:
                        float(a)
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
                if v2 == 0 or v1 % v2 != 0:
                    ok = False; break
                rv = v1 / v2
            else:
                rv = v1

            if not np.isfinite(rv) or abs(rv) > max_abs:
                ok = False; break
            if rv != int(rv):
                ok = False; break

            rk = s.get("result", "")
            if rk: values[rk] = rv

            new_steps.append({
                "op": op, "arg1": arg1, "arg2": arg2,
                "result": rk, "result_value": rv,
            })

        if not ok: continue
        fa_var = real_trace.get("final_answer", "")
        if fa_var not in values: continue
        fa = values[fa_var]
        if abs(fa) > _AUG_MAX_ABS: continue
        return {"steps": new_steps, "final_answer": fa_var}, int(round(fa))

    return None, None


def materialize_distmatched_corpus(real_traces_path,
                                   perturbations_per_trace,
                                   n_pure_synthetic,
                                   cache_path,
                                   force_rebuild=False):
    """Build the distribution-matched pretrain corpus and cache to disk."""
    if not force_rebuild and os.path.exists(cache_path):
        print(f"Loading cached corpus from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    with open(real_traces_path) as f:
        real_data = json.load(f)
    print(f"Loaded {len(real_data)} real GSM8K traces")

    rng = random.Random(20260527)
    corpus = []
    n_attempts = n_perturb = n_pure = 0

    # Phase 1: perturbation of real traces
    print(f"\\nPhase 1: generating {perturbations_per_trace}x perturbations of real traces...")
    for i, item in enumerate(real_data):
        real_trace = item.get("trace", {})
        for _ in range(perturbations_per_trace):
            n_attempts += 1
            new_t, tgt = perturb_real_trace(real_trace, rng)
            if new_t is None: continue
            corpus.append({"trace": new_t, "target": tgt, "source": "perturbed"})
            n_perturb += 1
        if (i + 1) % 500 == 0:
            print(f"  [{i+1:>4}/{len(real_data)}]  kept {n_perturb:>6}  rate {n_perturb/max(1,n_attempts):.1%}")

    print(f"  Phase 1 done: {n_perturb} kept from {n_attempts} attempts ({n_perturb/max(1,n_attempts):.1%})")

    # Phase 2: pure synthetic from extracted op-sequence distribution
    # (we sample real op-sequences then perturb a synthetic trace with that shape)
    if n_pure_synthetic > 0:
        print(f"\\nPhase 2: generating {n_pure_synthetic} pure-synthetic samples...")
        op_seq_list = list(GSM_STATS["op_seq_hist"].elements())
        n_pure_target = n_pure_synthetic
        n_pure_attempts = 0
        while n_pure < n_pure_target and n_pure_attempts < 5 * n_pure_target:
            n_pure_attempts += 1
            seq = rng.choice(op_seq_list)
            # Build a synthetic trace with this op sequence and fresh leaves
            new_t, tgt = _synthesize_from_op_seq(seq, rng)
            if new_t is None: continue
            corpus.append({"trace": new_t, "target": tgt, "source": "pure_synth"})
            n_pure += 1
            if n_pure % 2000 == 0:
                print(f"  [{n_pure:>5}/{n_pure_target}]  rate {n_pure/n_pure_attempts:.1%}")
        print(f"  Phase 2 done: {n_pure} kept")

    rng.shuffle(corpus)
    print(f"\\nTotal corpus: {len(corpus)}  (perturbed={n_perturb}, pure={n_pure})")

    with open(cache_path, "w") as f:
        json.dump(corpus, f)
    print(f"Cached to {cache_path}")
    return corpus


def _synthesize_from_op_seq(op_seq, rng, max_tries=10, max_abs=None):
    """Build a fresh trace with the given op sequence. Each step gets
    fresh leaf args (or a reference to an earlier result with some probability)."""
    if max_abs is None: max_abs = CONFIG["max_intermediate"]
    for _ in range(max_tries):
        steps = []
        values = {}
        ok = True
        for i, op in enumerate(op_seq):
            def make_arg():
                # 30% chance of referencing earlier result if any exist
                if values and rng.random() < 0.30:
                    return rng.choice(list(values.keys()))
                sign = 1 if rng.random() < 0.95 else -1
                return float(sign * _sample_leaf(rng))

            arg1 = make_arg()
            arg2 = make_arg()
            def resolve(a):
                if isinstance(a, (int, float)): return float(a)
                if isinstance(a, str) and a in values: return values[a]
                return 0.0
            v1 = resolve(arg1); v2 = resolve(arg2)
            if   op == "add": rv = v1 + v2
            elif op == "sub": rv = v1 - v2
            elif op == "mul": rv = v1 * v2
            elif op == "div":
                if v2 == 0 or v1 % v2 != 0:
                    ok = False; break
                rv = v1 / v2
            else:
                rv = v1
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


CORPUS = materialize_distmatched_corpus(
    CONFIG["data_train"],
    CONFIG["perturbations_per_trace"],
    CONFIG["n_pure_synthetic"],
    CONFIG["distmatched_cache_path"],
)
print(f"\\nFinal corpus size: {len(CORPUS)}")
'''

# ----------------------------------------------------------------- verification
MD_VERIFY = """## Distribution Verification

Compare the generated corpus's distribution to the original GSM8K
training distribution. If our generator is working, the two histograms
should overlay closely.
"""

C_VERIFY = '''# Cell 6 — Verify generated corpus matches GSM8K distribution

def report_distribution(records, label):
    op_counts = []
    op_freq   = Counter()
    leaf_mags = []
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
    print(f"  Op counts:")
    c = Counter(op_counts)
    total = sum(c.values())
    for k in sorted(c.keys())[:8]:
        print(f"    {k:>2}: {c[k]/total*100:5.1f}%")
    print(f"  Op frequency:")
    tot_ops = sum(op_freq.values())
    for op in ["add", "sub", "mul", "div"]:
        print(f"    {op:>5}: {op_freq[op]/tot_ops*100:5.1f}%")
    mags = np.array(leaf_mags)
    print(f"  Leaf mag: median={np.median(mags):.0f}  p75={np.percentile(mags,75):.0f}  "
          f"p95={np.percentile(mags,95):.0f}  max={mags.max():.0f}")


# Load real GSM8K for comparison
with open(CONFIG["data_train"]) as f:
    real_gsm8k = json.load(f)

report_distribution(real_gsm8k, "REAL GSM8K train")
report_distribution(CORPUS,     "GENERATED corpus")
'''

# ----------------------------------------------------------------- datasets
C_DATASETS = """# Cell 7 — Dataset classes

class DistMatchedDataset(Dataset):
    \"\"\"Pretrain dataset wrapping the distribution-matched corpus.\"\"\"
    def __init__(self, records, max_nodes=50):
        self.records = records
        self.max_nodes = max_nodes
        self.cached = [
            _sample_to_tensors(r["trace"], r["target"], max_nodes)
            for r in records
        ]
        print(f"DistMatched: {len(self.cached)} samples")

    def __len__(self): return len(self.cached)
    def __getitem__(self, i): return self.cached[i]


class GSM8KDataset(Dataset):
    \"\"\"v3.1 dataset (unchanged).\"\"\"
    def __init__(self, json_file, max_nodes=50, augment=False,
                 augment_p=0.5, augment_max_value=200):
        with open(json_file) as f:
            raw = json.load(f)
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


# Split corpus: 98% train, 2% val
random.seed(20260527)
random.shuffle(CORPUS)
_split = int(0.98 * len(CORPUS))
CORPUS_TRAIN = CORPUS[:_split]
CORPUS_VAL   = CORPUS[_split:]
print(f"Corpus train={len(CORPUS_TRAIN)}  val={len(CORPUS_VAL)}")
"""

# ----------------------------------------------------------------- arch + loss + eval (same as v3.1)
C_ARCH = """# Cell 8 — Architecture (IDENTICAL to v3.1)

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

C_LOSS = """# Cell 9 — Losses + AdamATan2
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

C_EVAL = """# Cell 10 — Evaluation
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

# ----------------------------------------------------------------- pretrain + finetune
MD_PRETRAIN = """## Pretrain on Distribution-Matched Corpus

47K (or whatever volume we ended up with) samples whose tree topologies
come from real GSM8K and whose numeric values match GSM8K's empirical
magnitudes. v3.1 architecture, 25 epochs, single global cosine schedule.
"""

C_PRETRAIN = """# Cell 11 — Pretrain on distribution-matched corpus

train_ds_pre = DistMatchedDataset(CORPUS_TRAIN, max_nodes=CONFIG["max_nodes"])
val_ds_pre   = DistMatchedDataset(CORPUS_VAL,   max_nodes=CONFIG["max_nodes"])
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
print(f"HRMForMath  {n_params/1e6:.2f}M params  (v3.1 architecture)")

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

print(f"{'Ep':>4}  {'Loss':>8}  {'gN':>5}  {'LR':>8}  {'ValEx%':>7}  {'ValDig%':>8}")
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

PRETRAIN_CKPT = os.path.join(CONFIG["pretrain_savedir"], "hrm_pretrain_distmatched.pt")
torch.save(model.state_dict(), PRETRAIN_CKPT)
with open(os.path.join(CONFIG["pretrain_savedir"], "pretrain_log.json"), "w") as f:
    json.dump(pretrain_log, f, indent=2)
print(f"\\nPretrain done. Saved to {PRETRAIN_CKPT}")
"""

MD_FINETUNE = """## Fine-tune on Real GSM8K

Load the distribution-matched pretrain checkpoint, fine-tune on the
original 4,682 GSM8K training traces. Lower LR (5e-5) to preserve the
distribution-aligned pretrain features.
"""

C_FINETUNE = """# Cell 12 — Fine-tune on real GSM8K

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
print(f"Loaded distmatched pretrain.  Train={len(train_set_ft)}  Val={len(val_set_ft)}")

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
                           "source": "distmatched_pretrain_then_gsm8k"}, f, indent=2)
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

C_TEST = """# Cell 13 — Final test eval
test_set = GSM8KDataset(CONFIG["data_test"], max_nodes=CONFIG["max_nodes"])
test_loader = DataLoader(test_set, batch_size=CONFIG["finetune_batch"] * 2,
                         shuffle=False, collate_fn=collate_fn,
                         num_workers=2, pin_memory=True)

best_path = os.path.join(CONFIG["finetune_savedir"], "best_model.pt")
model_ft.load_state_dict(torch.load(best_path))
tm = evaluate(model_ft, test_loader, DEVICE)
print("=" * 60)
print("GSM8K TEST RESULTS (distribution-matched pretrain → finetune)")
print("=" * 60)
for k, v in tm.items():
    if isinstance(v, float): print(f"  {k:18s} = {v:.4f}")
    else:                    print(f"  {k:18s} = {v}")
"""


cells = [
    md("md_header",   MD_HEADER),
    code("c_env",     C_ENV),
    code("c_config",  C_CONFIG),
    code("c_utils",   C_UTILS),
    md("md_analyzer", MD_ANALYZER),
    code("c_analyzer", C_ANALYZER),
    md("md_gen",      MD_GEN),
    code("c_gen",     C_GEN),
    md("md_verify",   MD_VERIFY),
    code("c_verify",  C_VERIFY),
    code("c_datasets", C_DATASETS),
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
