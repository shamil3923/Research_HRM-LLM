"""Builder for notebook_pretrain_dmath_hf.ipynb.

Generates a Jupyter notebook that:
  1. Loads the real HuggingFace `deepmind/math_dataset` (arithmetic modules).
  2. Converts (question, answer) → HRM trace via deterministic AST-based parsing.
  3. Pre-trains HRM-v3.1 on the converted corpus (curriculum by op-count).
  4. Fine-tunes on GSM8K from the pre-trained checkpoint.

Run:  python build_dmath_hf_notebook.py
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "notebook_pretrain_dmath_hf.ipynb")


def md(cell_id, text):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(cell_id, src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


# ---------------------------------------------------------------- header
MD_HEADER = """# HRM Pre-training on DeepMind Mathematics Dataset (HF, parsed) → Fine-tune on GSM8K

**Goal.** Pre-train HRM v3.1 on the *real* `deepmind/math_dataset`
arithmetic modules — converted in-process from `(question, answer)` strings
into HRM symbolic traces — then fine-tune on GSM8K.

**Why this notebook exists.** The existing `notebook_pretrain_dmath.ipynb`
uses synthetic problems. This notebook instead consumes the actual HF
dataset. Because every arithmetic module in `deepmind/math_dataset`
follows a small set of *fixed string templates*, we can deterministically
extract the arithmetic expression (regex + a handful of word→operator
substitutions), then `ast.parse` it into a Python expression tree, then
walk that tree in post-order to emit `parse_graph`-compatible steps.
**No LLM parsing**, no fuzzy heuristics — just templates and AST.

**Modules covered** (integer arithmetic only — decimals/surds are skipped):
- `arithmetic__add_or_sub`
- `arithmetic__add_sub_multiple`
- `arithmetic__mul`
- `arithmetic__mul_div_multiple`
- `arithmetic__mixed`

**Architecture is IDENTICAL to v3.1** — Graph-Aware Bridge (Dense GAT) +
HRM Core (H/L, ACT, Q-head) + Digit Head, RMSNorm, SwiGLU, Adam-atan2.
The existing `hrm_gsm8k_v3_1/best_model.pt` is left untouched; this
notebook writes new checkpoints under separate directories.

**Curriculum.** After parsing, every example carries a `num_ops` count.
We bucket by op-count (1, 2, 3, 4+) and train through 4 stages in that
order. This keeps the training signal smooth even though the HF modules
themselves are heterogeneous.

**Outputs.**
- `hrm_pretrain_dmath_hf.pt`   (after pre-train)
- `hrm_finetuned_gsm8k_hf.pt`  (after fine-tune)
"""

# ---------------------------------------------------------------- cell 1
C_ENV = """# Cell 1 — Environment
import sys, os, json, time, math, random, copy, re
import ast as _ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.version.cuda)
print("GPUs   :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1e9:.1f} GB")

assert torch.cuda.is_available(), "GPU required"
DEVICE = torch.device("cuda")
"""

# ---------------------------------------------------------------- cell 2
C_CONFIG = """# Cell 2 — Configuration
DATA_ROOT = "/kaggle/input/datasets/shamilmrm/llm-parsed-data"

CONFIG = dict(
    # --- Model (IDENTICAL to v3.1) ---
    dmodel     = 256,
    nheads     = 8,
    Hcycles    = 3,
    Lcycles    = 4,
    Hlayers    = 4,
    Llayers    = 4,
    max_nodes  = 50,

    # --- HF dataset materialization ---
    hf_dataset_name = "deepmind/math_dataset",
    hf_modules = [
        "arithmetic__add_or_sub",
        "arithmetic__add_sub_multiple",
        "arithmetic__mul",
        "arithmetic__mul_div_multiple",
        "arithmetic__mixed",
    ],
    hf_split            = "train",
    hf_max_per_module   = 60_000,   # cap per module before filtering
    hf_streaming        = True,     # stream, do not download full 2M
    hf_cache_dir        = "/kaggle/working/hf_cache",
    parsed_cache_path   = "/kaggle/working/checkpoints/hrm_pretrain_dmath_hf/parsed_traces.json",

    # --- Pre-training ---
    pretrain_savedir    = "/kaggle/working/checkpoints/hrm_pretrain_dmath_hf",
    pretrain_batch_size = 128,
    pretrain_peak_lr    = 5e-4,
    pretrain_lr_floor   = 1e-5,
    pretrain_max_value  = 200,      # passed through to perturb_trace_constants on aug
    pretrain_augment_p  = 0.2,
    pretrain_val_frac   = 0.02,

    # Curriculum: each stage filters parsed pool by num_ops_min..num_ops_max.
    pretrain_curriculum = [
        {"label": "1op",  "ops_min": 1, "ops_max": 1, "epochs": 3},
        {"label": "2op",  "ops_min": 2, "ops_max": 2, "epochs": 4},
        {"label": "3op",  "ops_min": 3, "ops_max": 3, "epochs": 5},
        {"label": "4op+", "ops_min": 4, "ops_max": 8, "epochs": 6},
    ],

    # --- Fine-tuning on GSM8K ---
    finetune_savedir   = "/kaggle/working/checkpoints/hrm_finetuned_gsm8k_hf",
    data_train         = f"{DATA_ROOT}/gsm8k_train_split.json",
    data_val           = f"{DATA_ROOT}/gsm8k_val_split.json",
    data_test          = f"{DATA_ROOT}/gsm8k_test_clean.json",
    finetune_epochs    = 250,
    finetune_batch     = 128,
    finetune_peak_lr   = 1e-4,
    finetune_lr_floor  = 5e-6,
    finetune_eval_every= 10,
    augment_p          = 0.3,
    augment_max_value  = 200,

    # --- ACT (same as v3.1) ---
    act_max_steps = 4,
    act_min_steps = 2,
    act_explore_p = 0.1,

    # --- Loss weights ---
    aux_loss_weight = 1.5,
    q_loss_weight   = 0.5,

    # --- Adam-atan2 ---
    optim_a     = 1.27,
    optim_b     = 1.0,
    optim_betas = (0.9, 0.95),
    optim_wd    = 0.01,
)
os.makedirs(CONFIG["pretrain_savedir"], exist_ok=True)
os.makedirs(CONFIG["finetune_savedir"], exist_ok=True)
os.makedirs(CONFIG["hf_cache_dir"], exist_ok=True)
print("Config OK.")
for k in ["dmodel", "Hcycles", "Lcycles", "hf_modules",
         "hf_max_per_module", "pretrain_curriculum",
         "pretrain_peak_lr", "finetune_peak_lr", "finetune_epochs"]:
    print(f"  {k:32s} = {CONFIG[k]}")
"""

# ---------------------------------------------------------------- cell 3
C_UTILS = """# Cell 3 — Shared utilities (IDENTICAL to v3.1 — do not modify)
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


def perturb_trace_constants(trace, max_value=200, rng=None):
    if rng is None: rng = random
    new_trace = copy.deepcopy(trace)
    steps = new_trace.get("steps", [])
    if not steps: return None, None
    vv = {}
    def maybe_replace(arg):
        if isinstance(arg, (int, float)):
            return float(rng.randint(1, max_value))
        if isinstance(arg, str):
            try: float(arg); return float(rng.randint(1, max_value))
            except ValueError: return arg
        return arg
    for s in steps:
        op = s.get("op", "const")
        s["arg1"] = maybe_replace(s.get("arg1", 0))
        s["arg2"] = maybe_replace(s.get("arg2", 0))
        def resolve(a):
            if isinstance(a, (int, float)): return float(a)
            if isinstance(a, str) and a in vv: return vv[a]
            try: return float(a)
            except Exception: return 0.0
        v1 = resolve(s["arg1"]); v2 = resolve(s["arg2"])
        if   op == "add":               rv = v1 + v2
        elif op == "sub":               rv = v1 - v2
        elif op == "mul":               rv = v1 * v2
        elif op == "div" and v2 != 0:   rv = v1 / v2
        else:                            rv = v1
        if not np.isfinite(rv) or abs(rv) > _AUG_MAX_ABS: return None, None
        s["result_value"] = rv
        rk = s.get("result", "")
        if rk: vv[rk] = rv
    fa_var = new_trace.get("final_answer", "")
    if fa_var not in vv: return None, None
    fa = vv[fa_var]
    if not np.isfinite(fa) or abs(fa) > _AUG_MAX_ABS: return None, None
    return new_trace, int(round(fa))


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
        nvals.append([
            float(np.sign(fa_val) * np.log1p(abs(fa_val))), 1.0, 0.0, 0.0,
        ])
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
    try:
        tgt_int = int(round(float(target)))
    except (OverflowError, ValueError):
        tgt_int = 0
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

# ---------------------------------------------------------------- cell 4 (NEW: HF converter)
MD_CONVERTER = """## HF `deepmind/math_dataset` → HRM trace converter

The arithmetic modules in DeepMind's dataset are *parametrically generated*
from a small set of string templates. So extraction is deterministic:

1. **Template strip** — pattern-match the question's outer wrapper
   (`"What is X?"`, `"Calculate X."`, `"Total of A and B."`, etc.) to
   recover an arithmetic expression `X`.
2. **Word → operator** — replace `" plus "`, `" minus "`, `" times "`,
   `" multiplied by "`, `" divided by "` with `+ - * / /`.
3. **AST parse** — `ast.parse(expr, mode="eval")` gives a clean
   binary-operation tree (Python's grammar already left-associates
   `a + b + c + d` into nested `BinOp`s — perfect for our binary
   trace format).
4. **Post-order walk** — emit one step per `BinOp`. `UnaryOp(-)` on a
   constant is inlined as a negative literal; on a non-constant it
   becomes `sub(0, x)`.
5. **Integer guard** — reject any example whose computed answer or
   any intermediate value isn't an integer in `[-_AUG_MAX_ABS, +_AUG_MAX_ABS]`,
   or whose final value doesn't match the dataset-provided answer.

Whatever survives all five gates is a *verified* trace: same schema as
v3.1, same digit head target, ready for `parse_graph`.
"""

C_CONVERTER = '''# Cell 4 — HF dataset converter (template + AST → HRM trace)

# Phrases we recognize as wrappers around an arithmetic expression.
# Each entry is (regex, transform_fn). The regex match groups are passed
# into the transform_fn which returns a Python-evaluable expression string.

_WORD_OPS = [
    (re.compile(r"\\bplus\\b",           re.I), "+"),
    (re.compile(r"\\bminus\\b",          re.I), "-"),
    (re.compile(r"\\btimes\\b",          re.I), "*"),
    (re.compile(r"\\bmultiplied by\\b",  re.I), "*"),
    (re.compile(r"\\bdivided by\\b",     re.I), "/"),
]

_WRAPPERS = [
    # "Subtract A from B." → B - A
    (re.compile(r"^subtract\\s+(.+?)\\s+from\\s+(.+?)[.?]?$", re.I),
     lambda m: f"({m.group(2)}) - ({m.group(1)})"),
    # "Add together A and B and C." → A + B + C
    (re.compile(r"^add together\\s+(.+?)[.?]?$", re.I),
     lambda m: " + ".join(_split_and(m.group(1)))),
    # "Put together A and B." → A + B
    (re.compile(r"^put together\\s+(.+?)[.?]?$", re.I),
     lambda m: " + ".join(_split_and(m.group(1)))),
    # "Total of A and B and C." → A + B + C
    (re.compile(r"^total of\\s+(.+?)[.?]?$", re.I),
     lambda m: " + ".join(_split_and(m.group(1)))),
    # "Sum A and B." → A + B
    (re.compile(r"^sum\\s+(.+?)[.?]?$", re.I),
     lambda m: " + ".join(_split_and(m.group(1)))),
    # "Product of A and B." → A * B
    (re.compile(r"^product of\\s+(.+?)[.?]?$", re.I),
     lambda m: " * ".join(_split_and(m.group(1)))),
    # "Multiply A and B." → A * B
    (re.compile(r"^multiply\\s+(.+?)[.?]?$", re.I),
     lambda m: " * ".join(_split_and(m.group(1)))),
    # "Divide A by B." → A / B
    (re.compile(r"^divide\\s+(.+?)\\s+by\\s+(.+?)[.?]?$", re.I),
     lambda m: f"({m.group(1)}) / ({m.group(2)})"),
    # "What is the product of A and B?" → A * B
    (re.compile(r"^what is the product of\\s+(.+?)[.?]?$", re.I),
     lambda m: " * ".join(_split_and(m.group(1)))),
    # "What is the result of A multiplied by B?" → A * B  (handled later by word ops)
    (re.compile(r"^what is the result of\\s+(.+?)[.?]?$", re.I),
     lambda m: m.group(1)),
    # "What is X?" / "What is X."
    (re.compile(r"^what is\\s+(.+?)[.?]?$", re.I),
     lambda m: m.group(1)),
    # "Calculate X." / "Work out X." / "Evaluate X."
    (re.compile(r"^(?:calculate|work out|evaluate)\\s+(.+?)[.?]?$", re.I),
     lambda m: m.group(1)),
    # bare expression
    (re.compile(r"^(.+?)[.?]?$"),
     lambda m: m.group(1)),
]


def _split_and(s):
    """Split a string like 'A and B and C' into ['A', 'B', 'C'].
    Respects parenthesis nesting so '(A and B) and C' is not split inside."""
    parts, buf, depth = [], "", 0
    i = 0
    s = s.strip()
    while i < len(s):
        ch = s[i]
        if ch == "(": depth += 1; buf += ch
        elif ch == ")": depth -= 1; buf += ch
        elif depth == 0 and s[i:i+5].lower() == " and ":
            parts.append(buf.strip()); buf = ""; i += 5; continue
        else:
            buf += ch
        i += 1
    if buf.strip(): parts.append(buf.strip())
    return [f"({p})" for p in parts]


def _extract_expr(question):
    s = question.strip()
    if s.endswith("."): s = s[:-1]
    if s.endswith("?"): s = s[:-1]
    s = s.strip()
    expr = None
    for pat, fn in _WRAPPERS:
        m = pat.match(s)
        if m:
            try:
                expr = fn(m)
            except Exception:
                expr = None
            if expr: break
    if expr is None:
        expr = s
    # word-ops → symbols
    for pat, sym in _WORD_OPS:
        expr = pat.sub(sym, expr)
    return expr.strip()


# Integer-only sanity: reject any decimal point appearing in numeric tokens.
_HAS_DECIMAL = re.compile(r"\\d\\.\\d")
# Allowed character set after extraction (numbers, operators, parens, spaces).
_ALLOWED = re.compile(r"^[\\d\\s+\\-*/().]+$")


_BINOP_NAME = {
    _ast.Add: "add", _ast.Sub: "sub",
    _ast.Mult: "mul", _ast.Div: "div", _ast.FloorDiv: "div",
}


def ast_to_trace(expr_str, max_steps=40):
    """Parse an arithmetic expression string and emit an HRM trace.
    Returns (trace_dict, target_int) or (None, None) on any failure
    (non-integer intermediate, overflow, unsupported node, etc.)."""
    try:
        tree = _ast.parse(expr_str, mode="eval").body
    except (SyntaxError, ValueError):
        return None, None

    steps = []
    counter = [0]

    def walk(node):
        # Returns (arg_repr, value):
        #   arg_repr is either a float literal or a string var name
        #   value is the integer numeric value
        if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
            v = node.value
            if isinstance(v, float) and v != int(v):
                return None, None
            iv = int(v)
            if abs(iv) > _AUG_MAX_ABS: return None, None
            return float(iv), iv

        if isinstance(node, _ast.UnaryOp):
            if isinstance(node.op, _ast.UAdd):
                return walk(node.operand)
            if isinstance(node.op, _ast.USub):
                inner_repr, inner_val = walk(node.operand)
                if inner_val is None: return None, None
                # Inline negative constants.
                if isinstance(inner_repr, float):
                    iv = -int(inner_val)
                    if abs(iv) > _AUG_MAX_ABS: return None, None
                    return float(iv), iv
                # Otherwise emit  sub(0, x)
                v = 0 - inner_val
                if abs(v) > _AUG_MAX_ABS: return None, None
                if len(steps) >= max_steps: return None, None
                rk = f"v{counter[0]}"; counter[0] += 1
                steps.append({"op": "sub", "arg1": 0.0, "arg2": inner_repr,
                              "result": rk, "result_value": float(v)})
                return rk, v
            return None, None

        if isinstance(node, _ast.BinOp):
            op_name = _BINOP_NAME.get(type(node.op))
            if op_name is None: return None, None
            la, lv = walk(node.left)
            if lv is None: return None, None
            ra, rv = walk(node.right)
            if rv is None: return None, None
            if   op_name == "add": res = lv + rv
            elif op_name == "sub": res = lv - rv
            elif op_name == "mul": res = lv * rv
            else:
                if rv == 0: return None, None
                if lv % rv != 0: return None, None
                res = lv // rv
            if abs(res) > _AUG_MAX_ABS: return None, None
            if len(steps) >= max_steps: return None, None
            rk = f"v{counter[0]}"; counter[0] += 1
            steps.append({"op": op_name, "arg1": la, "arg2": ra,
                          "result": rk, "result_value": float(res)})
            return rk, res

        return None, None

    root_repr, root_val = walk(tree)
    if root_val is None or not steps:
        return None, None
    # Root must be the last-emitted variable (i.e. the final BinOp).
    if not isinstance(root_repr, str) or root_repr != steps[-1]["result"]:
        return None, None
    trace = {"steps": steps, "final_answer": steps[-1]["result"]}
    return trace, int(root_val)


def convert_dmath_example(question, answer):
    """Convert one HF (question, answer) into (trace, target_int).
    Returns (None, None) if any sanity check fails."""
    expr = _extract_expr(question)
    if expr is None: return None, None
    if _HAS_DECIMAL.search(expr): return None, None
    # Cheap pre-filter: only operators/digits/whitespace/parens allowed.
    if not _ALLOWED.match(expr.replace("--", "+ ")):
        return None, None
    try:
        ans_int = int(str(answer).strip())
    except (ValueError, AttributeError, TypeError):
        return None, None
    if abs(ans_int) > _AUG_MAX_ABS: return None, None
    trace, computed = ast_to_trace(expr)
    if trace is None or computed != ans_int:
        return None, None
    return trace, ans_int


# Smoke tests — verify the converter on a few representative templates.
_SMOKE = [
    ("What is -54 - -125?",          "71"),
    ("Calculate -8 + 25 - -29.",     "46"),
    ("Subtract 6 from 18.",          "12"),
    ("Total of -3 and 7 and 4.",     "8"),
    ("What is (2 + 3)*4?",           "20"),
    ("Calculate 2*(-1)*(2/(-2)).",   "2"),
    ("What is the product of 3 and -2?", "-6"),
]
print(f"{'Question':<45} {'Ans':>5}  {'Parsed':>7}  {'#Steps':>7}")
for q, a in _SMOKE:
    t, p = convert_dmath_example(q, a)
    n = len(t["steps"]) if t else 0
    print(f"{q[:45]:<45} {a:>5}  {str(p):>7}  {n:>7}")
'''

# ---------------------------------------------------------------- cell 5 (NEW: load HF + bucket)
MD_LOAD = """## Materialize the HF dataset

Stream each arithmetic module from HuggingFace (no full download), run
each `(question, answer)` through the converter, drop everything that
doesn't yield a clean integer trace, and bucket the survivors by `num_ops`.
We cache the result to disk on first run so re-runs are instant.
"""

C_LOAD = """# Cell 5 — Load + convert + cache the HF arithmetic corpus

def materialize_hf_corpus():
    if os.path.exists(CONFIG["parsed_cache_path"]):
        print(f"Loading cached parsed corpus from {CONFIG['parsed_cache_path']}")
        with open(CONFIG["parsed_cache_path"]) as f:
            return json.load(f)

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Run: pip install datasets")

    all_records = []
    per_module_stats = {}
    for module in CONFIG["hf_modules"]:
        print(f"\\n[{module}] streaming...")
        ds = load_dataset(
            CONFIG["hf_dataset_name"],
            module,
            split=CONFIG["hf_split"],
            streaming=CONFIG["hf_streaming"],
            cache_dir=CONFIG["hf_cache_dir"],
            trust_remote_code=True,
        )
        seen, kept = 0, 0
        for ex in ds:
            if seen >= CONFIG["hf_max_per_module"]:
                break
            seen += 1
            q = ex.get("question", "")
            a = ex.get("answer", "")
            # HF returns bytes for some splits; decode if needed.
            if isinstance(q, bytes): q = q.decode("utf-8", errors="ignore")
            if isinstance(a, bytes): a = a.decode("utf-8", errors="ignore")
            trace, tgt = convert_dmath_example(q, a)
            if trace is None: continue
            all_records.append({
                "module":   module,
                "num_ops":  len(trace["steps"]),
                "trace":    trace,
                "target":   tgt,
            })
            kept += 1
        per_module_stats[module] = {"seen": seen, "kept": kept,
                                    "rate": kept / max(1, seen)}
        print(f"  seen={seen}  kept={kept}  rate={kept/max(1,seen):.2%}")

    print("\\nOverall per-module:")
    for m, s in per_module_stats.items():
        print(f"  {m:<36s}  kept {s['kept']:>6d} / {s['seen']:>6d}  ({s['rate']:.2%})")

    os.makedirs(os.path.dirname(CONFIG["parsed_cache_path"]), exist_ok=True)
    with open(CONFIG["parsed_cache_path"], "w") as f:
        json.dump(all_records, f)
    print(f"\\nCached {len(all_records)} records → {CONFIG['parsed_cache_path']}")
    return all_records


PARSED = materialize_hf_corpus()
print(f"\\nTotal parsed records: {len(PARSED)}")

# Bucket by num_ops for curriculum.
from collections import Counter
op_counter = Counter(r["num_ops"] for r in PARSED)
print("\\nnum_ops distribution (top 10):")
for k in sorted(op_counter)[:10]:
    print(f"  ops={k:>2d}  count={op_counter[k]:>7d}")
"""

# ---------------------------------------------------------------- cell 6 (NEW: dataset classes)
C_DATASETS = """# Cell 6 — Datasets: HF-parsed DMath + GSM8K (v3.1)

class DMathHFDataset(Dataset):
    \"\"\"Wraps a list of parsed HF records, optionally filtered by op-count.
    Augmentation re-uses v3.1's perturb_trace_constants to vary numeric
    literals while preserving structure — gives effectively-infinite data
    without re-streaming the HF dataset.\"\"\"
    def __init__(self, records, ops_min=1, ops_max=99,
                 max_nodes=50, augment=False, augment_p=0.2,
                 augment_max_value=200):
        self.records = [r for r in records
                        if ops_min <= r["num_ops"] <= ops_max]
        self.max_nodes = max_nodes
        self.augment = augment
        self.augment_p = augment_p
        self.augment_max_value = augment_max_value
        # Pre-cache base tensors for speed.
        self.cached = [
            _sample_to_tensors(r["trace"], r["target"], max_nodes)
            for r in self.records
        ]
        print(f"DMathHF: {len(self.cached)} samples  "
              f"(ops {ops_min}..{ops_max}, aug={augment} p={augment_p})")

    def __len__(self): return len(self.cached)

    def __getitem__(self, i):
        if self.augment and random.random() < self.augment_p:
            r = self.records[i]
            new_t, new_tgt = perturb_trace_constants(
                r["trace"], max_value=self.augment_max_value,
            )
            if new_t is not None and new_tgt is not None:
                return _sample_to_tensors(new_t, new_tgt, self.max_nodes)
        return self.cached[i]


class GSM8KDataset(Dataset):
    \"\"\"v3.1 dataset — unchanged.\"\"\"
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
        msg = f"GSM8K: {len(self.cached)} samples (max_nodes={max_nodes})"
        if augment: msg += f"  [aug p={augment_p}, max_value={augment_max_value}]"
        if skipped: msg += f"  [skipped {skipped} bad final_answer]"
        print(msg)

    def __len__(self): return len(self.cached)

    def __getitem__(self, i):
        if self.augment and random.random() < self.augment_p:
            trace, _ = self.raw[i]
            new_t, new_tgt = perturb_trace_constants(trace,
                                                     max_value=self.augment_max_value)
            if new_t is not None and new_tgt is not None:
                return _sample_to_tensors(new_t, new_tgt, self.max_nodes)
        return self.cached[i]


# Hold out a tiny val split from the parsed pool (random by index).
_val_n = max(500, int(CONFIG["pretrain_val_frac"] * len(PARSED)))
_rng_split = random.Random(20251114)
_idx = list(range(len(PARSED)))
_rng_split.shuffle(_idx)
_val_idx = set(_idx[:_val_n])
PRETRAIN_TRAIN_RECS = [r for i, r in enumerate(PARSED) if i not in _val_idx]
PRETRAIN_VAL_RECS   = [r for i, r in enumerate(PARSED) if i in _val_idx]
print(f"Pretrain split: train={len(PRETRAIN_TRAIN_RECS)}  val={len(PRETRAIN_VAL_RECS)}")
"""

# ---------------------------------------------------------------- cell 7 (Arch — identical)
C_ARCH = """# Cell 7 — Architecture (IDENTICAL to v3.1 — do not modify)

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

print("Architecture defined (identical to v3.1).")
"""

# ---------------------------------------------------------------- cell 8 (Loss/Opt — identical)
C_LOSS = """# Cell 8 — Losses + AdamATan2 (IDENTICAL to v3.1)
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
                if grad.is_sparse:
                    raise RuntimeError("AdamATan2 does not support sparse gradients")
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

# ---------------------------------------------------------------- cell 9 (Eval — identical)
C_EVAL = """# Cell 9 — Evaluation (IDENTICAL to v3.1)
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

# ---------------------------------------------------------------- cell 10 (pretrain)
MD_PRETRAIN = """## Pre-training (op-count curriculum)

We train through 4 stages — `1op → 2op → 3op → 4op+` — using the HF-parsed
records bucketed by `num_ops`. Model + optimizer + LR scheduler persist
across stages; only the data filter changes. Val is held out once from
the full parsed pool and re-used per stage (filtered to match the stage).
"""

C_PRETRAIN = """# Cell 10 — Pre-training loop with op-count curriculum

model = HRMForMath(vsz=len(OP_VOCAB), d=CONFIG["dmodel"], heads=CONFIG["nheads"],
                   Hc=CONFIG["Hcycles"], Lc=CONFIG["Lcycles"],
                   Hl=CONFIG["Hlayers"], Ll=CONFIG["Llayers"],
                   slen=CONFIG["max_nodes"]).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"HRMForMath  {n_params/1e6:.2f}M params")

opt = AdamATan2(model.parameters(), lr=CONFIG["pretrain_peak_lr"],
                betas=CONFIG["optim_betas"], weight_decay=CONFIG["optim_wd"],
                a=CONFIG["optim_a"], b=CONFIG["optim_b"])

# Build per-stage datasets up-front so we know `total_steps` for the schedule.
stage_loaders = []
for stage_idx, stage in enumerate(CONFIG["pretrain_curriculum"]):
    train_ds = DMathHFDataset(PRETRAIN_TRAIN_RECS,
                              ops_min=stage["ops_min"], ops_max=stage["ops_max"],
                              max_nodes=CONFIG["max_nodes"],
                              augment=True, augment_p=CONFIG["pretrain_augment_p"],
                              augment_max_value=CONFIG["pretrain_max_value"])
    val_ds   = DMathHFDataset(PRETRAIN_VAL_RECS,
                              ops_min=stage["ops_min"], ops_max=stage["ops_max"],
                              max_nodes=CONFIG["max_nodes"],
                              augment=False)
    if len(train_ds) == 0:
        print(f"  [warn] stage {stage_idx+1} ({stage['label']}) has 0 train samples; skipping")
        stage_loaders.append((None, None, 0))
        continue
    tl = DataLoader(train_ds, batch_size=CONFIG["pretrain_batch_size"],
                    shuffle=True, collate_fn=collate_fn,
                    num_workers=2, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=CONFIG["pretrain_batch_size"] * 2,
                    shuffle=False, collate_fn=collate_fn,
                    num_workers=2, pin_memory=True) if len(val_ds) else None
    stage_loaders.append((tl, vl, len(tl)))

total_steps = max(1, sum(steps * stg["epochs"]
                         for (_, _, steps), stg in zip(stage_loaders,
                                                       CONFIG["pretrain_curriculum"])))
warmup_steps = max(1, int(0.05 * total_steps))
floor_ratio  = CONFIG["pretrain_lr_floor"] / CONFIG["pretrain_peak_lr"]

def _lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor_ratio + (1.0 - floor_ratio) * cos

sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
scaler = torch.amp.GradScaler("cuda")

ACT_MIN = CONFIG["act_min_steps"]
log_rows = []
global_ep = 0

print("=" * 100)
print(f"{'Stg':>3}  {'Bkt':>5}  {'Ep':>4}  {'Loss':>8}  {'gN':>5}  {'LR':>8}  "
      f"{'ValEx%':>7}  {'ValDig%':>8}  {'AvgHlt':>6}")
print("=" * 100)

for stage_idx, (stage, (train_loader, val_loader, n_steps)) in enumerate(
        zip(CONFIG["pretrain_curriculum"], stage_loaders)):
    if train_loader is None:
        continue
    for ep_in_stage in range(stage["epochs"]):
        global_ep += 1
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
                        nqh, nqc = seg_outputs[s + 1][1], seg_outputs[s + 1][2]
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
        m = (evaluate(model, val_loader, DEVICE)
             if val_loader is not None else
             {"exact_acc": 0.0, "digit_acc": 0.0, "mean_halt_steps": 0.0})
        print(f"{stage_idx+1:>3}  {stage['label']:>5}  {global_ep:>4}  {al:>8.4f}  "
              f"{ag:>5.2f}  {lr:>8.1e}  {m['exact_acc']*100:>7.2f}  "
              f"{m['digit_acc']*100:>8.2f}  {m['mean_halt_steps']:>6.2f}")
        log_rows.append({"stage": stage_idx + 1, "bucket": stage["label"],
                         "epoch": global_ep, "loss": al,
                         "val_exact": m["exact_acc"], "val_digit": m["digit_acc"],
                         "mean_halt": m["mean_halt_steps"]})

    ck_path = os.path.join(CONFIG["pretrain_savedir"],
                           f"stage{stage_idx+1}_{stage['label']}.pt")
    torch.save(model.state_dict(), ck_path)
    print(f"  [saved] {ck_path}")

print("=" * 100)
PRETRAIN_CKPT = os.path.join(CONFIG["pretrain_savedir"], "hrm_pretrain_dmath_hf.pt")
torch.save(model.state_dict(), PRETRAIN_CKPT)
with open(os.path.join(CONFIG["pretrain_savedir"], "pretrain_log.json"), "w") as f:
    json.dump(log_rows, f, indent=2)
print(f"Pre-training done. Saved to {PRETRAIN_CKPT}")
"""

# ---------------------------------------------------------------- cell 11 (finetune)
MD_FINETUNE = """## Fine-tune on GSM8K

Load pretrained weights, fresh optimizer with **lower peak LR** (`1e-4`,
3× smaller than scratch) to preserve pretrained features, then run the
v3.1 training loop on GSM8K. The existing `hrm_gsm8k_v3_1/best_model.pt`
is left untouched — best model is written under `hrm_finetuned_gsm8k_hf/`.
"""

C_FINETUNE = """# Cell 11 — Fine-tune on GSM8K from pretrained checkpoint

train_set = GSM8KDataset(CONFIG["data_train"], max_nodes=CONFIG["max_nodes"],
                         augment=True, augment_p=CONFIG["augment_p"],
                         augment_max_value=CONFIG["augment_max_value"])
val_set   = GSM8KDataset(CONFIG["data_val"],   max_nodes=CONFIG["max_nodes"])
train_loader = DataLoader(train_set, batch_size=CONFIG["finetune_batch"],
                          shuffle=True, collate_fn=collate_fn,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=CONFIG["finetune_batch"] * 2,
                          shuffle=False, collate_fn=collate_fn,
                          num_workers=2, pin_memory=True)
print(f"Train: {len(train_set)}   Val: {len(val_set)}")

model_ft = HRMForMath(vsz=len(OP_VOCAB), d=CONFIG["dmodel"], heads=CONFIG["nheads"],
                      Hc=CONFIG["Hcycles"], Lc=CONFIG["Lcycles"],
                      Hl=CONFIG["Hlayers"], Ll=CONFIG["Llayers"],
                      slen=CONFIG["max_nodes"]).to(DEVICE)
missing, unexpected = model_ft.load_state_dict(torch.load(PRETRAIN_CKPT), strict=False)
print(f"Loaded pretrained checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

opt_ft = AdamATan2(model_ft.parameters(), lr=CONFIG["finetune_peak_lr"],
                   betas=CONFIG["optim_betas"], weight_decay=CONFIG["optim_wd"],
                   a=CONFIG["optim_a"], b=CONFIG["optim_b"])

total_steps_ft  = max(1, CONFIG["finetune_epochs"] * len(train_loader))
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

skipped_steps = consec_bad = 0
MAX_CONSEC_BAD = 50
run_aborted = False
best_acc = 0.0
ft_log = []
ACT_MIN = CONFIG["act_min_steps"]

print("=" * 100)
print(f"{'Ep':>4}  {'Loss':>8}  {'gN':>5}  {'LR':>8}  {'Ex%':>6}  {'Dig%':>7}  "
      f"{'Near%':>6}  {'NoOut':>5}  {'AvgHlt':>6}")
print("=" * 100)

for ep in range(CONFIG["finetune_epochs"]):
    model_ft.train()
    el = eg = stp = 0
    for batch in train_loader:
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
                    nqh, nqc = seg_outputs[s + 1][1], seg_outputs[s + 1][2]
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
            skipped_steps += 1; consec_bad += 1
            opt_ft.zero_grad(set_to_none=True); scaler_ft.update(); sch_ft.step()
            if consec_bad >= MAX_CONSEC_BAD:
                run_aborted = True; break
            continue
        consec_bad = 0
        scaler_ft.step(opt_ft); scaler_ft.update(); sch_ft.step()
        el += total_loss.item(); eg += gn.item(); stp += 1
    if run_aborted:
        print(f"ABORT at ep {ep+1}: {consec_bad} non-finite grad steps.")
        break

    al = el / max(stp, 1); ag = eg / max(stp, 1); lr = sch_ft.get_last_lr()[0]
    do_eval = ((ep + 1) % CONFIG["finetune_eval_every"] == 0) or ep == 0
    if do_eval:
        m = evaluate(model_ft, val_loader, DEVICE)
        ex, da, ne = m["exact_acc"], m["digit_acc"], m["near_acc"]
        imp = ex > best_acc
        if imp:
            best_acc = ex
            torch.save(model_ft.state_dict(),
                       os.path.join(CONFIG["finetune_savedir"], "best_model.pt"))
            with open(os.path.join(CONFIG["finetune_savedir"], "best_meta.json"), "w") as f:
                json.dump({"epoch": ep + 1, "val_exact_acc": ex,
                           "val_digit_acc": da, "val_near_acc": ne,
                           "mean_halt_steps": m["mean_halt_steps"],
                           "no_output": m["no_output"],
                           "source": "pretrained_dmath_hf_then_gsm8k"}, f, indent=2)
        mk = " *" if imp else ""
        print(f"{ep+1:>4}  {al:>8.4f}  {ag:>5.2f}  {lr:>8.1e}  "
              f"{ex*100:>6.2f}  {da*100:>7.1f}  {ne*100:>6.2f}  "
              f"{m['no_output']:>5}  {m['mean_halt_steps']:>6.2f}{mk}")
        ft_log.append({"epoch": ep + 1, "loss": al,
                       "exact_acc": ex, "digit_acc": da, "near_acc": ne,
                       "mean_halt": m["mean_halt_steps"], "no_output": m["no_output"]})
    elif (ep + 1) % 5 == 0:
        print(f"{ep+1:>4}  {al:>8.4f}  {ag:>5.2f}  {lr:>8.1e}")

print("=" * 100)
print(f"Fine-tune done. Best VAL exact = {best_acc*100:.2f}%   "
      f"(skipped {skipped_steps} non-finite steps)")
with open(os.path.join(CONFIG["finetune_savedir"], "finetune_log.json"), "w") as f:
    json.dump(ft_log, f, indent=2)
"""

# ---------------------------------------------------------------- cell 12 (test)
C_TEST = """# Cell 12 — Final test-set evaluation
test_set    = GSM8KDataset(CONFIG["data_test"], max_nodes=CONFIG["max_nodes"])
test_loader = DataLoader(test_set, batch_size=CONFIG["finetune_batch"] * 2,
                         shuffle=False, collate_fn=collate_fn,
                         num_workers=2, pin_memory=True)

best_path = os.path.join(CONFIG["finetune_savedir"], "best_model.pt")
model_ft.load_state_dict(torch.load(best_path))
tm = evaluate(model_ft, test_loader, DEVICE)
print("=" * 60)
print("GSM8K TEST RESULTS (pretrained on DMath-HF → finetuned on GSM8K)")
print("=" * 60)
for k, v in tm.items():
    if isinstance(v, float): print(f"  {k:18s} = {v:.4f}")
    else:                    print(f"  {k:18s} = {v}")
"""


cells = [
    md("md_header",       MD_HEADER),
    code("c_env",         C_ENV),
    code("c_config",      C_CONFIG),
    code("c_utils",       C_UTILS),
    md("md_converter",    MD_CONVERTER),
    code("c_converter",   C_CONVERTER),
    md("md_load",         MD_LOAD),
    code("c_load",        C_LOAD),
    code("c_datasets",    C_DATASETS),
    code("c_arch",        C_ARCH),
    code("c_loss",        C_LOSS),
    code("c_eval",        C_EVAL),
    md("md_pretrain",     MD_PRETRAIN),
    code("c_pretrain",    C_PRETRAIN),
    md("md_finetune",     MD_FINETUNE),
    code("c_finetune",    C_FINETUNE),
    code("c_test",        C_TEST),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {OUT}  ({sum(len(c['source']) for c in cells)} source lines, {len(cells)} cells)")
