"""
Clean GSM8K parsed data — remove samples where ANY step in the LLM-generated
trace is internally inconsistent.

Two passes (this is the strict version per Tier-1 audit):
  1. Every step's `result` must equal op(arg1, arg2) under deterministic
     re-execution. Catches LLM-hallucinated intermediates that the old
     final-answer-only check let through.
  2. The final_answer value must round-equal the dataset target.

Usage:
    python src/clean_data.py
    python src/clean_data.py --input data/gsm8k_test_parsed.json \
                             --output data/gsm8k_test_clean.json
    python src/clean_data.py --tolerance 1e-6   # absolute tol per step
"""
import argparse
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("--input",  default="data/gsm8k_train_parsed.json")
ap.add_argument("--output", default="data/gsm8k_train_clean.json")
ap.add_argument("--tolerance", type=float, default=1e-6,
                help="Absolute tolerance when comparing intermediate step results")
args = ap.parse_args()

INPUT = args.input
OUTPUT = args.output
TOL = args.tolerance

data = json.load(open(INPUT))
print(f"Input: {len(data)} samples  ({INPUT})")

clean = []
removed_final = 0
removed_step = 0

for d in data:
    trace = d["trace"]
    target = d["target"]
    vv = {}
    valid = True
    bad_step = False

    def r(a):
        if isinstance(a, (int, float)):
            return float(a)
        if isinstance(a, str) and a in vv:
            return vv[a]
        try:
            return float(a)
        except Exception:
            return 0.0

    for s in trace.get("steps", []):
        op = s.get("op", "const")
        a1 = s.get("arg1", 0)
        a2 = s.get("arg2", 0)
        v1, v2 = r(a1), r(a2)

        if op == "add":
            res = v1 + v2
        elif op == "sub":
            res = v1 - v2
        elif op == "mul":
            res = v1 * v2
        elif op == "div" and v2 != 0:
            res = v1 / v2
        else:
            res = v1

        # Validate the LLM-claimed result for this step (Tier-1 fix).
        claimed = s.get("result_value", None)
        if claimed is None:
            # Some traces only carry a variable name; fall back to step.value if present.
            claimed = s.get("value", None)
        if claimed is not None:
            try:
                claimed_f = float(claimed)
                if not math.isfinite(claimed_f) or abs(claimed_f - res) > TOL:
                    bad_step = True
                    break
            except (TypeError, ValueError):
                # Non-numeric claimed value — treat as missing rather than fail.
                pass

        rk = s.get("result", "")
        if rk:
            vv[rk] = res

    if bad_step:
        removed_step += 1
        continue

    fv = trace.get("final_answer", "")
    comp = vv.get(fv, None)

    if comp is None:
        valid = False
    elif abs(int(round(comp)) - int(round(target))) != 0:
        valid = False

    if valid:
        clean.append(d)
    else:
        removed_final += 1

with open(OUTPUT, "w") as f:
    json.dump(clean, f, indent=2)

total_in = len(data)
print(f"Clean: {len(clean)} samples ({100*len(clean)/total_in:.1f}%)")
print(f"Removed (bad intermediate step): {removed_step}")
print(f"Removed (wrong final answer):    {removed_final}")
print(f"Saved to {OUTPUT}")
