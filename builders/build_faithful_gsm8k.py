"""Build a FAITHFUL GSM8K trace dataset from the official <<a op b = c>> annotations.

Unlike the LLM-confabulated traces (which hit the target with fabricated numbers),
every step here comes from GSM8K's own ground-truth calculator annotations, so the
trace is guaranteed to use the problem's actual numbers and reasoning.

Pipeline per problem:
  1. Extract every <<expr=result>> from the answer (left-to-right = solution order).
  2. Turn each into a step {op, arg1, arg2, result:"vN"}.
  3. Link an operand to a PRIOR step's result (-> "vN" reference) when their values
     match, so the chain becomes a proper DAG (matches parse_graph's expectations).
  4. Validate: re-execute the trace and require final == gold (#### N).
  5. Measure number-grounding: fraction of literal operands present in the question.

Output schema (identical to data/gsm8k_train_split.json):
  {"trace": {"steps": [{op, arg1, arg2, result}], "final_answer": "vN"},
   "target": <number>, "question": <str>}

Run:  venv/bin/python3 build_faithful_gsm8k.py
"""
import ast, json, re, statistics as st
from pathlib import Path
from collections import Counter

OUT = Path(__file__).resolve().parent.parent / "data"  # repo-root data/
TRAIN_VAL_FRAC = 0.85
ALLOWED_OPS = {"add", "sub", "mul", "div"}
SYM2OP = {"+": "add", "-": "sub", "*": "mul", "x": "mul", "/": "div"}

ANNOT_RE = re.compile(r"<<\s*([^>]+?)\s*=\s*(-?[\d.]+)\s*>>")
FINAL_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
QNUM_RE = re.compile(r"\d+(?:\.\d+)?")
_AST_OP = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}


def num(s):
    s = str(s).replace(",", "")
    f = float(s)
    return int(f) if f.is_integer() else f


def question_numbers(q):
    out = set()
    for m in QNUM_RE.findall(q):
        out.add(num(m))
    return out


def extract_trace(question, answer):
    """Return (trace_dict, target, grounded_fraction) or (None, reason, None)."""
    fm = FINAL_RE.search(answer)
    if not fm:
        return None, "no_final", None
    target = num(fm.group(1))

    annots = ANNOT_RE.findall(answer)
    if not annots:
        return None, "no_annotations", None

    steps = []
    results = []          # value of each produced vN, in order
    qnums = question_numbers(question)
    literal_operands = []

    def link(value):
        """If value equals a prior result, return 'vK' reference; else the literal.
        Counts literals for the grounding metric."""
        for k in range(len(results) - 1, -1, -1):
            if abs(results[k] - value) < 1e-9:
                return f"v{k+1}"
        literal_operands.append(value)
        return value

    def emit(node):
        """Recursively turn an AST expression into binary steps.
        Returns (arg, value): arg is a literal number or a 'vN' reference."""
        if isinstance(node, ast.Expression):
            return emit(node.body)
        if isinstance(node, ast.Constant):
            v = num(node.value)
            return link(v), v
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            arg, v = emit(node.operand)
            return -v if not isinstance(arg, str) else arg, -v   # fold const negation
        if isinstance(node, ast.BinOp) and type(node.op) in _AST_OP:
            la, lv = emit(node.left)
            ra, rv = emit(node.right)
            op = _AST_OP[type(node.op)]
            if op == "div" and rv == 0:
                raise ValueError("div0")
            val = {"add": lv+rv, "sub": lv-rv, "mul": lv*rv, "div": (lv/rv if rv else None)}[op]
            idx = len(steps) + 1
            steps.append({"op": op, "arg1": la, "arg2": ra, "result": f"v{idx}"})
            results.append(val)
            return f"v{idx}", val
        raise ValueError("unsupported_node")

    for expr, res in annots:
        e = expr.replace("x", "*").replace("X", "*")
        try:
            tree = ast.parse(e, mode="eval")
            top_arg, top_val = emit(tree)
        except (ValueError, SyntaxError, ZeroDivisionError):
            return None, "unparsable_expr", None
        # a bare number annotation (<<6=6>>) yields no BinOp -> skip silently
        if not isinstance(top_arg, str):
            continue

    if not steps:
        return None, "no_steps", None

    # final answer = the LAST step whose computed value equals the gold target
    final_idx = None
    for k in range(len(results) - 1, -1, -1):
        if results[k] is not None and abs(results[k] - float(target)) < 1e-6:
            final_idx = k + 1
            break
    if final_idx is None:
        return None, "no_step_hits_target", None

    trace = {"steps": steps, "final_answer": f"v{final_idx}"}

    # validate: re-execute and require final == gold
    got = reexecute(trace)
    if got is None or abs(got - float(target)) > 1e-6:
        return None, "exec_mismatch", None

    # grounding: literal operands that appear in the question
    if literal_operands:
        g = sum(1 for x in literal_operands if x in qnums) / len(literal_operands)
    else:
        g = 1.0
    return trace, target, g


def reexecute(trace):
    vv = {}
    for s in trace["steps"]:
        def r(a):
            if isinstance(a, (int, float)):
                return float(a)
            if isinstance(a, str) and a in vv:
                return vv[a]
            try:
                return float(a)
            except Exception:
                return None
        v1, v2, op = r(s["arg1"]), r(s["arg2"]), s["op"]
        if v1 is None or v2 is None:
            return None
        if op == "add": rv = v1 + v2
        elif op == "sub": rv = v1 - v2
        elif op == "mul": rv = v1 * v2
        elif op == "div": rv = v1 / v2 if v2 != 0 else None
        else: return None
        if rv is None:
            return None
        vv[s["result"]] = rv
    return vv.get(trace["final_answer"])


def process(split_name, rows):
    kept, gfrac, steps_hist, ops = [], [], Counter(), Counter()
    reasons = Counter()
    for ex in rows:
        trace, target, g = extract_trace(ex["question"], ex["answer"])
        if trace is None:
            reasons[target] += 1   # 'target' holds the reason string on failure
            continue
        kept.append({"trace": trace, "target": target, "question": ex["question"]})
        gfrac.append(g)
        steps_hist[len(trace["steps"])] += 1
        for s in trace["steps"]:
            ops[s["op"]] += 1
    print(f"\n=== {split_name}: kept {len(kept)}/{len(rows)} ({len(kept)/len(rows):.1%}) ===")
    print(f"  step-count dist: {sorted(steps_hist.items())}")
    print(f"  op dist: {dict(ops)}")
    print(f"  mean number-grounding: {st.mean(gfrac):.1%}  (was 24% in confabulated set)")
    print(f"  fully-grounded samples: {sum(1 for x in gfrac if x>0.999)/len(gfrac):.1%}")
    print(f"  top drop reasons: {reasons.most_common(6)}")
    return kept


def main():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")
    train_full = list(ds["train"])
    test = list(ds["test"])

    n = int(len(train_full) * TRAIN_VAL_FRAC)
    train_rows, val_rows = train_full[:n], train_full[n:]

    OUT.mkdir(exist_ok=True)
    for name, rows, fn in [
        ("train", train_rows, "gsm8k_faithful_train.json"),
        ("val", val_rows, "gsm8k_faithful_val.json"),
        ("test", test, "gsm8k_faithful_test.json"),
    ]:
        kept = process(name, rows)
        with open(OUT / fn, "w") as f:
            json.dump(kept, f)
        print(f"  -> wrote {OUT/fn}")


if __name__ == "__main__":
    main()
