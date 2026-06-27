"""
Regenerate grounded explanations from HRM's perspective.

The first cell-14 run produced explanations that lead with the *correct* symbolic
logic and only mention HRM's decoded values as a side note. This rewrite flips
the framing: the LLM is told to NARRATE HRM's reasoning trajectory — what HRM
believed at each step, why it ended at its final answer, and where (if anywhere)
it diverged from the symbolic ground truth. The judge prompt is unchanged.

Also raises max_tokens 512 → 2048 to capture the full reasoning.

Reads:
    output/best_model2.pt           (PyTorch state_dict)
    data/gsm8k_val_split.json       (val problems + parsed traces)
    output/explanations.json        (sample selection — which 20 val indices to redo)
    .env                            (NVIDIA_API_KEY)

Writes:
    output/explanations.json        (overwrites with new HRM-first explanations)

Usage:
    python regenerate_explanations.py           # redo same 20 samples
    python regenerate_explanations.py --n 30    # do first 30 val samples
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter

import torch
import torch.nn.functional as F

from dotenv import load_dotenv
load_dotenv()

# Re-use the v3.2 model + parse logic from the exporter we built earlier.
from export_predictions_v32 import (
    HRMForMath, parse_graph, decode_digits, encode_number,
    OP_VOCAB, DIGIT_VOCAB, IDX2DIG, IDX2OP, MAX_DIGITS, DIGIT_VOCAB_SIZE,
)

ROOT = os.path.dirname(os.path.abspath(__file__))


def build_model(device, ckpt_path):
    model = HRMForMath(vsz=len(OP_VOCAB), d=256, heads=8,
                       Hc=3, Lc=4, Hl=4, Ll=4, slen=50).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def hrm_per_node_predictions(model, trace, device, act_max=4, act_min=2):
    """Run HRM and return: list of HRM's decoded int per symbolic step + final answer."""
    nids, nvals, ndigs, adj, nr = parse_graph(trace, max_nodes=50)
    ni = torch.tensor([nids],  dtype=torch.long,    device=device)
    nv = torch.tensor([nvals], dtype=torch.float32, device=device)
    am = torch.tensor([adj],   dtype=torch.float32, device=device)
    zH, zL = model.init_carry(1, ni.shape[1], device)
    for seg in range(act_max):
        dl, qh, qc, zH, zL, _, _ = model.step_with_trace(ni, nv, am, zH, zL)
        if (qh > qc).item() and seg + 1 >= act_min:
            break
    pred_tok = dl[0].argmax(-1).cpu().tolist()
    per_node = []
    for i in range(nr):
        per_node.append(decode_digits(pred_tok[i]))
    final = per_node[-1] if per_node else -1
    return per_node, final, seg + 1


def _format_trace_with_hrm(trace, hrm_per_node):
    sym = {"add": "+", "sub": "-", "mul": "×", "div": "÷"}
    vmap = {}
    lines = []
    steps = trace.get("steps", [])
    for i, s in enumerate(steps):
        op = s.get("op", "const")
        def resolve(a):
            if isinstance(a, (int, float)): return float(a)
            if isinstance(a, str) and a in vmap: return vmap[a]
            try: return float(a)
            except Exception: return 0.0
        v1, v2 = resolve(s.get("arg1", 0)), resolve(s.get("arg2", 0))
        if   op == "add":              rv = v1 + v2
        elif op == "sub":              rv = v1 - v2
        elif op == "mul":              rv = v1 * v2
        elif op == "div" and v2 != 0:  rv = v1 / v2
        else:                          rv = v1
        rk = s.get("result", f"v{i+1}")
        vmap[rk] = rv
        hrm_v = hrm_per_node[i] if i < len(hrm_per_node) else None
        agree = "✓ matches" if hrm_v == int(round(rv)) else f"✗ HRM is off by {abs(int(round(rv)) - hrm_v) if hrm_v is not None else 'n/a'}"
        lines.append(
            f"  Step {i+1}: symbolic {op:<4} {v1:g} {sym.get(op,op)} {v2:g} = {rv:g}  ({rk})\n"
            f"           HRM decoded at this step: {hrm_v}   [{agree}]"
        )
    return "\n".join(lines)


HRM_FIRST_SYSTEM = (
    "You are an interpretability analyst for a small neural network called HRM. "
    "HRM has just solved a math word problem by running through a fixed symbolic computation graph. "
    "At every node in that graph, HRM's hidden state was decoded into an integer prediction "
    "(this is what 'HRM decoded' means). Your job is to write a step-by-step narration of "
    "HRM's own reasoning trajectory — starting from HRM's first decoded value, walking through "
    "HRM's chain of decoded values, and ending at HRM's final answer.\n\n"
    "Rules:\n"
    "  1. LEAD with HRM. Every paragraph must start with what HRM did or believed at that step, "
    "not with the textbook solution.\n"
    "  2. Use the symbolic trace only as a reference frame to say whether HRM agreed or disagreed "
    "with the correct value at each step.\n"
    "  3. If HRM's decoded value matches the symbolic computation, say so explicitly.\n"
    "  4. If HRM's decoded value differs, give the magnitude of the disagreement and one short "
    "hypothesis for what might have caused it (e.g. place-value confusion, off-by-one, ignoring "
    "an argument). Do not lecture; speculate briefly.\n"
    "  5. End with one sentence summarising whether HRM's final answer was correct and what the "
    "single most consequential step in its trajectory was.\n"
    "  6. Do NOT redo the math from scratch. You are explaining what HRM did, not what the student "
    "should do."
)


def call_explainer(client, model_name, system, user, max_tokens, temperature):
    return client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=temperature, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"thinking": False}},
    ).choices[0].message.content.strip()


JUDGE_SYSTEM = (
    "You are a strict grader. Given a word problem, the symbolic computation trace, "
    "and a step-by-step explanation, rate the explanation's COHERENCE on a 1-5 integer scale.\n"
    "Specifically, score:\n"
    "  5 = each step clearly tied to a quantity in the problem; reasoning flow is unbroken\n"
    "  4 = mostly clear, minor jumps\n"
    "  3 = understandable but some steps unjustified\n"
    "  2 = confusing or several wrong jumps\n"
    "  1 = incoherent or unrelated to the problem\n"
    "Respond with ONLY a single digit 1-5."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="number of val samples to (re)explain. Default: same indices "
                         "as the existing explanations.json")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--ckpt", default="output/best_model2.pt")
    ap.add_argument("--val",  default="data/gsm8k_val_split.json")
    ap.add_argument("--existing", default="output/explanations.json")
    ap.add_argument("--out", default="output/explanations.json")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        sys.exit("NVIDIA_API_KEY missing (set in .env or environment)")
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("NVIDIA_BASE_URL",
                                            "https://integrate.api.nvidia.com/v1"),
                    api_key=api_key)
    model_name = os.environ.get("NVIDIA_MODEL", "qwen/qwen3.5-122b-a10b")

    device = (torch.device("mps") if (args.device == "auto" and torch.backends.mps.is_available())
              else torch.device("cpu" if args.device != "mps" else "mps"))
    print(f"device  : {device}")
    print(f"model   : {model_name}")

    val_raw = json.load(open(os.path.join(ROOT, args.val)))
    if args.n is None:
        existing = json.load(open(os.path.join(ROOT, args.existing)))
        indices = [s["i"] for s in existing.get("samples", [])]
        print(f"reusing the {len(indices)} sample indices from existing explanations.json")
    else:
        indices = list(range(min(args.n, len(val_raw))))
        print(f"explaining the first {len(indices)} val samples")

    model = build_model(device, os.path.join(ROOT, args.ckpt))

    samples_out, scores = [], []
    for k, idx in enumerate(indices):
        item = val_raw[idx]
        q = item.get("question", "")
        trace = item.get("trace", {})
        target = int(round(float(item.get("target", 0))))

        per_node, hrm_final, halt = hrm_per_node_predictions(model, trace, device)
        trace_block = _format_trace_with_hrm(trace, per_node)

        user_prompt = (
            f"Problem:\n{q}\n\n"
            f"Symbolic trace WITH HRM's decoded value at each step:\n{trace_block}\n\n"
            f"HRM's final answer: {hrm_final}\n"
            f"Correct answer: {target}\n"
            f"Halt step: {halt} of 4 ACT segments\n\n"
            f"Now narrate HRM's reasoning trajectory step-by-step, following the rules above."
        )
        try:
            text = call_explainer(client, model_name, HRM_FIRST_SYSTEM, user_prompt,
                                  max_tokens=args.max_tokens, temperature=0.2)
        except Exception as e:
            print(f"[{k+1:>3}/{len(indices)}] EXPLAINER ERROR on val#{idx}: {e}")
            continue

        # Coherence judge
        try:
            judge_user = (
                f"Problem:\n{q.strip()}\n\n"
                f"Trace:\n{json.dumps(trace.get('steps', []), indent=2)}\n\n"
                f"Explanation:\n{text}\n\nScore (1-5):"
            )
            judge_raw = call_explainer(client, model_name, JUDGE_SYSTEM, judge_user,
                                       max_tokens=4, temperature=0.0)
            score = next((int(ch) for ch in judge_raw if ch in "12345"), None)
        except Exception:
            score = None
        if score is not None:
            scores.append(score)

        samples_out.append({
            "i": idx, "question": q, "true": target,
            "hrm_pred": hrm_final, "explanation": text, "coherence": score,
            "halt_step": halt, "hrm_per_node": per_node,
        })
        print(f"[{k+1:>3}/{len(indices)}]  val#{idx:>4}  true={target}  hrm={hrm_final}  "
              f"halt={halt}  coh={score}  ({len(text)} chars)")
        time.sleep(0.4)   # gentle on the rate limit

    mean_c = sum(scores) / max(1, len(scores)) if scores else None
    dist = dict(Counter(scores))
    out = {"mean_coherence": mean_c, "scores": scores, "samples": samples_out}

    out_path = os.path.join(ROOT, args.out)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"  n = {len(samples_out)}, mean coherence = {mean_c}, dist = {dist}")


if __name__ == "__main__":
    main()
