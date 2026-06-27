"""
Combined static file server + live-inference backend for the v3.2 PyTorch HRM.

Replaces both `python -m http.server` and the older MLX-based serve_ui.py.
Loads output/best_model2.pt once at startup, then per /api/predict request:

  1. Parses the natural-language question with Qwen 3.5 (LLM parser) into
     a symbolic trace {steps:[{op,arg1,arg2,result}], final_answer}.
  2. Runs HRM on that trace, capturing per-cycle z_H / z_L norms,
     per-node decoded values, and the final answer.
  3. Generates a grounded, HRM-first explanation with Qwen 3.5 (LLM explainer)
     conditioned on HRM's actual per-step decodings.
  4. Returns one JSON blob the UI knows how to render.

Usage:
    python serve_v32.py                # serves http://localhost:8765
    python serve_v32.py --port 8000

Stop with Ctrl+C.
"""
from __future__ import annotations
import argparse, json, os, re, sys, threading, time, traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler

import torch
from dotenv import load_dotenv
load_dotenv()

# Re-use the v3.2 model + parse logic from the exporter we already built.
from export_predictions_v32 import (
    HRMForMath, parse_graph, decode_digits, encode_number,
    OP_VOCAB, DIGIT_VOCAB, IDX2DIG, IDX2OP, MAX_DIGITS, DIGIT_VOCAB_SIZE,
)
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(ROOT, "ui")

# ─── Globals (loaded once at startup) ────────────────────────────────────────
DEVICE = None
MODEL = None
LLM_CLIENT = None
LLM_MODEL = None
LOAD_LOCK = threading.Lock()


def init_model(ckpt_path: str, device_pref: str):
    global DEVICE, MODEL
    if device_pref == "auto":
        DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        DEVICE = torch.device(device_pref)
    print(f"[init] device = {DEVICE}", flush=True)
    print(f"[init] loading {ckpt_path}", flush=True)
    MODEL = HRMForMath(vsz=len(OP_VOCAB), d=256, heads=8,
                       Hc=3, Lc=4, Hl=4, Ll=4, slen=50).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    MODEL.load_state_dict(sd, strict=False)
    MODEL.eval()
    n = sum(p.numel() for p in MODEL.parameters())
    print(f"[init] HRMForMath ready ({n/1e6:.2f}M params)", flush=True)


def init_llm():
    global LLM_CLIENT, LLM_MODEL
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("[init] WARN: NVIDIA_API_KEY missing — /api/predict will return without explanation", flush=True)
        return
    from openai import OpenAI
    LLM_CLIENT = OpenAI(
        base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
    )
    LLM_MODEL = os.environ.get("NVIDIA_MODEL", "qwen/qwen3.5-122b-a10b")
    print(f"[init] LLM ready: {LLM_MODEL}", flush=True)


# ─── LLM parsing ─────────────────────────────────────────────────────────────
PARSER_SYSTEM = (
    "You are an expert at extracting deterministic reasoning traces from math "
    "word problems. Convert the problem into a JSON object with: "
    "  - steps: array of {op: 'add'|'sub'|'mul'|'div'|'const', arg1, arg2, result}. "
    "    arg1/arg2 are either numbers OR string names of earlier step results "
    "    (e.g. 'v1', 'v2'). result is the name of THIS step's output. "
    "  - final_answer: name of the step whose value is the answer. "
    "Return ONLY the JSON, no commentary."
)


LLM_TIMEOUT_PARSER = 35        # hard wall-clock cap, enforced via thread
LLM_TIMEOUT_EXPLAINER = 60     # ditto


import concurrent.futures
_LLM_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")
# Once we've seen the LLM time out, mark it dead for the rest of this session
# so subsequent /api/predict requests fail fast instead of hanging 12s each.
LLM_DEAD_REASON: str | None = None


def _call_with_timeout(fn, timeout):
    """Run fn() in a worker; if it doesn't return in `timeout` seconds raise TimeoutError."""
    fut = _LLM_POOL.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        # Best-effort cancel — the underlying HTTP socket may still keep churning,
        # but the request thread is no longer blocking us.
        fut.cancel()
        raise TimeoutError(f"LLM did not respond within {timeout}s")


def parse_question(question: str) -> tuple[dict, str | None]:
    """Convert a natural-language question into a symbolic trace via Qwen 3.5.

    Returns (trace, error_str). On any failure (no API key, timeout, bad JSON),
    `trace` is empty and `error_str` describes what happened.
    """
    global LLM_DEAD_REASON
    if LLM_CLIENT is None:
        return {}, "no NVIDIA_API_KEY configured"
    if LLM_DEAD_REASON is not None:
        return {}, f"LLM parser unavailable: {LLM_DEAD_REASON} (cached this session)"

    def _do_call():
        return LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": PARSER_SYSTEM},
                      {"role": "user",   "content": f"Problem:\n{question.strip()}\n\nReturn the JSON object now."}],
            temperature=0.0, max_tokens=1024, timeout=LLM_TIMEOUT_PARSER,
            extra_body={"chat_template_kwargs": {"thinking": False}},
        )

    try:
        resp = _call_with_timeout(_do_call, LLM_TIMEOUT_PARSER)
    except TimeoutError as e:
        LLM_DEAD_REASON = str(e)
        return {}, f"LLM parser unavailable: {e}"
    except Exception as e:
        LLM_DEAD_REASON = f"{type(e).__name__}: {e}"
        return {}, f"LLM parser unavailable: {type(e).__name__}: {e}"

    content = resp.choices[0].message.content or ""
    m = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if m: content = m.group(1)
    elif "```" in content:
        m2 = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
        if m2: content = m2.group(1)
    try:
        return json.loads(content), None
    except json.JSONDecodeError as e:
        return {}, f"LLM returned non-JSON: {e}"


# ─── HRM inference + per-step decoding ───────────────────────────────────────
@torch.no_grad()
def run_hrm(trace: dict) -> dict:
    nids, nvals, ndigs, adj, nr = parse_graph(trace, max_nodes=50)
    ni = torch.tensor([nids],  dtype=torch.long,    device=DEVICE)
    nv = torch.tensor([nvals], dtype=torch.float32, device=DEVICE)
    am = torch.tensor([adj],   dtype=torch.float32, device=DEVICE)

    zH, zL = MODEL.init_carry(1, ni.shape[1], DEVICE)
    h_states, l_states = None, None
    qh, qc = None, None
    halt_step = 0
    for seg in range(4):
        dl, qh, qc, zH, zL, h_states, l_states = MODEL.step_with_trace(ni, nv, am, zH, zL)
        halt_step = seg + 1
        if (qh > qc).item() and (seg + 1) >= 2:
            break

    li = max(0, nr - 1)
    final_logits = dl[0, li]
    pred_tok = final_logits.argmax(-1).cpu().tolist()
    pred_int = decode_digits(pred_tok)

    # Per-node decodings
    node_pred_tok = dl[0].argmax(-1).cpu().tolist()
    nodes, edges = [], []
    for i in range(nr):
        op_name = IDX2OP.get(int(nids[i]), "PAD")
        node_pred_int = decode_digits(node_pred_tok[i])
        node_target_int = decode_digits(ndigs[i])
        emb_norm = round(float(torch.norm(zH[0, i]).item()), 3)
        nodes.append({
            "id": i, "op": op_name,
            "arg1_norm": round(float(nvals[i][0]), 3),
            "arg2_norm": round(float(nvals[i][2]), 3),
            "embedding_norm": emb_norm,
            "predicted": node_pred_int,
            "target": node_target_int,
            "correct": node_pred_int == node_target_int,
        })
    for i in range(nr):
        for j in range(nr):
            if adj[i, j] > 0.5 and i != j:
                edges.append({"source": i, "target": j})

    # Per-position top-13 digit distributions for final node
    probs = F.softmax(final_logits, dim=-1).cpu().tolist()
    digit_probs = []
    for d in range(MAX_DIGITS):
        dist = {IDX2DIG.get(v, "?"): round(probs[d][v], 4) for v in range(DIGIT_VOCAB_SIZE)}
        digit_probs.append({
            "position": d,
            "predicted": IDX2DIG.get(int(pred_tok[d]), "?"),
            "distribution": dist,
        })

    # Per-node integer list (HRM's beliefs at every step)
    hrm_per_node = [decode_digits(node_pred_tok[i]) for i in range(nr)]

    return {
        "num_nodes": nr,
        "pred_int": pred_int,
        "pred_digits": [IDX2DIG.get(int(t), "?") for t in pred_tok],
        "graph": {"nodes": nodes, "edges": edges},
        "h_states": h_states or [],
        "l_states": l_states or [],
        "digit_probs": digit_probs,
        "q_halt":     round(float(qh.item()), 4) if qh is not None else None,
        "q_continue": round(float(qc.item()), 4) if qc is not None else None,
        "halt_step":  halt_step,
        "hrm_per_node": hrm_per_node,
    }


# ─── HRM-first explanation ───────────────────────────────────────────────────
EXPLAINER_SYSTEM = (
    "You are an interpretability analyst for a small neural network called HRM. "
    "HRM has just solved a math word problem by running through a fixed symbolic "
    "computation graph. At every node, HRM's hidden state was decoded into an "
    "integer prediction ('HRM decoded'). Your job: narrate HRM's reasoning "
    "trajectory step by step. Rules: (1) LEAD with what HRM did — every paragraph "
    "starts with HRM's decoded value at that step, not with the textbook solution. "
    "(2) Use the symbolic trace only as a reference frame to state whether HRM "
    "agreed or disagreed at each step. (3) If HRM disagreed, state the magnitude "
    "of the disagreement and one short hypothesis. (4) End with one summary "
    "sentence on the most consequential step in HRM's trajectory. (5) Do NOT "
    "redo the math from scratch — explain what HRM did."
)


def _trace_lines(trace, hrm_per_node):
    sym = {"add": "+", "sub": "-", "mul": "×", "div": "÷"}
    vmap = {}
    out = []
    for i, s in enumerate(trace.get("steps", [])):
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
        if hrm_v is not None and hrm_v == int(round(rv)):
            verdict = "✓ matches"
        elif hrm_v is not None:
            verdict = f"✗ HRM is off by {abs(int(round(rv)) - hrm_v)}"
        else:
            verdict = "?"
        out.append(
            f"  Step {i+1}: symbolic {op:<4} {v1:g} {sym.get(op,op)} {v2:g} = {rv:g}  ({rk})\n"
            f"           HRM decoded: {hrm_v}   [{verdict}]"
        )
    return "\n".join(out), int(round(vmap.get(trace.get("final_answer", ""), 0)))


def generate_explanation(question: str, trace: dict, hrm_out: dict) -> tuple[str, str | None]:
    """Returns (explanation, error_str)."""
    if LLM_CLIENT is None or not trace.get("steps"):
        return "", "LLM not configured or empty trace"
    block, sym_answer = _trace_lines(trace, hrm_out["hrm_per_node"])
    user = (
        f"Problem:\n{question.strip()}\n\n"
        f"Symbolic trace + HRM's decoded value at each step:\n{block}\n\n"
        f"HRM's final answer: {hrm_out['pred_int']}\n"
        f"Symbolically correct answer: {sym_answer}\n"
        f"HRM halted at ACT segment {hrm_out['halt_step']} of 4.\n\n"
        f"Now narrate HRM's reasoning trajectory step by step, following the rules above."
    )
    global LLM_DEAD_REASON
    if LLM_DEAD_REASON is not None:
        return "", f"LLM explainer unavailable: {LLM_DEAD_REASON} (cached this session)"

    def _do_call():
        return LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": EXPLAINER_SYSTEM},
                      {"role": "user",   "content": user}],
            temperature=0.2, max_tokens=2048, timeout=LLM_TIMEOUT_EXPLAINER,
            extra_body={"chat_template_kwargs": {"thinking": False}},
        )
    try:
        resp = _call_with_timeout(_do_call, LLM_TIMEOUT_EXPLAINER)
    except TimeoutError as e:
        LLM_DEAD_REASON = str(e)
        return "", f"LLM explainer unavailable: {e}"
    except Exception as e:
        LLM_DEAD_REASON = f"{type(e).__name__}: {e}"
        return "", f"LLM explainer unavailable: {type(e).__name__}: {e}"
    return resp.choices[0].message.content.strip(), None


# ─── HTTP handler ────────────────────────────────────────────────────────────
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve everything from ui/
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def end_headers(self):
        # Disable browser caching so file updates are picked up immediately.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stdout.write("[http] " + (fmt % args) + "\n")
        sys.stdout.flush()

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404, "Unknown endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body) if body else {}
            question = (payload.get("question") or "").strip()
            supplied_trace = payload.get("trace") or None
            want_explanation = bool(payload.get("explain", False))
            if not question and not supplied_trace:
                return self._respond_json({"error": "Missing 'question' (or 'trace')."}, 400)

            t0 = time.time()
            parser_error = None
            with LOAD_LOCK:
                # 1. Parse with LLM (unless user supplied a pre-parsed trace)
                if supplied_trace:
                    trace = supplied_trace
                else:
                    trace, parser_error = parse_question(question)
                if not trace.get("steps"):
                    return self._respond_json({
                        "error": parser_error or "Parser returned an empty trace.",
                        "hint": "If the NVIDIA API is unreachable, pass a pre-parsed trace "
                                "as { question, trace: {steps:[...], final_answer:'v2'} }.",
                        "trace": trace,
                    }, 503 if parser_error and "unavailable" in parser_error else 422)

                # 2. Run HRM (always works — uses local PyTorch model only)
                hrm = run_hrm(trace)

                # 3. Compute symbolic answer
                vmap = {}
                for s in trace.get("steps", []):
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
                    rk = s.get("result", "")
                    if rk: vmap[rk] = rv
                sym_ans = int(round(vmap.get(trace.get("final_answer", ""), 0)))

                # 4. Grounded explanation — OPT-IN via {"explain": true}. The default
                #    skip path keeps response time bounded to local HRM inference (~1s).
                #    When opted in but the LLM is unreachable, the response still ships
                #    with explainer_error set and the rest of the pipeline intact.
                if want_explanation:
                    explanation, explainer_error = generate_explanation(question, trace, hrm)
                else:
                    explanation, explainer_error = "", "skipped (set explain=true to request)"

            return self._respond_json({
                "question": question,
                "trace": trace,
                "computed_answer": sym_ans,
                "predicted_answer": hrm["pred_int"],
                "correct": hrm["pred_int"] == sym_ans,
                "near_match": abs(hrm["pred_int"] - sym_ans) <= 1 if hrm["pred_int"] != -1 else False,
                "num_nodes": hrm["num_nodes"],
                "pred_digits": hrm["pred_digits"],
                "graph": hrm["graph"],
                "h_states": hrm["h_states"],
                "l_states": hrm["l_states"],
                "digit_probs": hrm["digit_probs"],
                "q_halt": hrm["q_halt"],
                "q_continue": hrm["q_continue"],
                "halt_step": hrm["halt_step"],
                "hrm_per_node": hrm["hrm_per_node"],
                "explanation": explanation,
                "explainer_error": explainer_error,
                "parser_error": parser_error,
                "elapsed_sec": round(time.time() - t0, 2),
            })
        except Exception as e:
            traceback.print_exc()
            return self._respond_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _respond_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ckpt", default="output/best_model2.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = ap.parse_args()

    init_model(os.path.join(ROOT, args.ckpt), args.device)
    init_llm()

    srv = HTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}/  (UI dir: {UI_DIR})", flush=True)
    print("[serve] POST /api/predict   {'question': '...'}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down", flush=True)


if __name__ == "__main__":
    main()
