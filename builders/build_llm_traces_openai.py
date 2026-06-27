"""Phase-2 LLM parsing: GSM8K -> trace JSON via the OpenAI BATCH API (gpt-4o-mini).

SEPARATE pipeline. Does NOT read, modify, or delete any existing file (the Qwen
gsm8k_*_parsed/split files stay untouched). Writes ONLY:
    data/gsm8k_llm_traces_openai_<split>.json   (validated, bridge-ready)
    data/gsm8k_llm_traces_openai_stats.json

Output schema is exactly what the bridge consumes (parse_graph / GraphDataset /
chain_execute):
    [ {"trace": {"steps":[{"op","arg1","arg2","result"}], "final_answer":"vK"},
       "target": <int>, "question": <str>}, ... ]

Validation gates reject confabulation: a trace is kept only if it RE-EXECUTES to the
dataset's gold target; number-grounding is reported so you can compare to the
faithful baseline (~73%) before trusting the data.

Run:
  export OPENAI_API_KEY=sk-...
  pip install openai datasets
  venv/bin/python3 build_llm_traces_openai.py
"""
import json, re, time, os
from pathlib import Path
from collections import Counter

# ---------------- config ----------------
MODEL      = "gpt-4o-mini"
SPLITS     = ["train"]         # test already parsed; now do train (~7.5K, ~$0.85)
DRY_RUN    = False             # True = parse DRY_N items synchronously first (cheap check)
DRY_N      = 20
MAX_TOKENS = 600
OUTDIR     = Path(__file__).resolve().parent.parent / "data"  # repo-root data/
ALLOWED_OPS = {"add", "sub", "mul", "div"}

SYSTEM_PROMPT = """You are a strict math problem parser. Convert a GSM8K word problem and its final \
numeric answer into a structured trace of arithmetic steps. Output ONLY one JSON object, no prose, \
no markdown.

Schema:
{"question":"...","target":123,"trace":{"steps":[{"op":"add|sub|mul|div","arg1":<int or "vN">,"arg2":<int or "vN">,"result":"v0"}],"final_answer":"vK"}}

Rules:
- Each step is ONE atomic op on TWO args; args are integer literals or prior "vN" names.
- result names are v0,v1,v2,... in order, with no gaps.
- final_answer is the var whose value equals target.
- Use div ONLY for exact integer division.
- Use the numbers stated in the problem; do not invent numbers.
- All numbers are plain integers. Output a single JSON object only."""


def user_msg(question, target):
    return (f"Question:\n{question}\n\nFinal answer (from dataset):\n{target}\n\n"
            "Output ONLY the JSON object.")


# ---------------- validation (rejects confabulation) ----------------
_QNUM  = re.compile(r"\d+(?:\.\d+)?")
_FINAL = re.compile(r"####\s*(-?[\d,]+)")


def gold_target(answer_text):
    m = _FINAL.search(answer_text)
    if not m: return None
    try: return int(m.group(1).replace(",", ""))
    except ValueError: return None


def qnums(q):
    out = set()
    for s in _QNUM.findall(q):
        try: out.add(int(float(s)))
        except ValueError: pass
    return out


def validate(trace, target, question):
    """Return (clean_trace, grounding_fraction) or (None, reason)."""
    if not isinstance(trace, dict): return None, "not_dict"
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps: return None, "no_steps"
    vals, literals = {}, []
    for s in steps:
        op = s.get("op")
        if op not in ALLOWED_OPS: return None, "bad_op"
        def resolve(a):
            if isinstance(a, str) and a in vals: return vals[a]
            try:
                f = float(a)
                if f != int(f): return None
                literals.append(int(f)); return int(f)
            except (ValueError, TypeError): return None
        a1, a2 = resolve(s.get("arg1")), resolve(s.get("arg2"))
        if a1 is None or a2 is None: return None, "bad_arg"
        if op == "add": r = a1 + a2
        elif op == "sub": r = a1 - a2
        elif op == "mul": r = a1 * a2
        else:
            if a2 == 0 or a1 % a2 != 0: return None, "inexact_div"
            r = a1 // a2
        rk = s.get("result")
        if not rk: return None, "no_result"
        vals[rk] = r
    fa = trace.get("final_answer")
    if fa not in vals: return None, "final_missing"
    if vals[fa] != target: return None, "exec_mismatch"      # kills confabulation
    qn = qnums(question)
    grounded = sum(1 for x in literals if x in qn) / max(1, len(literals))
    return {"steps": steps, "final_answer": fa}, grounded


def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except json.JSONDecodeError: return None


def keep(parsed, answer_text, question):
    """Uses the DATASET's gold target (not the model's). Returns (record|None, info)."""
    tgt = gold_target(answer_text)
    if tgt is None: return None, "no_gold"
    trace = parsed.get("trace", parsed) if isinstance(parsed, dict) else None
    clean, info = validate(trace, tgt, question)
    if clean is None: return None, info
    return {"trace": clean, "target": tgt, "question": question}, info


# ---------------- data ----------------
def load_rows():
    from datasets import load_dataset
    rows = []
    for sp in SPLITS:
        ds = load_dataset("gsm8k", "main", split=sp)
        items = list(ds)[:DRY_N] if DRY_RUN else list(ds)
        for i, ex in enumerate(items):
            rows.append((sp, i, ex))
    return rows


# ---------------- request building ----------------
def build_request(cid, ex):
    return {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": MODEL, "max_tokens": MAX_TOKENS,
                     "response_format": {"type": "json_object"},
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user",
                                   "content": user_msg(ex["question"], gold_target(ex["answer"]))}]}}


# ---------------- runners ----------------
def run_dry(client, rows):
    responses = {}
    for sp, i, ex in rows:
        r = client.chat.completions.create(**build_request(f"{sp}-{i}", ex)["body"])
        responses[f"{sp}-{i}"] = r.choices[0].message.content
    return responses


def run_batch(client, rows):
    OUTDIR.mkdir(exist_ok=True)
    jl = OUTDIR / "_batch_input.jsonl"
    jl.write_text("\n".join(json.dumps(build_request(f"{sp}-{i}", ex)) for sp, i, ex in rows))
    up = client.files.create(file=open(jl, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                  completion_window="24h")
    print(f"batch {batch.id} submitted ({len(rows)} requests). Polling every 30s...")
    while True:
        b = client.batches.retrieve(batch.id)
        print(f"  [{time.strftime('%H:%M:%S')}] {b.status}  {b.request_counts}")
        if b.status in ("completed", "failed", "expired", "cancelled"): break
        time.sleep(30)
    if b.status != "completed":
        raise RuntimeError(f"batch ended: {b.status}")
    responses = {}
    for line in client.files.content(b.output_file_id).text.splitlines():
        d = json.loads(line)
        try:
            responses[d["custom_id"]] = d["response"]["body"]["choices"][0]["message"]["content"]
        except Exception:
            responses[d["custom_id"]] = None
    return responses


# ---------------- main ----------------
def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: set OPENAI_API_KEY"); return
    from openai import OpenAI
    client = OpenAI()
    rows = load_rows()
    print(f"splits={SPLITS}  problems={len(rows)}  mode={'DRY sync' if DRY_RUN else 'BATCH'}")

    responses = (run_dry if DRY_RUN else run_batch)(client, rows)

    import statistics as st
    per_split = {sp: [] for sp in SPLITS}
    reasons, gfrac = Counter(), []
    for sp, i, ex in rows:
        txt = responses.get(f"{sp}-{i}")
        if not txt: reasons["api_error"] += 1; continue
        parsed = parse_json(txt)
        if parsed is None: reasons["json_parse"] += 1; continue
        rec, info = keep(parsed, ex["answer"], ex["question"])
        if rec is None: reasons[info] += 1; continue
        per_split[sp].append(rec); gfrac.append(info)

    OUTDIR.mkdir(exist_ok=True)
    stats = {"model": MODEL, "reject_reasons": dict(reasons)}
    for sp in SPLITS:
        out = OUTDIR / f"gsm8k_llm_traces_openai_{sp}.json"
        json.dump(per_split[sp], open(out, "w"))
        stats[sp] = {"kept": len(per_split[sp])}
        print(f"  saved {len(per_split[sp])} -> {out}")
    kept = sum(len(v) for v in per_split.values())
    print(f"\nkept {kept}/{len(rows)} = {kept/len(rows)*100:.1f}%  (validated: re-exec == gold target)")
    if gfrac:
        gm = st.mean(gfrac)
        print(f"mean number-grounding: {gm*100:.1f}%  (faithful baseline ~73%; "
              f"much lower => gpt-4o-mini is confabulating)")
        stats["mean_grounding"] = gm
    print(f"top reject reasons: {reasons.most_common(6)}")
    json.dump(stats, open(OUTDIR / "gsm8k_llm_traces_openai_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
