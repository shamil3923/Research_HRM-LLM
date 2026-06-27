"""Re-parse GSM8K with Claude Opus 4.8 into HRM trace format.

Downloads GSM8K from HuggingFace, sends every problem to Claude Opus 4.8
in a single batch via the Anthropic Message Batches API (50% discount),
polls until done, validates each response, and writes train/val/test JSON
files in the exact format the HRM Graph-Aware Bridge consumes.

Output format per record (matches parse_graph in the notebooks):
{
  "trace": {
    "steps": [
      {"op": "add|sub|mul|div", "arg1": <int or "vN">, "arg2": <int or "vN">,
       "result": "vN", "result_value": <int>}, ...
    ],
    "final_answer": "vN"
  },
  "target": <int>
}

Usage:
  1. Install:  pip install anthropic datasets
  2. Set:      export ANTHROPIC_API_KEY=sk-ant-...
  3. Run:      python reparse_gsm8k_with_claude.py
  4. Files:    out/gsm8k_train_split.json
               out/gsm8k_val_split.json
               out/gsm8k_test_clean.json

Cost estimate (batch API, ~7.5K train + ~1K val + ~1.3K test = ~10K problems):
  - Input:  ~700 tokens/problem * 10K = 7M tokens @ $7.50/MTok (batch) = $52.50
  - Output: ~250 tokens/problem * 10K = 2.5M tokens @ $37.50/MTok (batch) = $93.75
  - Total: ~$150 for full corpus. Batch can take 1-24h to complete.

For testing first, use LIMIT_PER_SPLIT (default 50).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

# -------------------------------- config
OUT_DIR              = Path(__file__).parent / "out_reparsed"
LIMIT_PER_SPLIT      = None      # set to 50 for a cheap dry-run; None for full
MODEL_ID             = "claude-opus-4-8"
MAX_OUTPUT_TOKENS    = 768
TRAIN_VAL_SPLIT_FRAC = 0.85      # 85% train, 15% val from official train split
POLL_INTERVAL_SEC    = 60        # how often to poll batch status
USE_BATCH_API        = True      # set False to use the regular Messages API (no discount)

OUT_DIR.mkdir(exist_ok=True)


# -------------------------------- prompt
SYSTEM_PROMPT = """You are a precise math problem parser. You convert grade-school math word problems into structured computation traces in a strict JSON format. Output ONLY the JSON object, with no preamble, no markdown fences, no explanation. The trace must use ONLY integer arithmetic.

OUTPUT SCHEMA (output exactly this structure, no other keys):
{
  "steps": [
    {"op": "add"|"sub"|"mul"|"div", "arg1": <int or "vN">, "arg2": <int or "vN">, "result": "vN", "result_value": <int>}
  ],
  "final_answer": "vN"
}

RULES:
1. Each step uses ONE binary operation on TWO arguments.
2. arg1/arg2 are either integer literals or string references to prior step results ("v0", "v1", ...).
3. Result names are sequential: v0, v1, v2, ...
4. result_value MUST be an integer. If a real intermediate is fractional, find a different decomposition.
5. For division: only use it when the integer result is exact (no remainder).
6. The LAST step's result key must equal final_answer.
7. The final result_value MUST equal the problem's given answer.
8. Use the MINIMUM number of steps that correctly computes the answer.
9. Prefer add/sub when chaining (running totals), mul for "X each" rates, div for "split among N".
10. If the problem genuinely cannot be expressed as integer arithmetic, output: {"steps": [], "final_answer": "v0", "error": "non_integer"}

EXAMPLES:

Problem: Janet has 45 eggs. She sells 27 at $2 each. How much does she earn?
Answer: 36
Output:
{"steps":[{"op":"sub","arg1":45,"arg2":27,"result":"v0","result_value":18},{"op":"mul","arg1":"v0","arg2":2,"result":"v1","result_value":36}],"final_answer":"v1"}

Wait — re-read. "She sells 27 at $2 each" — earnings = 27 * 2 = 54, the 45-27 isn't the earnings. Let me re-do:
Output:
{"steps":[{"op":"mul","arg1":27,"arg2":2,"result":"v0","result_value":54}],"final_answer":"v0"}

(Use the second one — careful reading matters.)

Problem: A bag has 12 apples and 8 oranges. Tom takes 5 apples and 3 oranges. How many fruits remain?
Answer: 12
Output:
{"steps":[{"op":"add","arg1":12,"arg2":8,"result":"v0","result_value":20},{"op":"add","arg1":5,"arg2":3,"result":"v1","result_value":8},{"op":"sub","arg1":"v0","arg2":"v1","result":"v2","result_value":12}],"final_answer":"v2"}

Now parse the user's problem. Output ONLY the JSON object."""


def make_user_prompt(question: str, answer_text: str) -> str:
    ans = extract_final_answer(answer_text)
    return f"Problem: {question}\nAnswer: {ans}\n\nOutput ONLY the JSON object."


# -------------------------------- helpers
_FINAL_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*)")


def extract_final_answer(answer_text: str) -> str:
    """GSM8K answers end with '#### <number>'."""
    m = _FINAL_ANSWER_RE.search(answer_text)
    if not m:
        return answer_text.strip().split("\n")[-1].strip()
    return m.group(1).replace(",", "")


def validate_trace(trace_json, expected_answer: int):
    """Return (trace_dict, target_int) if valid, else (None, None) with reason."""
    if not isinstance(trace_json, dict): return None, None, "not a dict"
    if trace_json.get("error") == "non_integer": return None, None, "model: non_integer"
    steps = trace_json.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, None, "no steps"
    if len(steps) > 30:
        return None, None, "too many steps"

    values = {}
    for i, s in enumerate(steps):
        if not isinstance(s, dict): return None, None, f"step {i}: not dict"
        op = s.get("op")
        if op not in {"add", "sub", "mul", "div"}:
            return None, None, f"step {i}: bad op {op!r}"
        for k in ("arg1", "arg2"):
            if k not in s: return None, None, f"step {i}: missing {k}"
        if "result" not in s: return None, None, f"step {i}: missing result"
        if "result_value" not in s: return None, None, f"step {i}: missing result_value"

        def resolve(a):
            if isinstance(a, (int, float)):
                if isinstance(a, float) and a != int(a): return None
                return int(a)
            if isinstance(a, str):
                if a in values: return values[a]
                try:
                    v = float(a)
                    return int(v) if v == int(v) else None
                except ValueError:
                    return None
            return None

        v1, v2 = resolve(s["arg1"]), resolve(s["arg2"])
        if v1 is None or v2 is None:
            return None, None, f"step {i}: unresolved args"
        if   op == "add": expected = v1 + v2
        elif op == "sub": expected = v1 - v2
        elif op == "mul": expected = v1 * v2
        else:
            if v2 == 0 or v1 % v2 != 0:
                return None, None, f"step {i}: bad div {v1}/{v2}"
            expected = v1 // v2
        rv = s["result_value"]
        if isinstance(rv, float) and rv != int(rv):
            return None, None, f"step {i}: fractional result"
        rv = int(rv)
        if rv != expected:
            return None, None, f"step {i}: model said {rv} expected {expected}"
        values[s["result"]] = rv

    fa = trace_json.get("final_answer")
    if fa not in values: return None, None, f"final_answer {fa!r} not produced"
    if values[fa] != expected_answer:
        return None, None, f"final {values[fa]} != target {expected_answer}"

    return {"steps": steps, "final_answer": fa}, int(expected_answer), "ok"


def parse_response_content(text: str):
    """Strip optional ```json fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
        text = re.sub(r"```$", "", text).strip()
    # Find the first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# -------------------------------- dataset loading
def load_gsm8k():
    """Returns dict {split_name: [(question, answer_text, custom_id), ...]}."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("pip install datasets")

    print("Loading GSM8K from HuggingFace...")
    train_full = load_dataset("gsm8k", "main", split="train")
    test       = load_dataset("gsm8k", "main", split="test")

    # Split official train into our train/val (85/15)
    n_train_total = len(train_full)
    n_train = int(n_train_total * TRAIN_VAL_SPLIT_FRAC)

    splits = {
        "train": [(ex["question"], ex["answer"], f"train-{i}")
                  for i, ex in enumerate(train_full.select(range(n_train)))],
        "val":   [(ex["question"], ex["answer"], f"val-{i}")
                  for i, ex in enumerate(train_full.select(range(n_train, n_train_total)))],
        "test":  [(ex["question"], ex["answer"], f"test-{i}")
                  for i, ex in enumerate(test)],
    }
    if LIMIT_PER_SPLIT:
        for k in splits:
            splits[k] = splits[k][:LIMIT_PER_SPLIT]
    for k, v in splits.items():
        print(f"  {k:>6s}: {len(v)} problems")
    return splits


# -------------------------------- batch API
def submit_batch(client, examples):
    """examples: list of (question, answer_text, custom_id).
    Returns batch_id."""
    from anthropic.types.messages.batch_create_params import Request
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    requests = []
    for question, answer_text, custom_id in examples:
        requests.append(Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=MODEL_ID,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user",
                           "content": make_user_prompt(question, answer_text)}],
            ),
        ))
    print(f"Submitting batch of {len(requests)} requests to {MODEL_ID}...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch_id={batch.id}  created_at={batch.created_at}")
    return batch.id


def poll_until_done(client, batch_id):
    """Block until batch finishes. Returns the completed batch object."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print(f"  [{time.strftime('%H:%M:%S')}] {status}  "
              f"processing={counts.processing} succeeded={counts.succeeded} "
              f"errored={counts.errored} canceled={counts.canceled} "
              f"expired={counts.expired}")
        if status == "ended":
            return batch
        time.sleep(POLL_INTERVAL_SEC)


def fetch_batch_results(client, batch_id):
    """Yields (custom_id, response_text or None on failure)."""
    for line in client.messages.batches.results(batch_id):
        cid = line.custom_id
        if line.result.type == "succeeded":
            msg = line.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            yield cid, text
        else:
            print(f"  [error] {cid}: {line.result.type}")
            yield cid, None


# -------------------------------- per-request (fallback if batch disabled)
def call_one(client, question, answer_text):
    msg = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": make_user_prompt(question, answer_text)}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


# -------------------------------- driver
def process_split(client, split_name, examples):
    """Parse one split. Returns (kept_records, stats)."""
    print(f"\n=== Split: {split_name} ({len(examples)} problems) ===")

    if USE_BATCH_API:
        batch_id = submit_batch(client, examples)
        print("Polling for completion...")
        poll_until_done(client, batch_id)
        print("Downloading results...")
        responses = dict(fetch_batch_results(client, batch_id))
    else:
        responses = {}
        for q, a, cid in examples:
            try: responses[cid] = call_one(client, q, a)
            except Exception as e:
                print(f"  [error] {cid}: {e}")
                responses[cid] = None

    kept, parse_fail, validate_fail = [], 0, 0
    reasons = {}
    for q, a, cid in examples:
        text = responses.get(cid)
        if text is None:
            parse_fail += 1; reasons["api_error"] = reasons.get("api_error", 0) + 1
            continue
        parsed = parse_response_content(text)
        if parsed is None:
            parse_fail += 1; reasons["json_parse"] = reasons.get("json_parse", 0) + 1
            continue
        try:
            target_int = int(extract_final_answer(a))
        except ValueError:
            validate_fail += 1
            reasons["non_int_target"] = reasons.get("non_int_target", 0) + 1
            continue
        trace, target, reason = validate_trace(parsed, target_int)
        if trace is None:
            validate_fail += 1
            short = reason.split(":")[0]
            reasons[short] = reasons.get(short, 0) + 1
            continue
        kept.append({"trace": trace, "target": target,
                     "question": q, "id": cid})

    print(f"  Kept: {len(kept)}/{len(examples)} ({len(kept)/max(1,len(examples)):.1%})")
    print(f"  Parse failures: {parse_fail}, validation failures: {validate_fail}")
    print(f"  Top failure reasons: {sorted(reasons.items(), key=lambda x: -x[1])[:5]}")
    return kept, {"kept": len(kept), "total": len(examples),
                  "parse_fail": parse_fail, "validate_fail": validate_fail,
                  "reasons": reasons}


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: set ANTHROPIC_API_KEY")
        sys.exit(1)

    try:
        from anthropic import Anthropic
    except ImportError:
        print("pip install anthropic"); sys.exit(1)

    client = Anthropic()
    splits = load_gsm8k()

    OUT_NAMES = {
        "train": "gsm8k_train_split.json",
        "val":   "gsm8k_val_split.json",
        "test":  "gsm8k_test_clean.json",
    }

    all_stats = {}
    for split_name in ["train", "val", "test"]:
        kept, stats = process_split(client, split_name, splits[split_name])
        out_path = OUT_DIR / OUT_NAMES[split_name]
        with open(out_path, "w") as f:
            json.dump(kept, f, indent=None)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved {len(kept)} records → {out_path}  ({size_mb:.2f} MB)")
        all_stats[split_name] = stats

    with open(OUT_DIR / "reparse_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nAll splits done. Stats saved to {OUT_DIR / 'reparse_stats.json'}")
    print(f"\nUpload these three files to Kaggle as a new dataset, then point the")
    print(f"training notebook's DATA_ROOT at the new mount path.")


if __name__ == "__main__":
    main()
