"""
LLM Explanation Module — converts an HRM reasoning trace into a
step-by-step natural language explanation.

This is Module 7 from the research report (Section 6.8):
    "trace -> step-by-step natural language explanation"

It uses the same NVIDIA endpoint and Qwen 3.5 model as the parser.

Pipeline:
    GSM8K problem
      -> LLM parser  -> structured JSON trace          (cache_gsm8k.py)
      -> HRM core    -> predicted final answer         (notebook training)
      -> THIS MODULE -> human-readable explanation

Usage (programmatic):
    from src.explainer import explain
    text = explain(question, trace, predicted_answer)

Usage (CLI):
    python src/explainer.py --question "Janet has 3 apples..." \\
        --trace '{"steps":[{"op":"add","arg1":3,"arg2":5,"result":"v1"}, \\
                            {"op":"sub","arg1":"v1","arg2":2,"result":"v2"}], \\
                 "final_answer":"v2"}' \\
        --answer 6
"""
import argparse
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
_MODEL    = os.environ.get("NVIDIA_MODEL", "qwen/qwen3.5-122b-a10b")

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not _API_KEY:
            raise RuntimeError("NVIDIA_API_KEY missing in env / .env")
        _client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY)
    return _client


def _resolve_value(arg: Any, vmap: dict) -> float:
    if isinstance(arg, (int, float)):
        return float(arg)
    if isinstance(arg, str) and arg in vmap:
        return vmap[arg]
    try:
        return float(arg)
    except Exception:
        return 0.0


def trace_to_lines(trace: dict) -> list[str]:
    """
    Render the symbolic trace into ordered, human-readable arithmetic lines.

    Example output:
        Step 1: add  3 + 5 = 8       (v1)
        Step 2: sub  8 - 2 = 6       (v2)
        Final answer: v2 = 6
    """
    OP_SYM = {"add": "+", "sub": "-", "mul": "x", "div": "/"}
    vmap: dict[str, float] = {}
    lines: list[str] = []
    for i, s in enumerate(trace.get("steps", []), start=1):
        op = s.get("op", "const")
        a1 = _resolve_value(s.get("arg1", 0), vmap)
        a2 = _resolve_value(s.get("arg2", 0), vmap)
        if op == "add":               r = a1 + a2
        elif op == "sub":             r = a1 - a2
        elif op == "mul":             r = a1 * a2
        elif op == "div" and a2 != 0: r = a1 / a2
        else:                         r = a1
        rk = s.get("result", f"v{i}")
        vmap[rk] = r
        sym = OP_SYM.get(op, op)
        a1s = f"{a1:g}"
        a2s = f"{a2:g}"
        rs  = f"{r:g}"
        lines.append(f"  Step {i}: {op:<4} {a1s} {sym} {a2s} = {rs}   ({rk})")

    fa = trace.get("final_answer", "")
    if fa and fa in vmap:
        lines.append(f"  Final answer: {fa} = {vmap[fa]:g}")
    return lines


def explain(question: str,
            trace: dict,
            predicted_answer: float | int,
            model: str | None = None,
            max_retries: int = 3,
            temperature: float = 0.2) -> str:
    """
    Generate a natural-language explanation for a single problem.

    Returns a clean, multi-paragraph string.
    """
    rendered = "\n".join(trace_to_lines(trace))
    system = (
        "You are a math tutor. You will be given a word problem, the "
        "structured reasoning trace that the model used, and the model's "
        "final numeric answer. Your job is to write a short, clear, "
        "step-by-step explanation that a student could follow. "
        "Tie each computation step to the quantities mentioned in the "
        "problem. Do not invent extra steps. Do not change the final "
        "answer. End with a single bold line of the form: "
        "**Answer: <number>**."
    )
    user = (
        f"Problem:\n{question.strip()}\n\n"
        f"Reasoning trace:\n{rendered}\n\n"
        f"Model's final answer: {predicted_answer}\n\n"
        f"Write the explanation now."
    )

    client = _get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model or _MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                time.sleep(2 ** attempt * 3)
                continue
            time.sleep(1)
    raise RuntimeError(f"Explanation generation failed after {max_retries} retries: {last_err}")


def explain_batch(items: list[dict], **kw) -> list[dict]:
    """
    items: list of {"question", "trace", "predicted_answer"}
    Returns each item with an added "explanation" field.
    """
    out = []
    for it in items:
        try:
            text = explain(it["question"], it["trace"], it["predicted_answer"], **kw)
        except Exception as e:
            text = f"[explanation error: {e}]"
        rec = dict(it)
        rec["explanation"] = text
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--trace", required=True, help="JSON string of the trace")
    ap.add_argument("--answer", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    trace = json.loads(args.trace)
    try:
        ans = int(args.answer)
    except ValueError:
        ans = float(args.answer)

    print("\n--- TRACE ---")
    for line in trace_to_lines(trace):
        print(line)

    print("\n--- EXPLANATION ---")
    print(explain(args.question, trace, ans, model=args.model))


if __name__ == "__main__":
    main()
