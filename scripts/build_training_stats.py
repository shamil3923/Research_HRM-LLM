"""Build ui/training_stats.json from the latest training output.

Reads:
    output/best_meta.json       -> best val metrics
    output/training_log.json    -> per-eval-epoch history (for sparkline)
    output/test_results.json    -> headline test metrics + magnitude buckets
    output/explanations.json    -> coherence histogram

Writes:
    ui/training_stats.json      -> a single bundled summary the UI consumes
"""
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
UI_DIR = os.path.join(ROOT, "ui")


def _load(name):
    p = os.path.join(OUT_DIR, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing: {p}")
    return json.load(open(p))


meta = _load("best_meta.json")
log = _load("training_log.json")
test = _load("test_results.json")
exps = _load("explanations.json")

# Recover full question text from the val split for the explained samples.
val_split_path = os.path.join(ROOT, "data", "gsm8k_val_split.json")
val_raw = None
if os.path.exists(val_split_path):
    with open(val_split_path) as f:
        val_raw = json.load(f)


def _full_question(idx, fallback):
    """Look up full question by val-set index; fall back to the truncated string."""
    if val_raw is not None and 0 <= idx < len(val_raw):
        q = val_raw[idx].get("question")
        if q:
            return q
    return fallback


def _val_trace(idx):
    """Return the symbolic trace for this val sample so the UI can render steps."""
    if val_raw is not None and 0 <= idx < len(val_raw):
        return val_raw[idx].get("trace") or {}
    return {}

# Per-epoch sparkline (val exact + loss, downsampled to <= 50 points)
def _spark(rows, key, n=50):
    if not rows:
        return []
    if len(rows) <= n:
        return [{"epoch": r["epoch"], "v": r.get(key)} for r in rows]
    step = max(1, len(rows) // n)
    return [{"epoch": rows[i]["epoch"], "v": rows[i].get(key)} for i in range(0, len(rows), step)]


coherence_scores = exps.get("scores", [])
coh_dist = dict(Counter(coherence_scores))
for k in (1, 2, 3, 4, 5):
    coh_dist.setdefault(k, 0)

stats = {
    "model": {
        "name": "HRMForMath",
        "params": "6.3M",
        "architecture": "Bridge (GAT×3) + HRM (H×4 / L×4, Hc=3 Lc=4) + ACT (max=4)",
        "optimizer": "AdamATan2 (atan2 update, β=(0.9, 0.95))",
        "epochs_trained": log[-1]["epoch"] if log else None,
    },
    "best": {
        "epoch": meta["epoch"],
        "val_exact_acc": meta["val_exact_acc"],
        "val_digit_acc": meta["val_digit_acc"],
        "val_near_acc": meta["val_near_acc"],
        "mean_halt_steps": meta["mean_halt_steps"],
        "no_output": meta["no_output"],
    },
    "test": {
        "n": test["total"],
        "exact_acc": test["exact_acc"],
        "near_acc": test["near_acc"],
        "digit_acc": test["digit_acc"],
        "no_output": test["no_output"],
        "mean_halt_steps": test["mean_halt_steps"],
        "by_bucket": [
            {"bucket": k, "n": v["n"], "exact": v["exact"], "near": v["near"]}
            for k, v in sorted(test["by_bucket"].items(),
                               key=lambda kv: ("1d", "2d", "3d", "4d", "5d+").index(kv[0]))
        ],
        "n_errors": len(test.get("errors", [])),
    },
    "interpretability": {
        "n_explanations": len(coherence_scores),
        "mean_coherence": exps.get("mean_coherence"),
        "distribution": coh_dist,
    },
    "training": {
        "n_eval_points": len(log),
        "skipped_steps_total": log[-1].get("skipped_steps") if log else 0,
        "spark_loss":  _spark(log, "loss"),
        "spark_exact": _spark(log, "exact_acc"),
        "spark_halt":  _spark(log, "mean_halt_steps"),
    },
    # Full sample list for the UI's "sample explanations" panel.
    # `question` is the FULL text recovered from gsm8k_val_split.json (the
    # explanations.json file itself only saves the first 80 chars).
    # `trace` is the symbolic computation graph parsed by the LLM parser,
    # so the UI can render the per-step ground-truth values alongside HRM's.
    "explanation_samples": [
        {
            "i": s["i"],
            "question": _full_question(s["i"], s.get("question") or ""),
            "trace":    _val_trace(s["i"]),
            "true": s["true"],
            "hrm_pred": s["hrm_pred"],
            "coherence": s["coherence"],
            "explanation": s.get("explanation") or "",
            "explanation_truncated": (len(s.get("explanation") or "") >= 1800
                                       and not (s.get("explanation") or "").rstrip().endswith((".", "!", "?", "*"))),
            # Per-step HRM-decoded values (added by regenerate_explanations.py)
            "hrm_per_node": s.get("hrm_per_node"),
            "halt_step":    s.get("halt_step"),
        }
        for s in exps.get("samples", [])
    ],
}

out = os.path.join(UI_DIR, "training_stats.json")
with open(out, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Wrote {out}")
print(f"  best val exact   : {stats['best']['val_exact_acc']*100:.2f}%  @ epoch {stats['best']['epoch']}")
print(f"  test exact (n={stats['test']['n']}): {stats['test']['exact_acc']*100:.2f}%")
print(f"  mean coherence   : {stats['interpretability']['mean_coherence']}  (n={stats['interpretability']['n_explanations']})")
print(f"  buckets          : " + ", ".join(f"{b['bucket']}={b['exact']*100:.1f}%" for b in stats['test']['by_bucket']))
