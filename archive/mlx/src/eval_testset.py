"""
Evaluate a trained HRM checkpoint on the official GSM8K test split.

Inputs:
  - checkpoints/best_model.pt        (or --ckpt path)
  - data/gsm8k_test_clean.json       (parser output filtered by clean_data.py)

Outputs:
  - exact-match accuracy
  - near-match (+-1) accuracy
  - per-digit accuracy
  - per-magnitude breakdown (1d / 2d / 3d / 4d / 5d+)
  - JSON dump of every wrong prediction for error analysis

Usage:
    python src/eval_testset.py --ckpt checkpoints/best_model.pt \
                               --data data/gsm8k_test_clean.json
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from kaggle.hrm_gsm8k_pytorch import (
    OP_VOCAB, DIGIT_VOCAB, DIGIT_VOCAB_SIZE, MAX_DIGITS, IDX_TO_DIGIT,
    GSM8KDataset, collate_fn, HRMForMath, decode_digits_to_number,
)


def magnitude_bucket(t: int) -> str:
    t = abs(int(t))
    if t < 10:    return "1d"
    if t < 100:   return "2d"
    if t < 1000:  return "3d"
    if t < 10000: return "4d"
    return "5d+"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    bucket_total = defaultdict(int)
    bucket_exact = defaultdict(int)
    bucket_near  = defaultdict(int)
    dig_ok = dig_tot = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        digit_logits, _, _ = model(batch)
        B, N, D, V = digit_logits.shape
        last_idx = (batch["num_real_nodes"] - 1).clamp(0, N - 1)

        for b in range(B):
            pred_digits = digit_logits[b, last_idx[b].item()].argmax(-1).cpu().tolist()
            pred = decode_digits_to_number(pred_digits)
            true = int(batch["raw_target"][b].item())
            bk = magnitude_bucket(true)
            bucket_total[bk] += 1
            if pred == true:
                bucket_exact[bk] += 1
            if abs(pred - true) <= 1:
                bucket_near[bk] += 1

            tdig = batch["final_digit_tgt"][b].cpu().tolist()
            for d in range(D):
                if tdig[d] != DIGIT_VOCAB["PAD"]:
                    dig_tot += 1
                    if pred_digits[d] == tdig[d]:
                        dig_ok += 1

            if pred != true:
                rows.append({"pred": pred, "true": true, "abs_err": abs(pred - true)})

    total = sum(bucket_total.values())
    exact = sum(bucket_exact.values())
    near  = sum(bucket_near.values())
    return {
        "total": total,
        "exact_acc": exact / max(1, total),
        "near_acc":  near  / max(1, total),
        "digit_acc": dig_ok / max(1, dig_tot),
        "by_bucket": {
            k: {"n": bucket_total[k],
                "exact": bucket_exact[k] / max(1, bucket_total[k]),
                "near":  bucket_near[k]  / max(1, bucket_total[k])}
            for k in sorted(bucket_total.keys())
        },
        "errors": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best_model.pt")
    ap.add_argument("--data", default="data/gsm8k_test_clean.json")
    ap.add_argument("--errors-out", default="checkpoints/test_errors.json")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--Hcycles", type=int, default=4)
    ap.add_argument("--Lcycles", type=int, default=8)
    ap.add_argument("--Hlayers", type=int, default=8)
    ap.add_argument("--Llayers", type=int, default=8)
    ap.add_argument("--max-nodes", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = GSM8KDataset(args.data, max_nodes=args.max_nodes)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, collate_fn=collate_fn)

    model = HRMForMath(
        op_vocab_size=len(OP_VOCAB),
        d_model=args.d_model, n_heads=args.n_heads,
        H_cycles=args.Hcycles, L_cycles=args.Lcycles,
        H_layers=args.Hlayers, L_layers=args.Llayers,
        seq_len=args.max_nodes,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.ckpt}")

    res = evaluate(model, loader, device)

    print("\n" + "=" * 60)
    print(f"GSM8K TEST EVALUATION  ({res['total']} parsed test problems)")
    print("=" * 60)
    print(f"  Exact-match : {res['exact_acc']*100:6.2f}%")
    print(f"  Near-match  : {res['near_acc']*100:6.2f}%   (|pred-true| <= 1)")
    print(f"  Digit-level : {res['digit_acc']*100:6.2f}%")
    print(f"\nBy magnitude bucket:")
    print(f"  {'bucket':>6}  {'n':>5}  {'exact%':>7}  {'near%':>6}")
    for k, v in res["by_bucket"].items():
        print(f"  {k:>6}  {v['n']:>5}  {v['exact']*100:>7.2f}  {v['near']*100:>6.2f}")

    os.makedirs(os.path.dirname(args.errors_out), exist_ok=True)
    json.dump(res, open(args.errors_out, "w"), indent=2)
    print(f"\nFull error dump: {args.errors_out}")


if __name__ == "__main__":
    main()
