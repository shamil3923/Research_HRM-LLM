"""
Split cleaned GSM8K data into train/val sets (80/20).

Stratified by answer magnitude bucket so both splits have a similar
distribution of 1-digit / 2-digit / 3-digit / 4+ digit targets.

Usage:
    python src/split_data.py --seed 42 --val-frac 0.2
"""
import argparse
import json
import random
from collections import defaultdict


INPUT = "data/gsm8k_train_clean.json"
OUT_TRAIN = "data/gsm8k_train_split.json"
OUT_VAL = "data/gsm8k_val_split.json"


def magnitude_bucket(target: float) -> str:
    t = abs(int(round(target)))
    if t < 10:    return "1d"
    if t < 100:   return "2d"
    if t < 1000:  return "3d"
    if t < 10000: return "4d"
    return "5d+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--out-train", default=OUT_TRAIN)
    ap.add_argument("--out-val", default=OUT_VAL)
    args = ap.parse_args()

    data = json.load(open(args.input))
    print(f"Loaded {len(data)} clean samples from {args.input}")

    buckets = defaultdict(list)
    for d in data:
        buckets[magnitude_bucket(d["target"])].append(d)

    rng = random.Random(args.seed)
    train, val = [], []
    for k, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * args.val_frac)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    json.dump(train, open(args.out_train, "w"), indent=2)
    json.dump(val, open(args.out_val, "w"), indent=2)

    print(f"\nSplit (seed={args.seed}, val_frac={args.val_frac}):")
    print(f"  train: {len(train)}  -> {args.out_train}")
    print(f"  val:   {len(val)}    -> {args.out_val}")
    print(f"\nBucket distribution:")
    print(f"  {'bucket':>6}  {'train':>6}  {'val':>5}")
    for k in sorted(buckets.keys()):
        n_tr = sum(1 for d in train if magnitude_bucket(d["target"]) == k)
        n_va = sum(1 for d in val if magnitude_bucket(d["target"]) == k)
        print(f"  {k:>6}  {n_tr:>6}  {n_va:>5}")


if __name__ == "__main__":
    main()
