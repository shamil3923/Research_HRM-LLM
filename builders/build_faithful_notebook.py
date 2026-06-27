"""Builds notebook_faithful_train.ipynb — digit-aware HRM with a 3-stage
pretrain(synthetic) -> curriculum(faithful) -> finetune(faithful) pipeline.

The core architecture/data/generator code is inlined from faithful_hrm.py so the
notebook is self-contained on Kaggle.
Run from repo root: venv/bin/python3 builders/build_faithful_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root (builders/ -> root)
CORE = (ROOT / "faithful_hrm.py").read_text()

def md(t):  return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}

cells = []

cells.append(md(
"""# Faithful-HRM — GSM8K math reasoning (digit-aware bridge + HRM/ACT)

**What changed vs v3.1**
1. **Faithful dataset.** Traces come from GSM8K's own `<<a op b = c>>` annotations
   (number-grounding 73–75% vs 24% in the LLM-confabulated set). No confabulation.
2. **Digit-aware bridge (architectural).** Operands are encoded as DIGIT sequences,
   not a lossy `log1p` scalar — the HRM can now see digits and learn carry arithmetic.
3. **3-stage training (same distribution):**
   - **Stage 1 – Pretrain** on synthetic arithmetic graphs sampled to match GSM8K's
     op-mix / step-count / magnitude distribution (unlimited data).
   - **Stage 2 – Curriculum finetune** on faithful GSM8K, phased by step count.
   - **Stage 3 – Finetune** on all faithful GSM8K + number augmentation, ACT on.
4. Kept: HRM H/L core, ACT halting, Adam-atan2, deep supervision (aux loss).

**Add Data:** upload `gsm8k_faithful_train.json`, `gsm8k_faithful_val.json`,
`gsm8k_faithful_test.json` (produced by `build_faithful_gsm8k.py`).
"""))

cells.append(code(
"""# Cell 1 — Config
import os, glob, torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch", torch.__version__)

# Auto-locate the faithful dataset wherever it is mounted (Kaggle attaches it
# under /kaggle/input/<slug>). Falls back to env var, then local ./data.
_hits = glob.glob("/kaggle/input/**/gsm8k_faithful_train.json", recursive=True)
DATA_ROOT = os.path.dirname(_hits[0]) if _hits else os.environ.get("DATA_ROOT", "data")
print("DATA_ROOT:", DATA_ROOT)
assert os.path.exists(f"{DATA_ROOT}/gsm8k_faithful_train.json"), (
    f"faithful files not found under {DATA_ROOT!r}. "
    "Attach the dataset via 'Add Data' and check it contains gsm8k_faithful_*.json.")

CONFIG = dict(
    data_train=f"{DATA_ROOT}/gsm8k_faithful_train.json",
    data_val  =f"{DATA_ROOT}/gsm8k_faithful_val.json",
    data_test =f"{DATA_ROOT}/gsm8k_faithful_test.json",
    savedir   ="/kaggle/working/ckpt_faithful" if os.path.isdir("/kaggle") else "ckpt_faithful",

    # model
    dmodel=256, nheads=8, Hcycles=3, Lcycles=4, Hlayers=4, Llayers=4, max_nodes=40,

    # Encoding: True = HONEST (only leaf numbers fed; model computes+routes every
    # intermediate AND the answer). False = LEAKY ablation (operands+answer fed in,
    # which trivially hits ~100% — use only to demonstrate the leak).
    mask_refs=True,

    # Stage 1 — synthetic pretrain
    pretrain_n=60000, pretrain_epochs=8, pretrain_bs=256, pretrain_lr=6e-4,

    # Stage 2 — curriculum (phase = max step-count; epochs per phase)
    curriculum_phases=[(2, 12), (4, 12), (99, 16)], curr_bs=128, curr_lr=3e-4,

    # Stage 3 — full finetune + augmentation, ACT enabled
    finetune_epochs=80, finetune_bs=128, finetune_lr=2e-4,
    act_max_steps=3, act_min_steps=2, q_loss_weight=0.5, aux_loss_weight=1.0,
    augment_p=0.3, augment_max_value=300,

    # adam-atan2
    optim_betas=(0.9, 0.95), optim_wd=0.01, optim_a=1.27, optim_b=1.0,
)
os.makedirs(CONFIG["savedir"], exist_ok=True)
for k, v in CONFIG.items(): print(f"  {k:18s}= {v}")
"""))

cells.append(md("## Cell 2 — Core: encoding, faithful↔synthetic data, digit-aware HRM (inlined `faithful_hrm.py`)"))
cells.append(code("# Cell 2 — Core (inlined)\n" + CORE))

cells.append(code(
'''# Cell 3 — Adam-atan2 optimizer + reusable training-stage driver
import math, time, json
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

class AdamATan2(Optimizer):
    """Adam with bounded atan2 update + decoupled weight decay (HRM paper §3.6)."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, a=1.27, b=1.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b))
    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if not st:
                    st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p)
                m, v = st["m"], st["v"]; st["step"] += 1; t = st["step"]
                if g["weight_decay"]: p.mul_(1 - g["lr"]*g["weight_decay"])
                m.mul_(b1).add_(p.grad, alpha=1-b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1-b2)
                mh = m/(1-b1**t); vh = v/(1-b2**t)
                p.add_(torch.atan2(mh, vh.sqrt()*g["b"]), alpha=-g["lr"]*g["a"])

def make_opt(model):
    return AdamATan2(model.parameters(), lr=CONFIG["finetune_lr"],
                     betas=CONFIG["optim_betas"], weight_decay=CONFIG["optim_wd"],
                     a=CONFIG["optim_a"], b=CONFIG["optim_b"])

def run_stage(model, train_loader, val_loader, *, epochs, lr, tag,
              act_steps=1, act_min=1, aux_w=1.0, q_w=0.0, eval_every=2):
    """One training stage. act_steps>1 enables multi-segment ACT with halt supervision."""
    opt = make_opt(model)
    for pg in opt.param_groups: pg["lr"] = lr
    total = max(1, epochs*len(train_loader)); warm = max(1, int(0.05*total)); step = 0
    best = 0.0
    print(f"\\n===== STAGE: {tag}  (epochs={epochs}, lr={lr}, act_steps={act_steps}) =====")
    for ep in range(epochs):
        model.train(); el = es = 0
        for b in train_loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            lr_now = lr*step/warm if step < warm else lr*(0.05 + 0.95*0.5*(1+math.cos(
                math.pi*min(1.0,(step-warm)/max(1,total-warm)))))
            for pg in opt.param_groups: pg["lr"] = lr_now
            B, N = b["op_ids"].shape
            zH, zL = model.init_carry(B, N, DEVICE)
            segs = []
            for s in range(act_steps):
                dl, qh, qc, zH, zL = model.step(b, zH, zL); segs.append((dl, qh, qc))
            loss = 0.0
            for s, (dl, qh, qc) in enumerate(segs):
                nq = (torch.sigmoid(torch.maximum(segs[s+1][1], segs[s+1][2])).detach()
                      if s+1 < len(segs) else torch.sigmoid(qh).detach())
                w = 0.0 if (s+1) < act_min else q_w
                sl, _ = segment_loss(dl, b["final_digit_tgt"], b["node_digit_tgts"],
                                     b["num_real_nodes"], qh, qc, nq, aux_w, w, b["answer_idx"])
                loss = loss + sl
            loss = loss/len(segs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1; el += loss.item(); es += 1
        if (ep+1) % eval_every == 0 or ep == epochs-1:
            m = evaluate(model, val_loader, DEVICE, act_steps=act_steps)
            flag = ""
            if m["exact"] > best:
                best = m["exact"]; flag = " *"
                torch.save(model.state_dict(), os.path.join(CONFIG["savedir"], f"best_{tag}.pt"))
            print(f"  ep{ep+1:>3} loss={el/es:6.3f}  val_exact={m['exact']*100:5.2f}%  "
                  f"val_digit={m['digit']*100:5.1f}%  no_out={m['no_out']}{flag}")
    print(f"  -> best {tag} val_exact = {best*100:.2f}%")
    return best
'''))

cells.append(code(
'''# Cell 4 — Data: load faithful splits, generate same-distribution synthetic
import json
faithful_train = json.load(open(CONFIG["data_train"]))
faithful_val   = json.load(open(CONFIG["data_val"]))
faithful_test  = json.load(open(CONFIG["data_test"]))
print(f"faithful  train={len(faithful_train)}  val={len(faithful_val)}  test={len(faithful_test)}")

print(f"generating {CONFIG['pretrain_n']} synthetic arithmetic graphs (same distribution)...")
t0 = time.time()
synth = make_synthetic(CONFIG["pretrain_n"], seed=0, max_steps=9)
print(f"  done in {time.time()-t0:.1f}s")

MN = CONFIG["max_nodes"]; MR = CONFIG["mask_refs"]
print("encoding:", "HONEST (mask_refs=True)" if MR else "LEAKY ablation (mask_refs=False)")
pre_ds   = GraphDataset(synth, max_nodes=MN, mask_refs=MR)
train_ds = GraphDataset(faithful_train, max_nodes=MN, mask_refs=MR,
                        augment=True, augment_p=CONFIG["augment_p"],
                        augment_max_value=CONFIG["augment_max_value"])
val_ds   = GraphDataset(faithful_val, max_nodes=MN, mask_refs=MR)
test_ds  = GraphDataset(faithful_test, max_nodes=MN, mask_refs=MR)
val_loader  = DataLoader(val_ds,  batch_size=256, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate)

model = FaithfulHRM(d=CONFIG["dmodel"], heads=CONFIG["nheads"],
                    Hc=CONFIG["Hcycles"], Lc=CONFIG["Lcycles"],
                    Hl=CONFIG["Hlayers"], Ll=CONFIG["Llayers"], slen=MN).to(DEVICE)
print(f"FaithfulHRM params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
'''))

cells.append(md("## Stage 1 — Pretrain on synthetic arithmetic graphs"))
cells.append(code(
'''# Cell 5 — Stage 1: pretrain (teaches exact multi-step arithmetic, infinite data)
pre_loader = DataLoader(pre_ds, batch_size=CONFIG["pretrain_bs"], shuffle=True,
                        collate_fn=collate, drop_last=True)
run_stage(model, pre_loader, val_loader,
          epochs=CONFIG["pretrain_epochs"], lr=CONFIG["pretrain_lr"],
          tag="pretrain", act_steps=1, aux_w=1.0, q_w=0.0)
'''))

cells.append(md("## Stage 2 — Curriculum finetune on faithful GSM8K (easy→hard by step count)"))
cells.append(code(
'''# Cell 6 — Stage 2: curriculum by step count
for max_steps, epochs in CONFIG["curriculum_phases"]:
    n = train_ds.set_phase(max_steps)
    loader = DataLoader(train_ds, batch_size=CONFIG["curr_bs"], shuffle=True,
                        collate_fn=collate, drop_last=False)
    run_stage(model, loader, val_loader, epochs=epochs, lr=CONFIG["curr_lr"],
              tag=f"curr<=s{max_steps}", act_steps=1, aux_w=CONFIG["aux_loss_weight"], q_w=0.0)
train_ds.set_phase(99)
'''))

cells.append(md("## Stage 3 — Full finetune + augmentation, ACT enabled"))
cells.append(code(
'''# Cell 7 — Stage 3: full finetune with multi-segment ACT + number augmentation
train_ds.set_phase(99)
loader = DataLoader(train_ds, batch_size=CONFIG["finetune_bs"], shuffle=True,
                    collate_fn=collate, drop_last=False)
run_stage(model, loader, val_loader, epochs=CONFIG["finetune_epochs"],
          lr=CONFIG["finetune_lr"], tag="finetune",
          act_steps=CONFIG["act_max_steps"], act_min=CONFIG["act_min_steps"],
          aux_w=CONFIG["aux_loss_weight"], q_w=CONFIG["q_loss_weight"])
'''))

cells.append(md("## Test-set evaluation"))
cells.append(code(
'''# Cell 8 — Final held-out GSM8K test evaluation
best = os.path.join(CONFIG["savedir"], "best_finetune.pt")
if os.path.exists(best):
    model.load_state_dict(torch.load(best, map_location=DEVICE))
m = evaluate(model, test_loader, DEVICE, act_steps=CONFIG["act_max_steps"])
print(f"TEST  exact-match={m['exact']*100:.2f}%  digit-acc={m['digit']*100:.1f}%  "
      f"no_output={m['no_out']}/{m['n']}")
'''))

cells.append(code(
'''# Cell 9 — Per-step-count accuracy breakdown (shows where reasoning holds/breaks)
from collections import defaultdict
import torch
buckets = defaultdict(lambda: [0, 0])
model.eval()
with torch.no_grad():
    for tr, tgt, sc in [(r[0], r[1], r[2]) for r in test_ds.records]:
        b = collate([sample_to_tensors(tr, tgt, CONFIG["max_nodes"], CONFIG["mask_refs"])])
        b = {k: v.to(DEVICE) for k, v in b.items()}
        dl, _, _ = model(b, max_steps=CONFIG["act_max_steps"])
        idx = int(b["answer_idx"][0])
        p = decode_digits(dl[0, idx].argmax(-1).tolist())
        buckets[sc][1] += 1; buckets[sc][0] += int(p == int(round(tgt)))
print("steps  acc      n")
for sc in sorted(buckets):
    ok, n = buckets[sc]
    print(f"  {sc:>2}  {ok/n*100:5.1f}%  {n:>4}")
'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

# normalize source to list-of-lines (nbformat convention)
for c in nb["cells"]:
    s = c["source"]
    c["source"] = [l + "\n" for l in s.split("\n")]
    if c["source"]:
        c["source"][-1] = c["source"][-1].rstrip("\n")

out = ROOT / "notebooks" / "notebook_faithful_train.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
