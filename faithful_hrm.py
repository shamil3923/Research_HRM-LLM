"""
Faithful-HRM core: digit-aware Graph bridge + HRM(H/L)+ACT + digit head,
plus a synthetic arithmetic-graph generator and a 3-stage training driver
(pretrain on synthetic -> curriculum finetune -> full finetune).

This module is the single source of truth; the training notebook inlines it.
Designed to run on CPU (tiny) for smoke tests and CUDA (full) on Kaggle.
"""
import math, json, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ───────────────────────── vocab / encoding ─────────────────────────
OP_VOCAB = {"PAD": 0, "add": 1, "sub": 2, "mul": 3, "div": 4, "final_answer": 5}
OP_VOCAB_SIZE = len(OP_VOCAB)
DIGIT_VOCAB = {"PAD": 0, "0": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
               "6": 7, "7": 8, "8": 9, "9": 10, "NEG": 11, "EOS": 12}
DIGIT_VOCAB_SIZE = 13
MAX_DIGITS = 8
IDX2DIG = {v: k for k, v in DIGIT_VOCAB.items()}
MAX_ABS = 10 ** MAX_DIGITS - 1


def encode_number(val):
    n = int(round(val))
    d = []
    if n < 0:
        d.append(DIGIT_VOCAB["NEG"]); n = abs(n)
    for ch in str(n):
        d.append(DIGIT_VOCAB[ch])
    d.append(DIGIT_VOCAB["EOS"])
    while len(d) < MAX_DIGITS:
        d.append(DIGIT_VOCAB["PAD"])
    return d[:MAX_DIGITS]


def decode_digits(tokens):
    neg, s, seen = False, "", False
    for t in tokens:
        lbl = IDX2DIG.get(int(t), "PAD")
        if lbl == "PAD":
            continue
        if lbl == "EOS":
            break
        if lbl == "NEG":
            neg, seen = True, True
        else:
            s += lbl; seen = True
    if not seen or not s:
        return -1
    return -int(s) if neg else int(s)


# ───────────────────────── JSON trace -> tensors ─────────────────────────
def parse_graph(trace, max_nodes=40, mask_refs=True):
    """Digit-aware encoding of one trace.

    mask_refs=True (default, the honest setting): an operand that REFERENCES a
    prior node's result is NOT given its value — it is masked to zero, and the
    model must compute that value itself and route it through the graph edges.
    Only leaf literals (the problem's given numbers) are fed. The final_answer
    node is a pure readout (no operands) so the answer is never an input.

    mask_refs=False reproduces the leaky setting (operands + answer fed in) for
    the ablation that demonstrates why it hits ~100%.
    """
    op_ids, opnd_digits, is_ref, dig_tgts = [], [], [], []
    v2i, v2v = {}, {}
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)

    def resolve(a):
        if isinstance(a, (int, float)):
            return float(a), False
        if isinstance(a, str) and a in v2v:
            return v2v[a], True
        try:
            return float(a), False
        except Exception:
            return 0.0, False

    for i, step in enumerate(trace.get("steps", [])):
        if i >= max_nodes - 1:
            break
        op = step.get("op", "PAD")
        op_ids.append(OP_VOCAB.get(op, 0))
        v1, r1 = resolve(step.get("arg1", 0))
        v2, r2 = resolve(step.get("arg2", 0))
        if op == "add":   rv = v1 + v2
        elif op == "sub": rv = v1 - v2
        elif op == "mul": rv = v1 * v2
        elif op == "div" and v2 != 0: rv = v1 / v2
        else: rv = v1
        # mask referenced operands: the model must derive them via the graph
        d1 = encode_number(0) if (mask_refs and r1) else encode_number(v1)
        d2 = encode_number(0) if (mask_refs and r2) else encode_number(v2)
        opnd_digits.append([d1, d2])
        is_ref.append([float(r1), float(r2)])
        dig_tgts.append(encode_number(rv))
        rk = step.get("result", "")
        if rk:
            v2i[rk] = i; v2v[rk] = rv
        for k in ("arg1", "arg2"):
            a = step.get(k, "")
            if isinstance(a, str) and a in v2i:
                adj[v2i[a], i] = 1.0
        adj[i, i] = 1.0

    nr = len(op_ids)
    # The answer is read at the node that COMPUTES it (the final step), not a
    # bolt-on readout node — that copy-via-edge indirection bottlenecked even
    # 1-step problems. answer_idx is that node's index.
    fa = trace.get("final_answer", "")
    answer_idx = v2i.get(fa, max(0, nr - 1))

    while len(op_ids) < max_nodes:
        op_ids.append(0)
        opnd_digits.append([[0] * MAX_DIGITS, [0] * MAX_DIGITS])
        is_ref.append([0.0, 0.0])
        dig_tgts.append([0] * MAX_DIGITS)
    return op_ids, opnd_digits, is_ref, adj, dig_tgts, nr, answer_idx


def sample_to_tensors(trace, target, max_nodes, mask_refs=True):
    op_ids, opnd_digits, is_ref, adj, dig_tgts, nr, ans_idx = parse_graph(trace, max_nodes, mask_refs)
    tgt = int(round(float(target)))
    tgt = max(-MAX_ABS, min(MAX_ABS, tgt))
    return {
        "op_ids": torch.tensor(op_ids, dtype=torch.long),
        "opnd_digits": torch.tensor(opnd_digits, dtype=torch.long),   # (N,2,MAX_DIGITS)
        "is_ref": torch.tensor(is_ref, dtype=torch.float32),          # (N,2)
        "adj": torch.tensor(adj, dtype=torch.float32),                # (N,N)
        "node_digit_tgts": torch.tensor(dig_tgts, dtype=torch.long),  # (N,MAX_DIGITS)
        "final_digit_tgt": torch.tensor(encode_number(tgt), dtype=torch.long),
        "raw_target": torch.tensor(tgt, dtype=torch.long),
        "num_real_nodes": torch.tensor(nr, dtype=torch.long),
        "answer_idx": torch.tensor(ans_idx, dtype=torch.long),
    }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ───────────────────────── synthetic generator ─────────────────────────
def _reexec_and_label(trace):
    """Compute final answer for a generated trace; None if degenerate."""
    vv = {}
    for s in trace["steps"]:
        v1 = vv[s["arg1"]] if isinstance(s["arg1"], str) else float(s["arg1"])
        v2 = vv[s["arg2"]] if isinstance(s["arg2"], str) else float(s["arg2"])
        op = s["op"]
        if op == "add": rv = v1 + v2
        elif op == "sub": rv = v1 - v2
        elif op == "mul": rv = v1 * v2
        elif op == "div": rv = v1 / v2 if v2 != 0 else None
        else: rv = v1
        if rv is None or not np.isfinite(rv) or abs(rv) > MAX_ABS or rv != int(rv):
            return None
        vv[s["result"]] = rv
    return vv[trace["final_answer"]]


# op mix + step-count distribution approximating faithful GSM8K
_STEP_W = {1: 0.03, 2: 0.24, 3: 0.25, 4: 0.18, 5: 0.12, 6: 0.08, 7: 0.05, 8: 0.03, 9: 0.02}
_OPS = ["mul", "add", "sub", "div"]
_OP_W = [0.37, 0.29, 0.18, 0.16]


def gen_one(rng, max_steps=9):
    steps_choices = [k for k in _STEP_W if k <= max_steps]
    w = np.array([_STEP_W[k] for k in steps_choices]); w = w / w.sum()
    n_steps = int(rng.choice(steps_choices, p=w))
    pool = []                      # (token, value); token is int literal or "vN"
    n_leaf = rng.integers(2, 4)
    for _ in range(n_leaf):
        pool.append((int(rng.integers(1, 100)), None))
    steps = []
    for i in range(n_steps):
        for _try in range(8):
            op = str(rng.choice(_OPS, p=_OP_W))
            a, b = rng.integers(0, len(pool), size=2)
            ta, va = pool[a]; tb, vb = pool[b]
            av = va if va is not None else ta
            bv = vb if vb is not None else tb
            if op == "div":
                if bv == 0 or av % bv != 0:
                    continue
                rv = av // bv
            elif op == "add": rv = av + bv
            elif op == "sub": rv = av - bv
            else: rv = av * bv
            if abs(rv) > MAX_ABS:
                continue
            rk = f"v{i+1}"
            steps.append({"op": op, "arg1": ta, "arg2": tb, "result": rk})
            pool.append((rk, float(rv)))
            break
        else:
            return None
    if not steps:
        return None
    trace = {"steps": steps, "final_answer": steps[-1]["result"]}
    fa = _reexec_and_label(trace)
    if fa is None:
        return None
    return {"trace": trace, "target": float(fa), "question": ""}


def make_synthetic(n, seed=0, max_steps=9):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        r = gen_one(rng, max_steps=max_steps)
        if r is not None:
            out.append(r)
    return out


# ───────────────────────── dataset ─────────────────────────
def perturb_constants(trace, max_value=200, rng=None):
    rng = rng or random
    nt = copy.deepcopy(trace); vv = {}
    for s in nt["steps"]:
        for k in ("arg1", "arg2"):
            a = s[k]
            if isinstance(a, (int, float)) or (isinstance(a, str) and a not in vv and _isnum(a)):
                s[k] = float(rng.randint(1, max_value))
        v1 = vv[s["arg1"]] if isinstance(s["arg1"], str) and s["arg1"] in vv else float(s["arg1"])
        v2 = vv[s["arg2"]] if isinstance(s["arg2"], str) and s["arg2"] in vv else float(s["arg2"])
        op = s["op"]
        if op == "add": rv = v1 + v2
        elif op == "sub": rv = v1 - v2
        elif op == "mul": rv = v1 * v2
        elif op == "div": rv = v1 / v2 if v2 != 0 else None
        else: rv = v1
        if rv is None or not np.isfinite(rv) or abs(rv) > MAX_ABS:
            return None, None
        vv[s["result"]] = rv
    fa = vv.get(nt["final_answer"])
    if fa is None or abs(fa) > MAX_ABS:
        return None, None
    return nt, int(round(fa))


def _isnum(a):
    try: float(a); return True
    except Exception: return False


class GraphDataset(Dataset):
    """Holds faithful or synthetic records. Supports step-count curriculum via
    set_phase(max_steps) and optional number augmentation."""
    def __init__(self, records, max_nodes=40, augment=False, augment_p=0.3,
                 augment_max_value=200, mask_refs=True):
        self.max_nodes = max_nodes
        self.augment = augment
        self.augment_p = augment_p
        self.augment_max_value = augment_max_value
        self.mask_refs = mask_refs
        self.records = []
        for it in records:
            tr = it.get("trace", {})
            tgt = float(it.get("target", 0.0))
            if abs(tgt) > MAX_ABS:
                continue
            fa = tr.get("final_answer", "")
            if fa not in {s.get("result", "") for s in tr.get("steps", [])}:
                continue
            self.records.append((tr, tgt, len(tr.get("steps", []))))
        self.active = list(range(len(self.records)))

    def set_phase(self, max_steps):
        self.active = [i for i, (_, _, sc) in enumerate(self.records) if sc <= max_steps]
        return len(self.active)

    def __len__(self):
        return len(self.active)

    def __getitem__(self, idx):
        tr, tgt, _ = self.records[self.active[idx]]
        if self.augment and random.random() < self.augment_p:
            nt, ntgt = perturb_constants(tr, self.augment_max_value)
            if nt is not None:
                return sample_to_tensors(nt, ntgt, self.max_nodes, self.mask_refs)
        return sample_to_tensors(tr, tgt, self.max_nodes, self.mask_refs)


# ───────────────────────── model ─────────────────────────
class DenseGAT(nn.Module):
    def __init__(self, in_f, out_f, heads=4, concat=True, drop=0.1):
        super().__init__()
        self.heads, self.out_f, self.concat = heads, out_f, concat
        self.W = nn.Linear(in_f, heads * out_f, bias=False)
        self.as_ = nn.Linear(out_f, 1, bias=False)
        self.ad = nn.Linear(out_f, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x, adj):
        B, N, _ = x.shape
        xp = self.W(x).reshape(B, N, self.heads, self.out_f)
        s = self.as_(xp).squeeze(-1); d = self.ad(xp).squeeze(-1)
        e = F.leaky_relu(s.unsqueeze(2) + d.unsqueeze(1), 0.2)
        e = e.masked_fill(adj.unsqueeze(-1) == 0, -1e4)
        attn = self.drop(F.softmax(e, dim=2))
        h = torch.einsum("bnjh,bjhd->bnhd", attn, xp)
        return h.reshape(B, N, self.heads * self.out_f) if self.concat else h.mean(2)


class DigitAwareBridge(nn.Module):
    """Encodes each node from op + DIGIT-level operands + is_ref, then GAT."""
    def __init__(self, d, d_op=64, d_dig=32, gh=128, gl=3, heads=4):
        super().__init__()
        self.op_emb = nn.Embedding(OP_VOCAB_SIZE, d_op)
        self.dig_emb = nn.Embedding(DIGIT_VOCAB_SIZE, d_dig)
        self.opnd_proj = nn.Linear(MAX_DIGITS * d_dig, d)        # per operand
        self.fuse = nn.Linear(d_op + 2 * d + 2, d)
        self.gats = nn.ModuleList()
        ind, self.is_last = d, []
        for i in range(gl):
            out, co = (d, False) if i == gl - 1 else (gh, True)
            self.gats.append(DenseGAT(ind, out, heads=heads, concat=co))
            self.is_last.append(i == gl - 1)
            ind = out * (heads if co else 1)

    def forward(self, op_ids, opnd_digits, is_ref, adj):
        B, N, _, _ = opnd_digits.shape
        oe = self.op_emb(op_ids)                                  # (B,N,d_op)
        de = self.dig_emb(opnd_digits)                            # (B,N,2,MD,d_dig)
        de = de.reshape(B, N, 2, -1)                              # (B,N,2,MD*d_dig)
        opr = self.opnd_proj(de)                                 # (B,N,2,d)
        x = torch.cat([oe, opr[:, :, 0], opr[:, :, 1], is_ref], dim=-1)
        x = self.fuse(x)
        pad = (op_ids == 0)
        for layer, last in zip(self.gats, self.is_last):
            pm = pad.unsqueeze(2) | pad.unsqueeze(1)
            a2 = adj.clone(); a2[pm] = 0.0
            x = layer(x, a2)
            if not last:
                x = F.elu(x)
        return x


def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d, ex=2.0):
        super().__init__()
        i = int(round(ex * d * 2 / 3)); i = (i + 127) // 128 * 128
        self.gu = nn.Linear(d, i * 2, bias=False); self.dn = nn.Linear(i, d, bias=False)

    def forward(self, x):
        g, u = self.gu(x).chunk(2, dim=-1)
        return self.dn(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, d, h, ex=2.0):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.op = nn.Linear(d, d, bias=False); self.mlp = SwiGLU(d, ex)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.hd).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        if mask is not None: s = s + mask
        o = (F.softmax(s, -1) @ v).transpose(1, 2).reshape(B, N, D)
        x = rms_norm(x + self.op(o)); x = rms_norm(x + self.mlp(x))
        return x


class Module_(nn.Module):
    def __init__(self, nl, d, h, ex=2.0):
        super().__init__()
        self.layers = nn.ModuleList([Block(d, h, ex) for _ in range(nl)])

    def forward(self, hid, inj, mask=None):
        hid = hid + inj
        for l in self.layers: hid = l(hid, mask)
        return hid


class FaithfulHRM(nn.Module):
    def __init__(self, d=256, heads=8, Hc=3, Lc=4, Hl=4, Ll=4, ex=2.0, slen=40):
        super().__init__()
        self.Hc, self.Lc = Hc, Lc
        self.bridge = DigitAwareBridge(d)
        self.pos = nn.Embedding(slen, d)
        self.Hmod = Module_(Hl, d, heads, ex); self.Lmod = Module_(Ll, d, heads, ex)
        self.Hi = nn.Parameter(torch.randn(d) * 0.02)
        self.Li = nn.Parameter(torch.randn(d) * 0.02)
        self.dhead = nn.Linear(d, MAX_DIGITS * DIGIT_VOCAB_SIZE)
        self.qnorm = nn.LayerNorm(d)
        self.qhead = nn.Linear(d, 2)
        nn.init.zeros_(self.qhead.weight); self.qhead.bias.data.copy_(torch.tensor([-5., -5.]))

    def encode(self, b):
        x = self.bridge(b["op_ids"], b["opnd_digits"], b["is_ref"], b["adj"])
        N = b["op_ids"].shape[1]
        x = x + self.pos(torch.arange(N, device=x.device).unsqueeze(0))
        pad = (b["op_ids"] == 0)
        amask = pad.float().unsqueeze(1).unsqueeze(1) * -1e4
        return x, amask

    def init_carry(self, B, N, device):
        zH = self.Hi.view(1, 1, -1).expand(B, N, -1).contiguous()
        zL = self.Li.view(1, 1, -1).expand(B, N, -1).contiguous()
        return zH, zL

    def step(self, b, zH, zL):
        x, amask = self.encode(b)
        B, N, _ = x.shape
        with torch.no_grad():
            for h in range(self.Hc):
                for l in range(self.Lc):
                    if h == self.Hc - 1 and l == self.Lc - 1:
                        continue
                    zL = self.Lmod(zL, zH + x, amask)
                if h != self.Hc - 1:
                    zH = self.Hmod(zH, zL, amask)
        zL = self.Lmod(zL, zH + x, amask)
        zH = self.Hmod(zH, zL, amask)
        dl = self.dhead(zH).reshape(B, N, MAX_DIGITS, DIGIT_VOCAB_SIZE)
        ql = self.qhead(self.qnorm(zH[:, 0]))
        return dl, ql[:, 0], ql[:, 1], zH.detach(), zL.detach()

    def forward(self, b, max_steps=1):
        x, _ = self.encode(b)
        B, N, _ = x.shape
        zH, zL = self.init_carry(B, N, b["op_ids"].device)
        for _ in range(max_steps):
            dl, qh, qc, zH, zL = self.step(b, zH, zL)
        return dl, qh, qc


# ───────────────────────── loss / eval ─────────────────────────
def segment_loss(dl, final_tgt, node_tgts, num_real, qh, qc, next_q, aux_w, q_w, answer_idx):
    B, N, MD, V = dl.shape
    # main loss on the node that COMPUTES the answer
    idx = answer_idx.clamp(min=0)
    final_logits = dl[torch.arange(B), idx]                       # (B,MD,V)
    main = F.cross_entropy(final_logits.reshape(-1, V), final_tgt.reshape(-1),
                           ignore_index=DIGIT_VOCAB["PAD"])
    # aux loss on all real nodes
    node_mask = (torch.arange(N, device=dl.device).unsqueeze(0) < num_real.unsqueeze(1))
    aux_logits = dl.reshape(B * N, MD, V)
    aux_tgt = node_tgts.reshape(B * N, MD)
    aux_ce = F.cross_entropy(aux_logits.reshape(-1, V), aux_tgt.reshape(-1),
                             ignore_index=DIGIT_VOCAB["PAD"], reduction="none")
    aux_ce = aux_ce.reshape(B * N, MD).mean(1).reshape(B, N)
    aux = (aux_ce * node_mask).sum() / node_mask.sum().clamp(min=1)
    # ACT q-loss
    q = F.binary_cross_entropy_with_logits(qh, next_q) + \
        F.binary_cross_entropy_with_logits(qc, next_q)
    return main + aux_w * aux + q_w * q, main


@torch.no_grad()
def evaluate(model, loader, device, act_steps=1):
    model.eval()
    exact = total = digit_ok = digit_tot = no_out = 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        dl, _, _ = model(b, max_steps=act_steps)
        idx = b["answer_idx"].clamp(min=0)
        pred = dl[torch.arange(dl.shape[0]), idx].argmax(-1)        # (B,MD)
        for i in range(pred.shape[0]):
            p = decode_digits(pred[i].tolist())
            t = int(b["raw_target"][i])
            if p == -1: no_out += 1
            exact += int(p == t); total += 1
        gt = b["final_digit_tgt"]
        m = gt != DIGIT_VOCAB["PAD"]
        digit_ok += ((pred == gt) & m).sum().item(); digit_tot += m.sum().item()
    return {"exact": exact / max(total, 1), "digit": digit_ok / max(digit_tot, 1),
            "no_out": no_out, "n": total}
