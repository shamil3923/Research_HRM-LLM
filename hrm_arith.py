"""
HRM that LEARNS arithmetic (no symbolic executor).

Stage 1 — digit-aligned, reversed, place-value representation + H/L recurrence.
Position-local ops (add, sub) become learnable to ~100%; this is the reusable
arithmetic core for Stage 2 (multiplication-by-algorithm) and Stage 3 (chaining).

Representation (the fix that makes arithmetic learnable):
  - operands written LSB-first (reversed) so carries propagate left->right
  - digit p of arg1 is ALIGNED with digit p of arg2 at sequence position p
  - shared positional embedding gives place value
The model predicts the result digit at each position; we also predict a sign bit.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIG = 10                       # digit classes 0-9
OPS = {"add": 0, "sub": 1, "mul": 2}


# ----- reversed fixed-width digit helpers -----
def to_rev(n, width):
    s = [int(c) for c in str(abs(int(n)))][::-1]
    return s[:width] + [0] * (width - len(s))


def from_rev(digs):
    return int("".join(str(int(d)) for d in digs[::-1]) or "0")


# ----- HRM-style core over the digit sequence -----
def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d, ex=2.0):
        super().__init__()
        i = int(round(ex * d * 2 / 3)); i = (i + 63) // 64 * 64
        self.gu = nn.Linear(d, 2 * i, bias=False); self.dn = nn.Linear(i, d, bias=False)
    def forward(self, x):
        g, u = self.gu(x).chunk(2, -1)
        return self.dn(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.mlp = SwiGLU(d)
    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.hd).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        o = (F.softmax(a, -1) @ v).transpose(1, 2).reshape(B, N, D)
        x = rms_norm(x + self.o(o)); x = rms_norm(x + self.mlp(x))
        return x


class Stack(nn.Module):
    def __init__(self, nl, d, h):
        super().__init__()
        self.layers = nn.ModuleList([Block(d, h) for _ in range(nl)])
    def forward(self, x, inj):
        x = x + inj
        for l in self.layers: x = l(x)
        return x


class ArithHRM(nn.Module):
    """H/L recurrence over L digit positions. Predicts result digits + sign."""
    def __init__(self, width, d=128, h=4, Hl=2, Ll=2, Hc=2, Lc=3):
        super().__init__()
        self.W, self.Hc, self.Lc = width, Hc, Lc
        self.ea = nn.Embedding(DIG, d); self.eb = nn.Embedding(DIG, d)
        self.pos = nn.Embedding(width, d); self.eop = nn.Embedding(len(OPS), d)
        self.H = Stack(Hl, d, h); self.L = Stack(Ll, d, h)
        self.Hi = nn.Parameter(torch.randn(d) * 0.02)
        self.Li = nn.Parameter(torch.randn(d) * 0.02)
        self.dhead = nn.Linear(d, DIG)
        self.shead = nn.Linear(d, 2)          # sign (read from pos 0 pooled)

    def forward(self, da, db, op):
        B = da.shape[0]
        pos = torch.arange(self.W, device=da.device).unsqueeze(0)
        x = self.ea(da) + self.eb(db) + self.pos(pos) + self.eop(op).unsqueeze(1)
        zH = self.Hi.view(1, 1, -1).expand(B, self.W, -1).contiguous()
        zL = self.Li.view(1, 1, -1).expand(B, self.W, -1).contiguous()
        with torch.no_grad():
            for hcyc in range(self.Hc):
                for lcyc in range(self.Lc):
                    if hcyc == self.Hc - 1 and lcyc == self.Lc - 1:
                        continue
                    zL = self.L(zL, zH + x)
                if hcyc != self.Hc - 1:
                    zH = self.H(zH, zL)
        zL = self.L(zL, zH + x)
        zH = self.H(zH, zL)
        return self.dhead(zH), self.shead(zH.mean(1))      # (B,W,DIG), (B,2)


# ----- data: position-local ops with digit-count curriculum -----
def gen_batch(bs, op, max_digits, width, rng):
    lo, hi = 0, 10 ** max_digits
    a = rng.integers(lo, hi, bs); b = rng.integers(lo, hi, bs)
    if op == "add":
        r = a + b
    elif op == "sub":
        a, b = np.maximum(a, b), np.minimum(a, b)   # a>=b -> non-negative result
        r = a - b
    else:
        r = a * b
    da = np.stack([to_rev(x, width) for x in a])
    db = np.stack([to_rev(x, width) for x in b])
    dr = np.stack([to_rev(x, width) for x in r])
    sign = (r < 0).astype(np.int64)
    return (torch.tensor(da), torch.tensor(db),
            torch.full((bs,), OPS[op]), torch.tensor(dr), torch.tensor(sign),
            torch.tensor(r))


@torch.no_grad()
def evaluate(model, op, max_digits, width, rng, n=2000):
    model.eval()
    dev = next(model.parameters()).device
    da, db, o, dr, sg, r = gen_batch(n, op, max_digits, width, rng)
    logits, slog = model(da.to(dev), db.to(dev), o.to(dev))
    pred = logits.argmax(-1).cpu(); ps = slog.argmax(-1).cpu()
    val = torch.tensor([from_rev(pred[i].tolist()) for i in range(n)])
    val = torch.where(ps.bool(), -val, val)
    exact = (val == r).float().mean().item()
    digit = (pred == dr).float().mean().item()
    return exact, digit


# ===================== Stage 2 — learned long multiplication =====================
class MulHRM(nn.Module):
    """Recurrent accumulator. One segment per digit of b:
         acc_out = acc_in + (a << j) * b_j        (single-digit multiply + add)
    The shift (column placement) is bookkeeping; the ARITHMETIC (single-digit
    multiply with carry, then add) is what the network learns."""
    def __init__(self, W, d=160, h=4, Hl=2, Ll=2, Hc=2, Lc=3):
        super().__init__()
        self.W, self.Hc, self.Lc = W, Hc, Lc
        self.ea = nn.Embedding(DIG, d)            # shifted-a digit at position p
        self.eacc = nn.Embedding(DIG, d)          # accumulator digit at position p
        self.pos = nn.Embedding(W, d)
        self.ebj = nn.Embedding(DIG, d)           # the single digit b_j
        self.H = Stack(Hl, d, h); self.L = Stack(Ll, d, h)
        self.Hi = nn.Parameter(torch.randn(d) * 0.02)
        self.Li = nn.Parameter(torch.randn(d) * 0.02)
        self.head = nn.Linear(d, DIG)

    def forward(self, a_shift, acc_in, bj):
        B = a_shift.shape[0]
        pos = torch.arange(self.W, device=a_shift.device).unsqueeze(0)
        x = self.ea(a_shift) + self.eacc(acc_in) + self.pos(pos) + self.ebj(bj).unsqueeze(1)
        zH = self.Hi.view(1, 1, -1).expand(B, self.W, -1).contiguous()
        zL = self.Li.view(1, 1, -1).expand(B, self.W, -1).contiguous()
        with torch.no_grad():
            for hc in range(self.Hc):
                for lc in range(self.Lc):
                    if hc == self.Hc - 1 and lc == self.Lc - 1: continue
                    zL = self.L(zL, zH + x)
                if hc != self.Hc - 1: zH = self.H(zH, zL)
        zL = self.L(zL, zH + x); zH = self.H(zH, zL)
        return self.head(zH)


def shift_rev(n, j, W):
    return to_rev(int(n) * (10 ** j), W)          # a << j, as reversed digits


def gen_mul_train(bs, Da, Db, W, rng):
    """Flattened per-segment teacher-forced training tuples."""
    a = rng.integers(0, 10 ** Da, bs); b = rng.integers(0, 10 ** Db, bs)
    A_, ACC, BJ, TGT = [], [], [], []
    for ai, bi in zip(a, b):
        bd = to_rev(bi, Db); rp = 0
        for j in range(Db):
            A_.append(shift_rev(ai, j, W))
            ACC.append(to_rev(rp, W))
            BJ.append(bd[j])
            rp = rp + ai * bd[j] * (10 ** j)
            TGT.append(to_rev(rp, W))
    return (torch.tensor(np.array(A_)), torch.tensor(np.array(ACC)),
            torch.tensor(BJ), torch.tensor(np.array(TGT)))


@torch.no_grad()
def eval_mul(model, Da, Db, W, rng, n=2000):
    model.eval()
    a = rng.integers(0, 10 ** Da, n); b = rng.integers(0, 10 ** Db, n)
    acc = np.zeros((n, W), dtype=np.int64)
    dev = next(model.parameters()).device
    for j in range(Db):
        bj = np.array([to_rev(bi, Db)[j] for bi in b])
        ash = np.stack([shift_rev(ai, j, W) for ai in a])
        out = model(torch.tensor(ash).to(dev), torch.tensor(acc).to(dev),
                    torch.tensor(bj).to(dev))
        acc = out.argmax(-1).cpu().numpy()                 # rollout: model's own acc
    pred = np.array([from_rev(acc[i].tolist()) for i in range(n)])
    return float((pred == a * b).mean())


# ===================== Stage 3 — multi-operation chaining =====================
@torch.no_grad()
def mul_one(model, da, db, W):
    """Multiply two reversed-digit operands via the learned rollout. da,db: lists len W."""
    dev = next(model.parameters()).device
    acc = [0] * W
    for j in range(W):
        if db[j] == 0:
            continue
        ash = shift_rev(from_rev(da), j, W)
        out = model(torch.tensor([ash]).to(dev), torch.tensor([acc]).to(dev),
                    torch.tensor([db[j]]).to(dev))
        acc = out.argmax(-1)[0].cpu().tolist()
    return acc


@torch.no_grad()
def chain_execute(arith, mul, div, trace, W):
    """Execute a faithful trace using ONLY the learned arithmetic (add/sub via
    ArithHRM, mul via MulHRM, div via DivHRM). References read the model's OWN
    previously-computed results, so errors compound. Returns None on a
    non-integer-representable step (e.g. div by 0 or div is None)."""
    dev = next(arith.parameters()).device
    reg = {}
    def opnd(a):
        if isinstance(a, str) and a in reg:
            return reg[a]
        return to_rev(int(round(float(a))), W)
    last = None
    for s in trace["steps"]:
        d1, d2, op = opnd(s.get("arg1", 0)), opnd(s.get("arg2", 0)), s.get("op")
        if op in ("add", "sub"):
            if op == "sub" and from_rev(d1) < from_rev(d2):
                d1, d2 = d2, d1                      # trained on a>=b
            logits, _ = arith(torch.tensor([d1]).to(dev), torch.tensor([d2]).to(dev),
                              torch.tensor([OPS[op]]).to(dev))
            res = logits.argmax(-1)[0].cpu().tolist()
        elif op == "mul":
            res = mul_one(mul, d1, d2, W)
        elif op == "div":
            if div is None or from_rev(d2) == 0:
                return None
            q, _ = div_one(div, from_rev(d1), from_rev(d2), W)   # quotient (GSM8K div is exact)
            res = to_rev(q, W)
        else:
            return None
        reg[s.get("result", "")] = res
        last = res
    fa = trace.get("final_answer", "")
    return from_rev(reg.get(fa, last))


def gen_chain(bs, max_steps, max_operand, W, rng, ops=("add", "sub", "mul", "div")):
    """Synthetic multi-step traces over add/sub/mul/div with exact integer division
    and intermediates kept within W digits. Supports deep chains (set max_steps)."""
    out = []
    while len(out) < bs:
        ns = int(rng.integers(1, max_steps + 1))
        vals = [int(rng.integers(1, max_operand)) for _ in range(2)]
        steps, ok = [], True
        for i in range(ns):
            placed = False
            for _try in range(10):
                op = str(rng.choice(ops))
                ai, bi = int(rng.integers(0, len(vals))), int(rng.integers(0, len(vals)))
                a, b = vals[ai], vals[bi]
                if op == "sub":
                    if a < b: ai, bi, a, b = bi, ai, b, a   # keep a>=b
                    r = a - b
                elif op == "add":
                    r = a + b
                elif op == "mul":
                    r = a * b
                else:  # div: only exact integer division
                    if b == 0 or a % b != 0:
                        continue
                    r = a // b
                if r < 0 or r >= 10 ** W:
                    continue
                placed = True; break
            if not placed:
                ok = False; break
            arg1 = f"v{ai}" if (rng.random() < 0.6 and ai >= 2) else vals[ai]
            arg2 = f"v{bi}" if (rng.random() < 0.6 and bi >= 2) else vals[bi]
            steps.append({"op": op, "arg1": arg1, "arg2": arg2, "result": f"v{len(vals)}"})
            vals.append(r)
        if not ok or not steps:
            continue
        out.append({"trace": {"steps": steps, "final_answer": f"v{len(vals)-1}"},
                    "target": vals[-1], "nsteps": len(steps)})
    return out


# ===================== Stage 4 — learned long division =====================
class DivHRM(nn.Module):
    """Long division, one segment per dividend digit (MSB-first):
         cur = rem_in*10 + brought_digit ;  qd = cur // divisor ;  rem_out = cur % divisor
    Learns the single-quotient-digit division (how many times the divisor fits,
    0-9) + the new remainder. Divisor & remainder are reversed-digit vectors."""
    def __init__(self, W, d=192, h=6, Hl=2, Ll=2, Hc=2, Lc=3):
        super().__init__()
        self.W, self.Hc, self.Lc = W, Hc, Lc
        self.ediv = nn.Embedding(DIG, d)          # divisor digit at position p
        self.erem = nn.Embedding(DIG, d)          # remainder-in digit at position p
        self.pos = nn.Embedding(W, d)
        self.ebrought = nn.Embedding(DIG, d)      # the brought-down dividend digit
        self.H = Stack(Hl, d, h); self.L = Stack(Ll, d, h)
        self.Hi = nn.Parameter(torch.randn(d) * 0.02)
        self.Li = nn.Parameter(torch.randn(d) * 0.02)
        self.rem_head = nn.Linear(d, DIG)         # new remainder digits (per position)
        self.q_head = nn.Linear(d, DIG)           # quotient digit (pooled)

    def forward(self, divisor, rem_in, brought):
        B = divisor.shape[0]
        pos = torch.arange(self.W, device=divisor.device).unsqueeze(0)
        x = self.ediv(divisor) + self.erem(rem_in) + self.pos(pos) + self.ebrought(brought).unsqueeze(1)
        zH = self.Hi.view(1, 1, -1).expand(B, self.W, -1).contiguous()
        zL = self.Li.view(1, 1, -1).expand(B, self.W, -1).contiguous()
        with torch.no_grad():
            for hc in range(self.Hc):
                for lc in range(self.Lc):
                    if hc == self.Hc - 1 and lc == self.Lc - 1: continue
                    zL = self.L(zL, zH + x)
                if hc != self.Hc - 1: zH = self.H(zH, zL)
        zL = self.L(zL, zH + x); zH = self.H(zH, zL)
        return self.q_head(zH.mean(1)), self.rem_head(zH)     # (B,DIG), (B,W,DIG)


def gen_div_train(bs, Dq, Dd, W, rng):
    """Teacher-forced per-step tuples for long division.
    Dd = max divisor digits, Dq drives dividend size (dividend ~ Dq+Dd digits)."""
    DIV, REMIN, BR, QD, REMOUT = [], [], [], [], []
    made = 0
    while made < bs:
        b = int(rng.integers(1, 10 ** Dd))
        if b == 0: continue
        a = int(rng.integers(0, 10 ** (Dq + Dd)))
        rem = 0
        for ch in str(a):                          # MSB-first
            cur = rem * 10 + int(ch)
            qd = cur // b
            rem_out = cur - qd * b
            DIV.append(to_rev(b, W)); REMIN.append(to_rev(rem, W))
            BR.append(int(ch)); QD.append(qd); REMOUT.append(to_rev(rem_out, W))
            rem = rem_out
        made += 1
    return (torch.tensor(np.array(DIV)), torch.tensor(np.array(REMIN)),
            torch.tensor(BR), torch.tensor(QD), torch.tensor(np.array(REMOUT)))


@torch.no_grad()
def div_one(model, a_int, b_int, W):
    """Rollout long division -> (quotient, remainder) using the model's own state."""
    dev = next(model.parameters()).device
    divisor = torch.tensor([to_rev(b_int, W)]).to(dev)
    rem = to_rev(0, W); q = []
    for ch in str(int(a_int)):
        qd_l, rem_l = model(divisor, torch.tensor([rem]).to(dev),
                            torch.tensor([int(ch)]).to(dev))
        qd = int(qd_l.argmax(-1)[0]); rem = rem_l.argmax(-1)[0].cpu().tolist()
        q.append(qd)
    quotient = int("".join(str(d) for d in q) or "0")
    return quotient, from_rev(rem)


@torch.no_grad()
def eval_div(model, Dq, Dd, W, rng, n=1500):
    model.eval()
    okq = okr = 0
    for _ in range(n):
        b = int(rng.integers(1, 10 ** Dd)); a = int(rng.integers(0, 10 ** (Dq + Dd)))
        q, r = div_one(model, a, b, W)
        okq += int(q == a // b); okr += int(r == a % b)
    return okq / n, okr / n
