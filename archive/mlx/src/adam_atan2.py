"""
Adam-atan2 optimizer (PyTorch port of HRM's reference implementation).

Replaces the divisive Adam update
    delta = -lr * m_hat / (sqrt(v_hat) + eps)
with an arctangent update that is bounded and never divides by ~0:
    delta = -lr * a * atan2(m_hat, sqrt(v_hat) * b)

Decoupled weight decay is applied BEFORE the moment-based update (per the
HRM reference). This matches §3.6 of the research report.

Defaults a=1.27, b=1.0, betas=(0.9, 0.95) follow the HRM Sudoku config.
"""
from __future__ import annotations

import math
import torch
from torch.optim.optimizer import Optimizer


class AdamATan2(Optimizer):
    """Adam with atan2 update rule + decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        a: float = 1.27,
        b: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            a = group["a"]
            b = group["b"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamATan2 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # 1) Decoupled weight decay BEFORE the update
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # 2) Update biased moments
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3) Bias-corrected estimates
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                m_hat = m / bc1
                v_hat = v / bc2

                # 4) atan2 update (no epsilon, never divides by ~0)
                delta = torch.atan2(m_hat, v_hat.sqrt() * b)
                p.add_(delta, alpha=-lr * a)

        return loss


if __name__ == "__main__":
    # Quick smoke test: minimize x^2
    x = torch.nn.Parameter(torch.tensor([3.0, -2.0, 5.0]))
    opt = AdamATan2([x], lr=0.1)
    for i in range(200):
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        opt.step()
    print(f"After 200 steps: x = {x.detach().tolist()}, loss = {loss.item():.6f}")
    assert all(abs(v) < 0.5 for v in x.detach().tolist()), "AdamATan2 did not converge"
    print("AdamATan2 OK.")
