"""
Optimizer utilities for HRM-MLX GSM8K training.

Provides a thin wrapper around mlx.optimizers.Adam with:
- Gradient clipping by global norm
- Parameter group support (different LRs for bridge vs transformer)
- Step counting for LR schedule callbacks
"""

import mlx.core as mx
import mlx.optimizers as mlx_opt
from mlx.utils import tree_flatten, tree_map
import math


def clip_grad_norm(grads, max_norm: float = 0.5):
    """
    Clip gradients by global L2 norm.
    Returns (clipped_grads, grad_norm_before_clipping).
    """
    flat = tree_flatten(grads)
    total_sq = sum(float(mx.sum(g * g).item()) for _, g in flat if g is not None)
    total_norm = math.sqrt(total_sq)
    clip_coeff = max_norm / (total_norm + 1e-8)
    if clip_coeff < 1.0:
        grads = tree_map(lambda g: g * clip_coeff if g is not None else g, grads)
    return grads, total_norm


class WarmupFlatSchedule:
    """
    Linear warmup for `warmup_steps` steps, then holds at `peak_lr`.
    Call `restart(current_step)` at curriculum phase boundaries for fresh warmup.
    """
    def __init__(self, peak_lr: float, warmup_steps: int):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self._offset = 0

    def restart(self, current_step: int):
        self._offset = current_step

    def __call__(self, step: int) -> float:
        local = step - self._offset
        if local < self.warmup_steps:
            return self.peak_lr * max(local, 1) / max(self.warmup_steps, 1)
        return self.peak_lr


class WarmupCosineSchedule:
    """
    Linear warmup then cosine decay.
    Call `restart(current_step)` at curriculum phase boundaries.
    """
    def __init__(self, peak_lr: float, warmup_steps: int, cycle_steps: int, min_lr: float = 1e-5):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.min_lr = min_lr
        self._offset = 0

    def restart(self, current_step: int):
        self._offset = current_step

    def __call__(self, step: int) -> float:
        local = step - self._offset
        if local < self.warmup_steps:
            return self.peak_lr * max(local, 1) / max(self.warmup_steps, 1)
        progress = min((local - self.warmup_steps) / max(self.cycle_steps - self.warmup_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine


class TrainingOptimizer:
    """
    Wrapper combining Adam optimizer + LR schedule + gradient clipping.
    Tracks step count internally.
    """
    def __init__(
        self,
        peak_lr: float = 3e-4,
        warmup_steps: int = 20,
        max_grad_norm: float = 0.5,
        betas: list = None,
        schedule: str = "flat",  # "flat" or "cosine"
        cycle_steps: int = 300,
        min_lr: float = 1e-5,
    ):
        if betas is None:
            betas = [0.9, 0.95]

        self.peak_lr = peak_lr
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

        self._optimizer = mlx_opt.Adam(learning_rate=peak_lr, betas=betas)

        if schedule == "flat":
            self._schedule = WarmupFlatSchedule(peak_lr, warmup_steps)
        elif schedule == "cosine":
            self._schedule = WarmupCosineSchedule(peak_lr, warmup_steps, cycle_steps, min_lr)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def restart_lr(self):
        """Restart LR schedule at current step (call at phase boundaries)."""
        self._schedule.restart(self.step_count)

    @property
    def current_lr(self) -> float:
        return self._schedule(self.step_count)

    def step(self, model, grads):
        """
        Apply one optimizer step:
          1. Clip gradients
          2. Update LR from schedule
          3. Apply Adam update
        Returns grad_norm (before clipping) for logging.
        """
        # Clip
        clipped_grads, grad_norm = clip_grad_norm(grads, self.max_grad_norm)

        # Schedule
        lr = self._schedule(self.step_count)
        self._optimizer.learning_rate = lr
        self.step_count += 1

        # Apply
        self._optimizer.update(model, clipped_grads)

        return grad_norm
