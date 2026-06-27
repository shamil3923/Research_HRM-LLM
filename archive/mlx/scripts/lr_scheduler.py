"""
Learning rate scheduling for HRM training
Implements cosine decay with warmup as used in original PyTorch implementation
"""

import math
from typing import Optional


class CosineScheduleWithWarmup:
    """
    Cosine learning rate schedule with linear warmup
    
    Matches the original PyTorch HRM implementation:
    - Linear warmup for warmup_steps
    - Cosine decay from base_lr to min_lr after warmup
    """
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        num_cycles: float = 0.5
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.num_cycles = num_cycles
        self.min_lr = base_lr * min_lr_ratio
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step"""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * float(step) / float(max(1, self.warmup_steps))
        
        # Cosine decay after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress))
        
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor
    
    def update_optimizer_lr(self, optimizer, step: int):
        """Update optimizer learning rate"""
        new_lr = self.get_lr(step)
        optimizer.learning_rate = new_lr
        return new_lr


def create_lr_scheduler(
    base_lr: float = 1e-4,
    warmup_steps: int = 2000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1
) -> CosineScheduleWithWarmup:
    """Create learning rate scheduler with HRM defaults"""
    return CosineScheduleWithWarmup(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio
    )