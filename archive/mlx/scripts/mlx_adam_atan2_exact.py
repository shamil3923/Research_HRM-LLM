"""
Exact MLX port of PyTorch AdamATan2 optimizer
Based on the original adam-atan2-pytorch implementation
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Optimizer
from typing import Dict, Any


class AdamATan2Exact(Optimizer):
    """
    Exact MLX port of PyTorch AdamATan2 optimizer
    
    This matches the original implementation exactly:
    - Weight decay applied before momentum updates
    - Exact mathematical formulation: atan2(exp_avg / bias_correct1, sqrt(exp_avg_sq * b² / bias_correct2))
    - Parameters: a=1.27, b=1.0 (PyTorch defaults)
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        a: float = 1.27,  # Scaling factor - PyTorch default
        b: float = 1.0,   # Denominator factor - PyTorch default
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.a = a
        self.b = b
    
    def init_single(self, parameter: mx.array, state: dict) -> dict:
        """Initialize optimizer state for a single parameter"""
        return {
            "exp_avg": mx.zeros_like(parameter),
            "exp_avg_sq": mx.zeros_like(parameter), 
            "steps": 0
        }
    
    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict,
    ) -> mx.array:
        """Apply AdamATan2 update to a single parameter - EXACT PyTorch match"""
        
        # Weight decay FIRST (line 74 in original)
        if self.weight_decay > 0:
            parameter = parameter * (1 - self.learning_rate * self.weight_decay)
        
        # Initialize state if needed (lines 84-90 in original)
        if "exp_avg" not in state:
            state["exp_avg"] = mx.zeros_like(parameter)
            state["exp_avg_sq"] = mx.zeros_like(parameter)
            state["steps"] = 0
        
        # Get state
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        steps = state["steps"]
        
        # Increment steps (line 96 in original)
        steps += 1
        
        # Bias corrections (lines 100-101 in original)
        bias_correct1 = 1.0 - self.beta1 ** steps
        bias_correct2 = 1.0 - self.beta2 ** steps
        
        # Update biased moments (lines 105-106 in original)
        # exp_avg.lerp_(grad, 1. - beta1) equivalent to:
        exp_avg = exp_avg * self.beta1 + gradient * (1 - self.beta1)
        # exp_avg_sq.lerp_(grad * grad, 1. - beta2) equivalent to:
        exp_avg_sq = exp_avg_sq * self.beta2 + (gradient * gradient) * (1 - self.beta2)
        
        # Update state
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq
        state["steps"] = steps
        
        # EXACT computation from lines 112-113 in original:
        # den = exp_avg_sq.mul(b * b / bias_correct2).sqrt_()
        # update = exp_avg.mul(1. / bias_correct1).atan2_(den)
        den = mx.sqrt(exp_avg_sq * (self.b * self.b / bias_correct2))
        numerator = exp_avg * (1.0 / bias_correct1)
        update = mx.arctan2(numerator, den)
        
        # Final parameter update (line 124 in original):
        # p.add_(update, alpha = -lr * a)
        parameter = parameter - update * (self.learning_rate * self.a)
        
        return parameter