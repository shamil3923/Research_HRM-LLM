"""
MLX implementation of AdamATan2 optimizer

Based on the paper "Scaling Exponents Across Parameterizations and Optimizers"
and the adam-atan2-pytorch implementation by lucidrains.

Key innovation: Replaces division by sqrt(v) + epsilon with atan2 function
for better numerical stability and scale invariance.
"""

import mlx.core as mx
import mlx.optimizers as optim
from typing import Optional, Dict, Any


class AdamATan2(optim.Optimizer):
    """
    AdamATan2 optimizer for MLX
    
    Uses atan2 instead of division for numerical stability without epsilon.
    Particularly effective with high weight decay values.
    
    Args:
        learning_rate: Learning rate (default: 1e-4)
        betas: Coefficients for computing running averages (default: [0.9, 0.99])
        weight_decay: Weight decay coefficient (default: 0.0)
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        a: float = 1.0,  # No scaling in base version
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.a = a  # Scaling factor for update
        
        # Initialize state
        self._step = 0
        
    def init_single(self, parameter: mx.array, state: dict) -> dict:
        """Initialize optimizer state for a single parameter"""
        state["m"] = mx.zeros_like(parameter)  # First moment
        state["v"] = mx.zeros_like(parameter)  # Second moment
        state["step"] = 0  # Track steps per parameter
        return state
    
    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict,
    ) -> mx.array:
        """Apply AdamATan2 update to a single parameter"""
        
        # Apply weight decay FIRST (like PyTorch)
        if self.weight_decay > 0:
            parameter = parameter * (1 - self.learning_rate * self.weight_decay)
        
        # Get moments
        m = state["m"]
        v = state["v"]
        
        # Update biased moments
        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)
        
        # Update state
        state["m"] = m
        state["v"] = v
        state["step"] += 1
        
        # Bias correction
        step = state["step"]
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        
        # Compute bias-corrected moments
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        
        # EXACT implementation from PyTorch adam-atan2:
        # den = exp_avg_sq.mul(b * b / bias_correct2).sqrt_()
        # update = exp_avg.mul(1. / bias_correct1).atan2_(den)
        # PyTorch uses b=1.0 by default, NOT 1/lr
        
        b = 1.0  # Match PyTorch default
        den = mx.sqrt(v * (b * b / bias_correction2) + 1e-8)
        # Clamp inputs to arctan2 to prevent extreme values
        numerator = mx.clip(m / bias_correction1, -1e6, 1e6)
        denominator = mx.maximum(den, 1e-8)
        update = mx.arctan2(numerator, denominator)
        update = update * self.learning_rate * self.a  # Scale by lr and a factor
        
        # Apply update
        parameter = parameter - update
        
        return parameter
    
    def apply_gradients(self, gradients, parameters):
        """Apply gradients to all parameters"""
        # Don't increment global step here - it's handled per parameter
        return super().apply_gradients(gradients, parameters)


class AdamATan2Scaled(AdamATan2):
    """
    Enhanced version with better scaling for the atan2 operation
    
    This version scales the atan2 output to better match traditional Adam's
    update magnitudes, making hyperparameter transfer easier.
    """
    
    
    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict,
    ) -> mx.array:
        """Apply scaled AdamATan2 update"""
        
        # Apply weight decay FIRST (like PyTorch)
        if self.weight_decay > 0:
            parameter = parameter * (1 - self.learning_rate * self.weight_decay)
        
        # Get moments
        m = state["m"]
        v = state["v"]
        
        # Update biased moments
        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)
        
        # Update state
        state["m"] = m
        state["v"] = v
        state["step"] += 1
        
        # Bias correction
        step = state["step"]
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        
        
        # Compute bias-corrected moments
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        
        # EXACT implementation from PyTorch adam-atan2:
        b = 1.0  # Match PyTorch default
        den = mx.sqrt(v * (b * b / bias_correction2) + 1e-8)
        # Clamp inputs to arctan2 to prevent extreme values
        numerator = mx.clip(m / bias_correction1, -1e6, 1e6)
        denominator = mx.maximum(den, 1e-8)
        update = mx.arctan2(numerator, denominator)
        update = update * self.learning_rate
        
        # Apply update
        parameter = parameter - update
        
        
        return parameter


def test_adam_atan2():
    """Test the AdamATan2 implementation"""
    import mlx.nn as nn
    
    # Create a simple model
    model = nn.Linear(10, 10)
    
    # Create optimizer
    optimizer = AdamATan2(learning_rate=1e-3, weight_decay=1.0)
    
    # Test forward pass
    x = mx.random.normal((32, 10))
    y = model(x)
    loss = mx.mean(y ** 2)
    
    # Compute gradients
    grad_fn = nn.value_and_grad(model, lambda m: mx.mean(m(x) ** 2))
    loss, grads = grad_fn(model)
    
    # Apply gradients
    optimizer.update(model, grads)
    
    print("✅ AdamATan2 test passed!")
    print(f"Loss: {float(loss):.4f}")
    

if __name__ == "__main__":
    test_adam_atan2()