"""Test the exact AdamATan2 implementation"""
import mlx.core as mx
import mlx.nn as nn
from mlx_adam_atan2_exact import AdamATan2Exact

# Simple test
model = nn.Linear(512, 512)

optimizer = AdamATan2Exact(
    learning_rate=1e-4,
    betas=(0.9, 0.99),  # Original PyTorch defaults
    weight_decay=1.0,
    a=1.27,  # PyTorch default
    b=1.0    # PyTorch default
)
optimizer.init(model.trainable_parameters())

print("Testing exact PyTorch AdamATan2 port...")

for step in range(100):
    x = mx.random.normal((256, 512))
    
    def loss_fn(model):
        y = model(x)
        return mx.mean(y ** 2) * 5000.0  # HRM-like loss magnitude
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss_val, grads = loss_and_grad_fn(model)
    
    loss_float = float(loss_val)
    
    if step % 10 == 0:
        weight_norm = float(mx.sqrt(mx.sum(model.weight ** 2)))
        grad_norm = float(mx.sqrt(mx.sum(grads['weight'] ** 2)))
        print(f"Step {step:3d}: loss={loss_float:8.2f}, weight_norm={weight_norm:.4f}, grad_norm={grad_norm:.4f}")
    
    if mx.any(mx.isnan(loss_val)):
        print(f"❌ NaN at step {step}!")
        break
    
    optimizer.update(model, grads)

print("Test completed!")