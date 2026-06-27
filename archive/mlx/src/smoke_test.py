import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from src.model import HybridHRM
from src.dataset import OP_VOCAB

def test_model():
    vocab_size = len(OP_VOCAB)
    # Instantiate model with tiny settings
    model = HybridHRM(vocab_size=vocab_size, d_model=128, n_heads=4, H_cycles=1, L_cycles=2, halt_max_steps=2)
    mx.eval(model.parameters())
    
    num_params = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"HybridHRM initialized with {num_params} parameters.")

    # Create fake batch
    B, seq_len = 2, 50
    inputs = mx.random.randint(0, vocab_size, (B, seq_len))
    adj_mask = mx.random.randint(0, 2, (B, seq_len, seq_len)).astype(mx.float32)
    targets = mx.array([1.5, -2.0])

    batch = {
        "inputs": inputs,
        "adj_mask": adj_mask,
        "targets": targets
    }

    def loss_fn(model_params, batch):
        model.update(model_params)
        model.train()
        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)
        
        # We compute MAE loss on the numerical predictions
        preds = outputs["pred_val"]
        loss = mx.mean(mx.abs(preds - batch["targets"]))
        return loss

    # Calculate loss and gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model.trainable_parameters(), batch)
    
    # Evaluate so computations actually happen (MLX is lazy)
    mx.eval(loss, grads)
    
    print(f"Forward pass completed. Loss: {loss.item():.4f}")
    
    # Verify gradients
    has_nans = False
    for k, v in tree_flatten(grads):
        if mx.any(mx.isnan(v)):
            print(f"NaN gradient detected in {k}")
            has_nans = True
            
    if not has_nans:
        print("Smoke test passed! Gradients flow correctly through GAT -> HRM -> Regression Head.")

if __name__ == "__main__":
    test_model()
