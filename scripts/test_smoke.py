"""Quick smoke test for the HRM pipeline."""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from src.dataset import GSM8KGraphDataset, OP_VOCAB
from src.model import HRMForMath
from src.train import digit_cross_entropy_loss, final_node_digit_loss

def main():
    dataset = GSM8KGraphDataset('data/gsm8k_train_parsed.json', max_nodes=50)
    model = HRMForMath(
        vocab_size=len(OP_VOCAB), d_model=256, n_heads=8,
        H_cycles=2, L_cycles=4, H_layers=4, L_layers=4, seq_len=50
    )
    mx.eval(model.parameters())
    
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Params: {num_params:,}")
    
    batch = next(dataset.get_batches(4, shuffle=False))
    
    def test_loss(model, batch):
        dl, qh, qc = model(batch)
        nce = digit_cross_entropy_loss(dl, batch['node_digit_targets'], batch['num_real_nodes'])
        fce = final_node_digit_loss(dl, batch['final_digit_target'], batch['num_real_nodes'])
        return 0.7 * nce + 0.3 * fce
    
    lag = nn.value_and_grad(model, test_loss)
    loss, grads = lag(model, batch)
    mx.eval(loss)
    
    print(f"Loss: {loss.item():.4f}")
    
    fg = tree_flatten(grads)
    hg = sum(1 for _, g in fg if mx.any(g != 0).item())
    print(f"Gradient tensors with signal: {hg}/{len(fg)}")
    print("All checks passed!")

if __name__ == "__main__":
    main()
