import torch
import os
import sys

# Ensure kaggle directory is in path to import the model
sys.path.append(os.path.abspath('kaggle'))
from hrm_gsm8k_pytorch import HRMForMath, OP_VOCAB

print("Initializing Graph-Aware HRM model (dummy weights)...")
model = HRMForMath(
    vocab_size=len(OP_VOCAB), d_model=512, n_heads=8,
    H_cycles=4, L_cycles=8, H_layers=8, L_layers=8, seq_len=50,
)

os.makedirs('checkpoints/gsm8k', exist_ok=True)
dummy_path = 'checkpoints/gsm8k/dummy_graph_model.pt'
torch.save(model.state_dict(), dummy_path)
print(f"Saved dummy PyTorch checkpoint to {dummy_path}")
