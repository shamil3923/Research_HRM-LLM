import torch
from kaggle.hrm_gsm8k_pytorch import HRMForMath, OP_VOCAB

# Initialize the new Graph-Aware architecture (as defined in the notebook)
model = HRMForMath(
    vocab_size=len(OP_VOCAB), d_model=512, n_heads=8,
    H_cycles=4, L_cycles=8, H_layers=8, L_layers=8, seq_len=50,
)

print("Attempting to load best_model1.pt into the Graph-Aware architecture...")
try:
    sd = torch.load('checkpoints/gsm8k/best_model1.pt', map_location='cpu')
    if 'model_state' in sd:
        sd = sd['model_state']
    model.load_state_dict(sd)
    print("SUCCESS!")
except Exception as e:
    print("\nERROR LOADING MODEL:")
    print(str(e))
