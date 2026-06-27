import torch
import numpy as np

pt_path = 'checkpoints/gsm8k/best_model1.pt'
npz_path = 'checkpoints/gsm8k/best_model1.npz'

print(f"Loading {pt_path}...")
sd = torch.load(pt_path, map_location='cpu')

if 'model_state' in sd:
    sd = sd['model_state']

print(f"Found {len(sd)} keys. First 10:", list(sd.keys())[:10])

# Convert to numpy arrays
out_dict = {}
for k, v in sd.items():
    if isinstance(v, torch.Tensor):
        out_dict[k] = v.numpy()

print(f"Converting and saving to {npz_path}...")
np.savez(npz_path, **out_dict)
print("Done!")
