# HRM GSM8K — Kaggle GPU Training Guide

## Quick Start

### Step 1: Upload Files to Kaggle

Create a new **Kaggle Notebook** with **GPU T4 x2** enabled, then upload:

1. `kaggle/hrm_gsm8k_pytorch.py` — The training script
2. `data/gsm8k_train_clean.json` — Your cleaned dataset (~614KB)

### Step 2: Run in Kaggle Notebook

```python
# Cell 1: Setup
!pip install torch --quiet

# Cell 2: Upload your data
# Use Kaggle's "Add Data" button, or upload manually:
import os
os.makedirs("data", exist_ok=True)
# If uploaded as Kaggle dataset:
# !cp /kaggle/input/hrm-gsm8k/gsm8k_train_clean.json data/

# Cell 3: Train (default config — ~20M params, should reach 25-35%)
!python hrm_gsm8k_pytorch.py \
    --epochs 500 \
    --batch_size 128 \
    --d_model 512 \
    --H_cycles 4 \
    --L_cycles 8 \
    --peak_lr 3e-4

# Cell 4: Aggressive config (if T4 memory allows — ~45M params, target 40%+)
!python hrm_gsm8k_pytorch.py \
    --epochs 800 \
    --batch_size 64 \
    --d_model 768 \
    --n_heads 12 \
    --H_cycles 4 \
    --L_cycles 8 \
    --H_layers 10 \
    --L_layers 10 \
    --peak_lr 2e-4
```

### Step 3: Download Results

```python
# Cell 5: Check results
import json
with open("checkpoints/gsm8k_gpu/best_meta.json") as f:
    print(json.dumps(json.load(f), indent=2))

# Cell 6: Download checkpoint
from IPython.display import FileLink
FileLink("checkpoints/gsm8k_gpu/best_model.pt")
```

---

## Architecture Comparison

| Parameter | MLX (M4 Mac) | PyTorch Default | PyTorch Aggressive |
|---|---|---|---|
| d_model | 256 | **512** | **768** |
| H_cycles | 2 | **4** | **4** |
| L_cycles | 4 | **8** | **8** |
| H_layers | 4 | **8** | **10** |
| L_layers | 4 | **8** | **10** |
| batch_size | 32 | **128** | **64** |
| Parameters | ~5.8M | **~20M** | **~45M** |
| Mixed Precision | No | **Yes (fp16)** | **Yes (fp16)** |
| Optimizer | Adam | **AdamW + OneCycleLR** | **AdamW + OneCycleLR** |
| Expected Accuracy | 12.8% | **25-35%** | **35-45%** |

## Key Improvements in PyTorch Version

1. **Mixed Precision (fp16)**: 2x memory savings + faster matmul on GPU
2. **AdamW + OneCycleLR**: Better generalization than flat LR
3. **Larger model**: 512-768 d_model vs 256 — more representational capacity
4. **More reasoning cycles**: H=4, L=8 gives the model more "thinking time"
5. **Pin memory + DataLoader workers**: Faster data pipeline
6. **Gradient accumulation ready**: Can simulate larger batches if needed

## Kaggle GPU Quotas

- **Free tier**: 30 hours/week of T4 GPU
- **T4 GPU**: 16GB VRAM — fits d_model=512 easily, d_model=768 with batch_size=64
- **Training time estimate**: 500 epochs ≈ 2-4 hours on T4
