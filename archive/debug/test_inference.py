import json
import torch
import numpy as np

import sys
sys.path.append(".")
from src.serve_ui import load_model, run_inference, MODEL

load_model()
with open("data/gsm8k_train_clean.json", "r") as f:
    data = json.load(f)

for i in range(5):
    item = data[i]
    trace = item.get("trace", {})
    result = run_inference(trace)
    print(f"True: {result['computed_answer']} | Pred: {result['predicted_answer']} | Digits: {result['pred_digits']}")

