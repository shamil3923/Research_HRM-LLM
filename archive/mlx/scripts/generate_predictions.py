import json
import torch
import numpy as np

import sys
sys.path.append(".")
from src.serve_ui import load_model, run_inference, MODEL

def main():
    print("Loading model...")
    load_model()
    
    print("Loading data...")
    with open("data/gsm8k_train_clean.json", "r") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} samples.")
    
    samples_to_run = min(197, len(data))
    predictions = []
    
    correct = 0
    for i in range(samples_to_run):
        item = data[i]
        trace = item.get("trace", {})
        if not trace:
            continue
            
        result = run_inference(trace)
        if result:
            result["question"] = item.get("question", f"Question {i+1}")
            result["explanation"] = item.get("explanation", "")
            if result.get("correct", False):
                correct += 1
            predictions.append(result)
            
        if (i+1) % 20 == 0:
            print(f"Processed {i+1}/{samples_to_run} | Accuracy: {correct/(i+1)*100:.1f}%")
            
    acc = correct/len(predictions)
    print(f"Final Accuracy on {len(predictions)} samples: {acc*100:.1f}%")
    
    output = {
        "model_params": "37.6M",
        "total": len(predictions),
        "accuracy": acc,
        "samples": predictions
    }
    
    with open("ui/predictions.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print("Saved ui/predictions.json!")

if __name__ == "__main__":
    main()
