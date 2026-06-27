import torch
sd = torch.load('checkpoints/gsm8k/best_model.pt', map_location='cpu')
if 'config' in sd:
    print("Config:", sd['config'])
if 'model_state' in sd:
    print("First 20 model_state keys:", list(sd['model_state'].keys())[:20])
else:
    print("First 20 keys:", list(sd.keys())[:20])
