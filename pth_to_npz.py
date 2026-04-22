import torch
import numpy as np

# 1. Load the .pth file
# Use map_location='cpu' to ensure it loads even if you don't have a GPU
checkpoint = torch.load('/Users/cinnamon/Downloads/PupperPPO.pth', map_location='cpu')

# 2. Extract the weights (state_dict)
# Check if the file is a raw state_dict or a dictionary containing 'state_dict'
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# 3. Convert tensors to numpy arrays safely
numpy_weights = {}
for key, val in state_dict.items():
    if isinstance(val, torch.Tensor):
        numpy_weights[key] = val.detach().cpu().numpy()
    else:
        print(f"Skipping non-tensor key: {key} (Type: {type(val)})")

# 4. Save as .npz
np.savez('model_weights.npz', **numpy_weights)

print("Conversion complete: model_weights.npz created.")