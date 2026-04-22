import numpy as np
data = np.load('a2_fixed.npz', allow_pickle=True)
for key in data.files:
    print(f"Key: {key}")
    content = data[key]
    print(f"Content Type: {type(content)}")
    if isinstance(content, np.ndarray):
        print(f"Content Shape: {content.shape}")