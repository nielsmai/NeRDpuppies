import numpy as np

# 1. Load your current file
data = np.load('a2.npz', allow_pickle=True)

# 2. Extract and Flatten weights in the expected order
weights_to_combine = [
    data['w0'], data['b0'],
    data['w2'], data['b2'],
    data['w4'], data['b4'],
    data['mu_w'], data['mu_b']
]
flattened_weights = np.concatenate([w.flatten() for w in weights_to_combine])

# 3. Get Observation Statistics
mu = data['obs_mean']
# Ensure we use standard deviation (sqrt of variance)
std = np.sqrt(data['obs_var'] + 1e-8)

# 4. THE FIX FOR NUMPY 2.0+ "inhomogeneous shape" ERROR
# We create an empty object array of size 3 and manually insert the arrays.
# This prevents NumPy from trying to "guess" a uniform shape.
ars_fixed = np.empty(3, dtype=object)
ars_fixed[0] = flattened_weights
ars_fixed[1] = mu
ars_fixed[2] = std

# 5. Save it
# We use the key 'lin_policy_plus' so it's the first (and only) file in the npz
np.savez('a2_fixed_final.npz', lin_policy_plus=ars_fixed)

print("Success! a2_fixed_final.npz created.")