"""
Convert PupperPPO.pth -> pupper_policy.tflite
Directly builds a TF model from weights — no onnx2tf, no tf2onnx needed.

Usage:
    python3 convert_to_tflite.py --pth PupperPPO.pth --out pupper_policy.tflite

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install tensorflow numpy
"""

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pth", required=True)
parser.add_argument("--out", default="pupper_policy.tflite")
args = parser.parse_args()

# ── Step 1: Load weights from .pth ───────────────────────────────────────────
print("\n[1/3] Loading weights from", args.pth)
import torch

ck = torch.load(args.pth, map_location="cpu", weights_only=False)
sd = ck["model"]

def t(k):
    return sd[k].float().numpy()

obs_mean = t("_orig_mod.running_mean_std.running_mean")
obs_std  = np.sqrt(t("_orig_mod.running_mean_std.running_var") + 1e-8)
w0 = t("_orig_mod.a2c_network.actor_mlp.0.weight")
b0 = t("_orig_mod.a2c_network.actor_mlp.0.bias")
w2 = t("_orig_mod.a2c_network.actor_mlp.2.weight")
b2 = t("_orig_mod.a2c_network.actor_mlp.2.bias")
w4 = t("_orig_mod.a2c_network.actor_mlp.4.weight")
b4 = t("_orig_mod.a2c_network.actor_mlp.4.bias")
mu_w = t("_orig_mod.a2c_network.mu.weight")
mu_b = t("_orig_mod.a2c_network.mu.bias")

print("  obs_dim:", obs_mean.shape[0])
print("  ac_dim: ", mu_b.shape[0])

# ── Step 2: Build TF model directly from weights ──────────────────────────────
print("\n[2/3] Building TensorFlow model")
import tensorflow as tf

class PupperActor(tf.Module):
    def __init__(self):
        super().__init__()
        # Normalizer
        self.obs_mean = tf.constant(obs_mean, dtype=tf.float32)
        self.obs_std  = tf.constant(obs_std,  dtype=tf.float32)
        # Weights — note TF Linear is x @ W + b, so transpose vs PyTorch
        self.w0  = tf.constant(w0.T,  dtype=tf.float32)
        self.b0  = tf.constant(b0,    dtype=tf.float32)
        self.w2  = tf.constant(w2.T,  dtype=tf.float32)
        self.b2  = tf.constant(b2,    dtype=tf.float32)
        self.w4  = tf.constant(w4.T,  dtype=tf.float32)
        self.b4  = tf.constant(b4,    dtype=tf.float32)
        self.wmu = tf.constant(mu_w.T, dtype=tf.float32)
        self.bmu = tf.constant(mu_b,   dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 37], dtype=tf.float32)])
    def __call__(self, obs):
        x = (obs - self.obs_mean) / self.obs_std
        x = tf.clip_by_value(x, -5.0, 5.0)
        x = tf.nn.relu(tf.matmul(x, self.w0) + self.b0)
        x = tf.nn.relu(tf.matmul(x, self.w2) + self.b2)
        x = tf.nn.relu(tf.matmul(x, self.w4) + self.b4)
        return tf.matmul(x, self.wmu) + self.bmu

model = PupperActor()

# Verify matches numpy dry run
test_input = np.zeros((1, 37), dtype=np.float32)
test_input[0, 2]  = -1.0
test_input[0, 10] =  0.8
test_input[0, 11] = -1.6
tf_out = model(tf.constant(test_input)).numpy()
print("  TF test output:", tf_out.round(4))

# ── Step 3: Convert to TFLite ─────────────────────────────────────────────────
print("\n[3/3] Converting to TFLite:", args.out)
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [model.__call__.get_concrete_function()], model
)
tflite_model = converter.convert()

with open(args.out, "wb") as f:
    f.write(tflite_model)
print("  Saved:", args.out)
print("  Size:", round(os.path.getsize(args.out) / 1024, 1), "KB")

# ── Verify TFLite output ──────────────────────────────────────────────────────
print("\n[Verify] Checking TFLite output...")
interpreter = tf.lite.Interpreter(model_path=args.out)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()

interpreter.set_tensor(inp[0]["index"], test_input)
interpreter.invoke()
tflite_out = interpreter.get_tensor(out[0]["index"])

print("  TF output:     ", tf_out.round(4))
print("  TFLite output: ", tflite_out.round(4))
diff = np.abs(tf_out - tflite_out).max()
print("  Max diff:", diff)
if diff < 1e-4:
    print("  MATCH - conversion successful!")
else:
    print("  WARNING - outputs differ")

print("\nDone! Copy", args.out, "to the Pi.")
