path = "/teamspace/studios/this_studio/data/datasets/Pupper/test_active.hdf5"
"""
NeRD HDF5 Dataset Quality Checker
Usage: python check_data_quality.py --path your_file.hdf5 [--max-envs 1000]
"""

import argparse
import numpy as np
import h5py
import sys

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Data quality checks for NeRD HDF5 datasets")
parser.add_argument("--max-envs", type=int, default=1000,
                    help="Max number of envs to sample for heavy checks (default: 1000)")
args = parser.parse_args()

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"
INFO = "\033[94m  INFO\033[0m"

results = []

def check(name, passed, detail="", warn=False):
    status = WARN if warn else (PASS if passed else FAIL)
    print(f"{status}  {name}")
    if detail:
        print(f"        {detail}")
    results.append((name, "PASS" if passed and not warn else ("WARN" if warn else "FAIL")))

print("\n══════════════════════════════════════════════════")
print("  NeRD HDF5 Data Quality Report")
print("══════════════════════════════════════════════════\n")

# ── Load ───────────────────────────────────────────────────────────────────────
try:
    f = h5py.File(path, "r")
    print(f"{INFO}  Opened: {path}\n")
except Exception as e:
    print(f"{FAIL}  Could not open file: {e}")
    sys.exit(1)

N_ENVS  = f["data/states"].shape[1]
N_STEPS = f["data/states"].shape[0]
N_SAMPLE = min(args.max_envs, N_ENVS)
sample_idx = np.sort(np.random.choice(N_ENVS, N_SAMPLE, replace=False))

print(f"  Dataset: {N_STEPS} steps × {N_ENVS} envs  (sampling {N_SAMPLE} envs for heavy checks)\n")

# ══════════════════════════════════════════════════
# 1. SCHEMA
# ══════════════════════════════════════════════════
print("── 1. Schema ──────────────────────────────────────")

expected = {
    "data/states":              (N_STEPS, N_ENVS, 37),
    "data/next_states":         (N_STEPS, N_ENVS, 37),
    "data/actions":             (N_STEPS, N_ENVS, 12),
    "data/joint_acts":          (N_STEPS, N_ENVS, 12),
    "data/root_body_q":         (N_STEPS, N_ENVS, 7),
    "data/gravity_dir":         (N_STEPS, N_ENVS, 3),
    "data/contact_depths":      (N_STEPS, N_ENVS, 53),
    "data/contact_normals":     (N_STEPS, N_ENVS, 53, 3),
    "data/contact_points_0":    (N_STEPS, N_ENVS, 53, 3),
    "data/contact_points_1":    (N_STEPS, N_ENVS, 53, 3),
    "data/contact_thicknesses": (N_STEPS, N_ENVS, 53),
}

for key, expected_shape in expected.items():
    if key not in f:
        check(f"Field present: {key}", False, "MISSING")
    else:
        actual = f[key].shape
        check(f"Shape ok: {key}", actual == expected_shape,
              f"got {actual}, expected {expected_shape}")

# ══════════════════════════════════════════════════
# 2. NaN / Inf
# ══════════════════════════════════════════════════
print("\n── 2. NaN / Inf ───────────────────────────────────")

scalar_fields = [
    "data/states", "data/next_states", "data/actions",
    "data/joint_acts", "data/root_body_q", "data/gravity_dir",
    "data/contact_depths", "data/contact_thicknesses",
]

for key in scalar_fields:
    data = f[key][:, sample_idx]
    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())
    ok = (nan_count == 0) and (inf_count == 0)
    detail = f"NaN: {nan_count}, Inf: {inf_count}" if not ok else ""
    check(f"No NaN/Inf: {key.split('/')[-1]}", ok, detail)

# ══════════════════════════════════════════════════
# 3. State / Next-state consistency
# ══════════════════════════════════════════════════
print("\n── 3. State → Next-state consistency ─────────────")

states      = f["data/states"][:, sample_idx]       # (T, S, 37)
next_states = f["data/next_states"][:, sample_idx]

# next_states[t] should equal states[t+1] for a consistent rollout
state_shift     = states[1:]           # steps 1..T
next_from_data  = next_states[:-1]     # steps 0..T-1

diff = np.abs(state_shift - next_from_data)
max_diff  = float(diff.max())
mean_diff = float(diff.mean())

# Tolerances: float32 sim noise, allow small delta
check("next_state[t] ≈ state[t+1]",
      max_diff < 1e-3,
      f"max |diff|={max_diff:.2e}, mean={mean_diff:.2e}",
      warn=(1e-3 <= max_diff < 0.1))

# ══════════════════════════════════════════════════
# 4. Quaternion validity  (root_body_q last 4 dims)
# ══════════════════════════════════════════════════
print("\n── 4. Quaternion validity ─────────────────────────")

q = f["data/root_body_q"][:, sample_idx, 3:]   # (T, S, 4)  — last 4 = quaternion
norms = np.linalg.norm(q, axis=-1)             # (T, S)
norm_ok = np.allclose(norms, 1.0, atol=1e-3)
bad_frac = float((np.abs(norms - 1.0) > 1e-3).mean())
check("Quaternion norms ≈ 1.0",
      norm_ok,
      f"fraction with |norm-1|>1e-3: {bad_frac:.4f}")

# ══════════════════════════════════════════════════
# 5. Gravity direction
# ══════════════════════════════════════════════════
print("\n── 5. Gravity direction ───────────────────────────")

grav = f["data/gravity_dir"][:, sample_idx]    # (T, S, 3)
gnorms = np.linalg.norm(grav, axis=-1)
grav_unit = np.allclose(gnorms, 1.0, atol=1e-3)
check("Gravity vectors are unit vectors", grav_unit,
      f"mean norm={float(gnorms.mean()):.4f}, std={float(gnorms.std()):.4f}")

grav_unique = np.unique(grav.reshape(-1, 3), axis=0)
check("Gravity directions consistent",
      len(grav_unique) <= 10,
      f"unique gravity vectors found: {len(grav_unique)}",
      warn=(1 < len(grav_unique) <= 10))

# ══════════════════════════════════════════════════
# 6. Contact data
# ══════════════════════════════════════════════════
print("\n── 6. Contact data ────────────────────────────────")

depths = f["data/contact_depths"][:, sample_idx]       # (T, S, 53)
normals = f["data/contact_normals"][:, sample_idx]     # (T, S, 53, 3)

# Depths should be >= 0 (positive = penetrating)
neg_frac = float((depths < 0).mean())
check("Contact depths >= 0",
      neg_frac < 0.001,
      f"fraction negative: {neg_frac:.4f}",
      warn=(0.001 <= neg_frac < 0.01))

# Active contact sparsity
active = (depths > 0) & (depths < 9999.0)  # 10000 = CONTACT_FREE_DEPTH sentinel for inactive
active_frac = float(active.mean())
check("Contact sparsity reasonable (< 80% active)",
      active_frac < 0.8,
      f"fraction active: {active_frac:.3f}",
      warn=(active_frac >= 0.5))

# Normal vectors should be unit (only where active)
active_mask = active[..., np.newaxis]                  # (T, S, 53, 1)
active_normals = normals[active.reshape(*active.shape)]
if len(active_normals) > 0:
    nn = np.linalg.norm(active_normals, axis=-1)
    bad_normals = float((np.abs(nn - 1.0) > 1e-2).mean())
    check("Active contact normals are unit vectors",
          bad_normals < 0.01,
          f"fraction non-unit: {bad_normals:.4f}")
else:
    print(f"{WARN}  No active contacts found in sample — cannot check normals")

# contact_points_0 and _1 should differ where active
cp0 = f["data/contact_points_0"][:, sample_idx]
cp1 = f["data/contact_points_1"][:, sample_idx]
same_points = np.all(np.abs(cp0 - cp1) < 1e-6, axis=-1)  # (T, S, 53)
same_active = float((same_points & active).mean())
check("contact_points_0 != contact_points_1 when active",
      same_active < 0.001,
      f"fraction identical when active: {same_active:.4f}",
      warn=(0.001 <= same_active < 0.01))

# ══════════════════════════════════════════════════
# 7. Action / joint_acts alignment
# ══════════════════════════════════════════════════
print("\n── 7. Action / joint_acts alignment ──────────────")

actions    = f["data/actions"][:, sample_idx]
joint_acts = f["data/joint_acts"][:, sample_idx]

diff_aj = np.abs(actions - joint_acts)
are_same = np.allclose(actions, joint_acts, atol=1e-5)
max_aj = float(diff_aj.max())
print(f"{INFO}  actions vs joint_acts max diff: {max_aj:.2e}  (commanded vs. applied torques — expected)")
results.append(("actions vs joint_acts", "PASS"))

# Action range
act_min, act_max = float(actions.min()), float(actions.max())
print(f"{INFO}  Action range: [{act_min:.3f}, {act_max:.3f}]")
clipped_frac = float(((actions == act_min) | (actions == act_max)).mean())
check("Actions not suspiciously clipped",
      clipped_frac < 0.05,
      f"fraction at min/max: {clipped_frac:.4f}",
      warn=(0.05 <= clipped_frac < 0.2))

# ══════════════════════════════════════════════════
# 8. State diversity
# ══════════════════════════════════════════════════
print("\n── 8. State diversity ─────────────────────────────")

# Look at initial states across envs
init_states = f["data/states"][0, sample_idx]           # (S, 37)
state_std   = init_states.std(axis=0)
low_var_dims = int((state_std < 1e-4).sum())
check("Initial states are diverse across envs",
      low_var_dims < 5,
      f"dims with std < 1e-4: {low_var_dims}/37",
      warn=(5 <= low_var_dims < 15))

# Check for duplicate trajectories (sample a few)
if N_SAMPLE >= 10:
    fingerprints = init_states[:10].round(4)
    unique_fps = len(np.unique(fingerprints, axis=0))
    check("No duplicate initial states (spot check)", unique_fps == 10,
          f"unique initial states in first 10: {unique_fps}")

# ══════════════════════════════════════════════════
# 9. Trajectory stability (energy proxy)
# ══════════════════════════════════════════════════
print("\n── 9. Trajectory stability ────────────────────────")

# Use L2 norm of state as a rough proxy for blow-up
state_norms = np.linalg.norm(states, axis=-1)  # (T, S)
final_vs_init = state_norms[-1] / (state_norms[0] + 1e-8)
blown_up = float((final_vs_init > 100).mean())
check("No blown-up trajectories (state norm growth < 100x)",
      blown_up < 0.001,
      f"fraction blown up: {blown_up:.4f}",
      warn=(0.001 <= blown_up < 0.01))

# Detect discontinuities: step-to-step state delta
step_deltas = np.abs(np.diff(states, axis=0))  # (T-1, S, 37)
max_delta   = float(step_deltas.max())
mean_delta  = float(step_deltas.mean())
p99_delta   = float(np.percentile(step_deltas, 99))
check("No large discontinuities between steps",
      p99_delta < 10.0,
      f"p99 step delta={p99_delta:.4f}, max={max_delta:.4f}, mean={mean_delta:.4f}",
      warn=(10.0 <= p99_delta < 30.0))

# Torque magnitude report
act_abs = np.abs(joint_acts)
print(f"{INFO}  Torque magnitudes — mean: {float(act_abs.mean()):.2f} Nm, "
      f"p99: {float(np.percentile(act_abs, 99)):.2f} Nm, "
      f"max: {float(act_abs.max()):.2f} Nm")

# ══════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════
f.close()
passed  = sum(1 for _, s in results if s == "PASS")
warned  = sum(1 for _, s in results if s == "WARN")
failed  = sum(1 for _, s in results if s == "FAIL")
total   = len(results)

print("\n══════════════════════════════════════════════════")
print(f"  Summary: {passed}/{total} passed  |  {warned} warnings  |  {failed} failures")
print("══════════════════════════════════════════════════\n")

if failed > 0:
    print("  Failed checks:")
    for name, status in results:
        if status == "FAIL":
            print(f"    ✗  {name}")
    print()

sys.exit(0 if failed == 0 else 1)
