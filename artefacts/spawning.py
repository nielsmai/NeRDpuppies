#!/usr/bin/env python3
"""Minimal test: verify Warp imports and basic simulation works."""

# =============================================================================
# IMPORTS (THIS WAS MISSING)
# =============================================================================
import warp as wp          # ← Critical!
import warp.sim
import numpy as np
import os

# =============================================================================
# INIT
# =============================================================================
print(f"🔧 Initializing Warp...")
wp.init()

# Use GPU if available, fallback to CPU
device = "cuda" if wp.is_cuda_available() else "cpu"
print(f"✅ Device: {device}")

# =============================================================================
# BUILD SIMPLE MODEL (No URDF - just a falling box)
# =============================================================================
print(f"\n🔧 Building simple test model...")
builder = warp.sim.ModelBuilder()

# Add a single body (box) at height 1.0m
body_idx = builder.add_body(
    xform=wp.transform([0.0, 0.0, 1.0], wp.quat_identity())
)

# Add box collision
builder.add_shape_box(
    body=body_idx,
    xform=wp.transform(),
    hx=0.1, hy=0.1, hz=0.1,  # 20cm cube
    density=1000.0,          # 1 kg
)

# Finalize model
model = builder.finalize(device)
model.ground = True
model.gravity = wp.vec3(0.0, 0.0, -9.81)

print(f"✅ Model built: 1 body, ground enabled")

# =============================================================================
# SIMULATE: Let the box fall
# =============================================================================
print(f"\n=== Simulating: Box drop test ===")
state_in = model.state()
state_out = model.state()

# Initialize FK
wp.sim.eval_fk(model, state_in.joint_q, state_in.joint_qd, None, state_in)
wp.sim.collide(model, state_in)

integrator = warp.sim.XPBDIntegrator(iterations=4)
dt = 1.0/60.0

for step in range(60):  # 1 second
    wp.sim.collide(model, state_in)
    integrator.simulate(model, state_in, state_out, dt=dt)
    state_in, state_out = state_out, state_in
    
    pos = state_in.body_q.numpy()[0][:3]
    if step % 10 == 0:
        print(f"Step {step:2d}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

final_z = state_in.body_q.numpy()[0][2]
print(f"\n✅ Final Z: {final_z:.4f}m")

if final_z < 0.9:
    print("🎉 Box fell under gravity — Warp is working!")
else:
    print("⚠️  Box didn't fall — check gravity/collision settings")

print("=" * 70)