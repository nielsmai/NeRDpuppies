import warp as wp
import warp.sim as wp_sim
import warp.sim.render
import numpy as np
import os

wp.init()
device = "cuda" if wp.is_cuda_available() else "cpu"

# --- 1. MODEL BUILDING ---
builder = wp_sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))

# Baseline Justification: 
# Using a 1cm margin to prevent 'tunneling' through the floor
builder.default_shape_ke = 1.0e4      # Contact Stiffness
builder.default_shape_kd = 1.0e2      # Contact Damping
builder.default_shape_mu = 0.8       # Friction
builder.default_shape_margin = 0.01  # 1cm margin

wp_sim.parse_urdf(
    "/teamspace/studios/this_studio/urdf/standford_pupper_clean.urdf", 
    builder, 
    floating=True,
    armature=0.01, 
    collapse_fixed_joints=True
)

model = builder.finalize(device)
model.ground = True

# --- 2. STATE INITIALIZATION ---
state_in = model.state()
state_out = model.state()

# POSE: Set base at 0.2m (Drop test)
q = state_in.body_q.numpy()
q[0] = [0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 1.0] 
state_in.body_q.assign(q)

# JOINTS: Slightly bend the knees (0.4 rad) to avoid singularity/locking
# Based on your URDF order: indices 1, 4, 7, 10 are usually the pitch joints
joint_q = state_in.joint_q.numpy()
for i in [1, 4, 7, 10]:
    joint_q[i] = 0.4
state_in.joint_q.assign(joint_q)

# --- 3. SIMULATION & RENDERING ---
# Using 1/1000s step and 20 iterations for a high-precision diagnostic
integrator = wp_sim.XPBDIntegrator(iterations=20) 
renderer = wp_sim.render.SimRenderer(model, "pupper_diag.usd", fps=60)

print(f"{'Step':<8} | {'Z-Height':<12} | {'Vel_Z':<12}")
print("-" * 40)

for i in range(300):
    wp_sim.collide(model, state_in)
    integrator.simulate(model, state_in, state_out, dt=1.0/1000.0)
    state_in, state_out = state_out, state_in
    
    # Telemetry
    if i % 30 == 0:
        curr_q = state_in.body_q.numpy()[0]
        curr_qd = state_in.body_qd.numpy()[0]
        print(f"{i:<8} | {curr_q[2]:<12.4f} | {curr_qd[5]:<12.4f}")

    renderer.begin_frame(i * (1.0/1000.0))
    renderer.render(state_in)
    renderer.end_frame()

renderer.save()
print("-" * 40)
print("Diagnostic Complete. Please provide the Step/Z-Height/Vel_Z table.")