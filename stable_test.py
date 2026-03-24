import warp as wp
import warp.sim as wp_sim
import warp.sim.render
import numpy as np

# 0. Initialize
wp.init()
device = "cuda" if wp.is_cuda_available() else "cpu"

# 1. Setup Model
builder = wp_sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))
builder.default_shape_margin = 0.001  # Smaller margin for stability

# Load your simplified box URDF
wp_sim.parse_urdf("urdf/standford_pupper.urdf", builder, floating=True)

model = builder.finalize(device)
model.ground = True

# Joint PD Control (Stiffness keeps it from falling to its side)
model.joint_ke = 1000.0  # Spring stiffness
model.joint_kd = 100.0   # Damping (prevents jitter)

# 2. Setup State
state_in = model.state()
state_out = model.state()

# Helper function to set position + rotation safely
def set_base_pose(state, x, y, z, roll_deg=0):
    q = state.body_q.numpy()
    
    # Standard quaternion for "No Rotation" (x, y, z, w)
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    
    if roll_deg == 180:
        # Flip 180 degrees around X-axis
        qx, qy, qz, qw = 1.0, 0.0, 0.0, 0.0
        
    q[0] = [x, y, z, qx, qy, qz, qw]
    state.body_q.assign(q)

# Initialize pose: Start at 0.4m height to avoid ground collision
set_base_pose(state_in, 0.0, 0.0, 0.4, roll_deg=0)
state_in.body_qd.zero_() # Ensure zero initial velocity

# 3. Simulation Settings
integrator = wp_sim.XPBDIntegrator(iterations=30) # More iterations = more stable
renderer = wp_sim.render.SimRenderer(model, "outputs/stable_box.usd", fps=60)

# 4. Simulation Loop
dt = 1.0 / 120.0 # Standard simulation step

for i in range(200):
    state_in.clear_forces()
    wp.sim.collide(model, state_in)
    
    # The simulation step
    integrator.simulate(model, state_in, state_out, dt)
    
    # Swap states
    state_in, state_out = state_out, state_in
    
    # Log every 20 steps
    if i % 20 == 0:
        curr_q = state_in.body_q.numpy()
        print(f"Step {i:3d} | Body Z: {curr_q[0][2]:.4f}")

    # Render
    renderer.begin_frame(i * dt)
    renderer.render(state_in)
    renderer.end_frame()

renderer.save()
print("Done! Check outputs/stable_box.usd")