import os
import h5py
import numpy as np
import warp as wp
import warp.sim as wp_sim
import warp.sim.render

# 0. Initialize
wp.init()
device = "cuda" if wp.is_cuda_available() else "cpu"

# 1. Setup Model
# We use collapse_fixed_joints=True to match your 12-joint dataset
builder = wp_sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))
wp_sim.parse_urdf(
    "/teamspace/studios/this_studio/urdf/standford_pupper_clean.urdf", 
    builder, 
    floating=True,
    collapse_fixed_joints=True 
)

model = builder.finalize(device)
model.ground = True # Turn off ground to prevent playback jitter

# 2. Setup State and Load Data
state_in = model.state()

DATASET_PATH = "/teamspace/studios/this_studio/data/datasets/Pupper/test_active.hdf5"
with h5py.File(DATASET_PATH, 'r') as f:
    # Get first trajectory: [100 frames, 1 robot, 37 dims]
    traj_data = f['data']['states'][:, 0, :].astype(np.float32)

# 3. Setup Renderer
renderer = wp_sim.render.SimRenderer(model, "outputs/dataset_playback.usd", fps=60)

# 4. Playback Loop
print(f"🎬 Playing back {len(traj_data)} frames from dataset...")

for i in range(len(traj_data)):
    # Extract the first 19 values: [Pos(3), Quat(4), Joints(12)]
    q_frame = traj_data[i, :19]
    
    # Optional: Force height if the data is "floating" or "sinking"
    q_frame[2] = 0.25 
    
    # Manually assign the dataset pose to the state
    state_in.joint_q.assign(q_frame)
    state_in.joint_qd.zero_() # No velocity needed for pure playback
    
    # CRITICAL: Forward Kinematics updates the body parts based on joint_q
    # Without this, the legs won't move in the render
    wp.sim.eval_fk(model, state_in.joint_q, state_in.joint_qd, None, state_in)
    
    # Render
    renderer.begin_frame(i * (1.0/60.0))
    renderer.render(state_in)
    renderer.end_frame()

renderer.save()
print("✅ Done! Check outputs/dataset_playback.usd")
