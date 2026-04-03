import warp as wp
import warp.sim as wp_sim
import warp.sim.render
import numpy as np

# 0. Initialize
wp.init()
device = "cuda" if wp.is_cuda_available() else "cpu"

# 1. Setup Model
builder = wp_sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))
builder.default_shape_margin = 0.001
wp_sim.parse_urdf("/teamspace/studios/this_studio/urdf/standford_pupper_clean.urdf", builder, floating=True)
model = builder.finalize(device)

# === STRUCTURE (before simulation) ===
print("=== STRUCTURE ===")
print("body count:", model.body_count)
print("shape count:", model.shape_count)


# === INTEGRATOR CHECK ===
print("\n=== INTEGRATOR CHECK ===")
state_test = model.state()
print("joint_q available:", hasattr(state_test, 'joint_q') and state_test.joint_q is not None)
print("body_q available:", hasattr(state_test, 'body_q') and state_test.body_q is not None)

model.ground = True
model.joint_ke = 1000.0
model.joint_kd = 100.0

# 2. Setup State
state_in = model.state()
state_out = model.state()

def set_base_pose(state, x, y, z, roll_deg=0):
    q = state.body_q.numpy()
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    if roll_deg == 180:
        qx, qy, qz, qw = 1.0, 0.0, 0.0, 0.0
    q[0] = [x, y, z, qx, qy, qz, qw]
    state.body_q.assign(q)

set_base_pose(state_in, 0.0, 0.0, 0.4, roll_deg=0)
state_in.body_qd.zero_()

# 3. Simulation Settings
integrator = wp_sim.XPBDIntegrator(iterations=30)
renderer = wp_sim.render.SimRenderer(model, "outputs/test_local_kat.usd", fps=60)

# 4. Simulation Loop
dt = 1.0 / 120.0
for i in range(200):
    state_in.clear_forces()
    wp.sim.collide(model, state_in)
    integrator.simulate(model, state_in, state_out, dt)
    state_in, state_out = state_out, state_in

    # Body states at step 0
    if i == 0:
        q = state_in.body_q.numpy()
        print("\n=== BODY STATES AT STEP 0 ===")
        for j in range(model.body_count):
            print(f"  body[{j}] pos={q[j][:3]}  quat={q[j][3:]}")

    # Log every 20 steps
    if i % 20 == 0:
        curr_q = state_in.body_q.numpy()
        print(f"Step {i:3d} | Body Z: {curr_q[0][2]:.4f}")

    # Stabilized state at step 160
    if i == 160:
        q = state_in.body_q.numpy()
        print("\n=== STABILIZED HEIGHT ===")
        print(f"  base z = {q[0][2]:.4f}")

        depths = model.rigid_contact_depth.numpy()
        print(f"\n=== CONTACTS ===")
        print(f"  contact depth array shape: {depths.shape}")
        print(f"  depths: {depths}")

        # joint_q if available
        if hasattr(state_in, 'joint_q') and state_in.joint_q is not None:
            state_gen_q = state_in.joint_q.numpy()
            print("\n=== JOINT_Q ===")
            print(f"  shape: {state_gen_q.shape}")
            print(f"  values: {state_gen_q}")
        else:
            print("\n=== JOINT_Q: not available (XPBD uses maximal coords) ===")

    # Render
    renderer.begin_frame(i * dt)
    renderer.render(state_in)
    renderer.end_frame()

renderer.save()
print("Done! Check outputs/test_local_kat.usd")