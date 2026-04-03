import warp as wp
import warp.sim as wp_sim
import warp.sim.render
import os

# ─── 0. Init ───────────────────────────────────────────────────────────────────
wp.init()
device = "cuda" if wp.is_cuda_available() else "cpu"
os.makedirs("outputs", exist_ok=True)

URDF_PATH = "/teamspace/studios/this_studio/puppersim/puppersim/puppersim/data/pupper_v2a.urdf"

# ─── 1. Build Model ────────────────────────────────────────────────────────────

# up_vector=(0,0,1): the URDF defines the robot in Z-up space,
# so we tell Warp to match that instead of its default Y-up
builder = wp_sim.ModelBuilder()

# contact parameters tuned for XPBD stability
builder.default_shape_ke     = 1e4    # contact stiffness
builder.default_shape_kd     = 100.0  # contact damping
builder.default_shape_kf     = 200.0  # friction stiffness
builder.default_shape_margin = 0.001  # collision detection margin

# floating=True: adds a free joint so the chassis can move in space
# density=0: use the inertia values from the URDF, not auto-computed from geometry
# armature=0.01: small regularization on joint inertia for numerical stability
wp_sim.parse_urdf(
    URDF_PATH,
    builder,
    floating=True,
    density=0,
    armature=0.01,
)

model = builder.finalize(device)

# adds infinite ground plane at z=0
model.ground = False

print(f"Bodies: {model.body_count}, Joints: {model.joint_count}")

# ─── 2. Initial State ──────────────────────────────────────────────────────────
state_in  = model.state()
state_out = model.state()

# zero all velocities so robot starts perfectly at rest
state_in.body_qd.zero_()

# body_q stores each body pose as [x, y, z, qx, qy, qz, qw]
# body 8 = pupper_v2_dji_chassis (confirmed from link order in URDF)
# z=0.17 comes from INIT_POSITION in pupper_constants.py
# quaternion [0,0,0,1] = identity = no rotation
bq = state_in.body_q.numpy()
bq[8] = [0.0, 0.0, 0.17, 0.0, 0.0, 0.0, 1.0]
state_in.body_q.assign(bq)

# ─── 3. Integrator ─────────────────────────────────────────────────────────────
# XPBDIntegrator: position-based solver, stable under contact
# SemiImplicitIntegrator exploded on contact with this robot
# iterations=10: constraint solver iterations per timestep
integrator = wp_sim.XPBDIntegrator(iterations=10)

# ─── 4. Renderer ───────────────────────────────────────────────────────────────
# SimRenderer writes USD — open in usdview, Omniverse, or Blender
renderer = wp_sim.render.SimRenderer(model, "/teamspace/studios/this_studio/outputs/pupper.usd", fps=60)

# ─── 5. Warmup ─────────────────────────────────────────────────────────────────
# 500 steps at dt=1/2000 lets the robot fall and settle on the ground
# before we start recording — avoids capturing the initial drop in the USD
print("Warming up...")
for i in range(500):
    wp.sim.collide(model, state_in)
    integrator.simulate(model, state_in, state_out, dt=1.0/2000.0)
    state_in, state_out = state_out, state_in
    if i % 100 == 0:
        bq = state_in.body_q.numpy()
        print(f"  warmup {i:4d} | chassis z={bq[8][2]:.4f}")

# ─── 6. Render Loop ────────────────────────────────────────────────────────────
# 200 frames at 60fps = ~3 seconds of animation
# 10 substeps per frame (dt=1/600) keeps physics stable at render framerate
print("Recording...")
sim_time  = 0.0
render_dt = 1.0 / 60.0
sim_dt    = 1.0 / 600.0
substeps  = int(render_dt / sim_dt)  # 10

for frame in range(200):
    for _ in range(substeps):
        state_in.clear_forces()
        wp.sim.collide(model, state_in)
        integrator.simulate(model, state_in, state_out, dt=sim_dt)
        state_in, state_out = state_out, state_in

    renderer.begin_frame(sim_time)
    renderer.render(state_in)
    renderer.end_frame()
    sim_time += render_dt

renderer.save()
print("Saved to outputs/ou.usd")