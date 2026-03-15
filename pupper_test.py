import warp as wp
import warp.sim as wp_sim
import warp.sim.render

wp.init()
device = "cuda" if wp.is_cuda_available() else "cpu"

# 1. Setup Model
builder = wp_sim.ModelBuilder()
builder.default_shape_ke = 1e4
builder.default_shape_kd = 250.0
builder.default_shape_kf = 500.0
builder.default_shape_margin = 0.01

wp_sim.parse_urdf("urdf/standford_pupper.urdf", builder, floating=True)

model = builder.finalize(device)
model.ground = True

# 2. Setup State
state_in = model.state()
state_out = model.state()

state_in.body_q.numpy()[0] = [0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
state_in.body_q.assign(state_in.body_q.numpy())

# 3. Initialize the Integrator
integrator = wp_sim.XPBDIntegrator(iterations=10)

# 4. Setup Renderer
renderer = wp_sim.render.SimRenderer(model, "output4.usd", fps=60)

# 5. Simulation Loop
state_in.body_qd.zero_()

# warm up — let the integrator accept the initial state
import numpy as np
for _ in range(10):
    wp.sim.collide(model, state_in)
    integrator.simulate(model, state_in, state_out, dt=1.0/60.0)
    state_in, state_out = state_out, state_in

# now run the real loop
for i in range(100):
    state_in.clear_forces()
    wp.sim.collide(model, state_in)
    integrator.simulate(model, state_in, state_out, dt=1.0/60.0)
    state_in, state_out = state_out, state_in

    renderer.begin_frame(i / 60.0)
    renderer.render(state_in)
    renderer.end_frame()

renderer.save()